from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.seqtrack_utils import sample_target, transform_image_to_crop
import cv2
from lib.utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh
from lib.models.seqtrack import build_seqtrack
from lib.test.tracker.seqtrack_utils import Preprocessor
from lib.utils.box_ops import clip_box
import numpy as np


class SEQTRACK(BaseTracker):
    def __init__(self, params, dataset_name):
        super(SEQTRACK, self).__init__(params)
        network = build_seqtrack(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.seq_format = self.cfg.DATA.SEQ_FORMAT
        self.num_template = self.cfg.TEST.NUM_TEMPLATES
        self.bins = self.cfg.MODEL.BINS
        if self.cfg.TEST.WINDOW == True: # for window penalty
            self.hanning = torch.tensor(np.hanning(self.bins)).unsqueeze(0).cuda()
            self.hanning = self.hanning
        else:
            self.hanning = None
        self.start = self.bins + 1 # start token
        self.network = network.cuda()
        self.network.eval()
        self.network.requires_grad_(False)
        self.preprocessor = Preprocessor()
        self.state = None
        self.debug = params.debug
        self.frame_id = 0

        # online update settings
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS.DEFAULT
        print("Update interval is: ", self.update_intervals)
        if hasattr(self.cfg.TEST.UPDATE_THRESHOLD, DATASET_NAME):
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD[DATASET_NAME]
        else:
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD.DEFAULT
        print("Update threshold is: ", self.update_threshold)



    def initialize(self, image, info: dict):

        # get the initial templates
        z_patch_arr, _ = sample_target(image, info['init_bbox'], self.params.template_factor,
                                       output_sz=self.params.template_size)

        template = self.preprocessor.process(z_patch_arr)
        self.template_list = [template] * self.num_template

        # get the initial sequence i.e., [start]
        batch = template.shape[0]
        self.init_seq = (torch.ones([batch, 1]).to(template) * self.start).type(dtype=torch.int64)

        self.state = info['init_bbox']
        self.frame_id = 0

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor = sample_target(image, self.state, self.params.search_factor,
                                                   output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr)
        images_list = self.template_list + [search]
        sweep = getattr(self.params, 'encoder_sweep', False)
        full_depth = len(self.network.encoder.body.blocks)
        latency_ms = None

        with torch.no_grad():
            if sweep:
                depths = list(getattr(self.params, 'sweep_depths', range(1, full_depth + 1)))
                if full_depth not in depths or len(depths) != len(set(depths)):
                    raise ValueError('sweep_depths must be unique and include the full encoder depth')
                depths.sort()
                features = self.network.forward_encoder_sweep(images_list, depths)
                depth_results = {}
                gt_bbox = (info or {}).get('gt_bbox')
                for depth in depths:
                    decoded = self.network.inference_decoder(
                        xz=[features[depth]], sequence=self.init_seq,
                        window=self.hanning, seq_format=self.seq_format)
                    box = self._decode_output_to_box(decoded, resize_factor, H, W)
                    result = {'box': [float(v) for v in box]}
                    if gt_bbox is not None:
                        result['iou'] = self._xywh_iou(box, gt_bbox)
                    conf = decoded['confidence'][0].tolist()
                    conf_indices = (2, 3, 0, 1) if self.seq_format == 'whxy' else (0, 1, 2, 3)
                    result.update(zip(('conf_x', 'conf_y', 'conf_w', 'conf_h'),
                                      (float(conf[i]) for i in conf_indices)))
                    result['conf_mean'] = float(np.mean(conf))
                    result['conf_min'] = float(np.min(conf))
                    tokens = decoded['pred_boxes'][0].tolist()
                    result.update(zip(('token_x', 'token_y', 'token_w', 'token_h'),
                                      (int(tokens[i]) for i in conf_indices)))
                    depth_results[depth] = result
                for d1, d2 in zip(depths[:-1], depths[1:]):
                    if 'iou' in depth_results[d1] and 'iou' in depth_results[d2]:
                        depth_results[d1][f'delta_to_{d2}'] = (
                            depth_results[d2]['iou'] - depth_results[d1]['iou'])
                out_dict = decoded  # depths are sorted; the last output is full depth
                pred_box = depth_results[full_depth]['box']
            else:
                depth = getattr(self.params, 'encoder_depth', full_depth)
                if depth == 12 and full_depth != 12 and not getattr(self.params, 'encoder_depth_explicit', False):
                    depth = full_depth
                measure = getattr(self.params, 'measure_encoder_latency', False) and torch.cuda.is_available()
                if measure:
                    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    torch.cuda.synchronize()
                    start.record()
                xz = self.network.forward_encoder_depth(images_list, depth)
                out_dict = self.network.inference_decoder(
                    xz=xz, sequence=self.init_seq,
                    window=self.hanning, seq_format=self.seq_format)
                if measure:
                    end.record()
                    end.synchronize()
                    latency_ms = start.elapsed_time(end)
                pred_box = self._decode_output_to_box(out_dict, resize_factor, H, W)

        self.state = pred_box
        conf_score = out_dict['confidence'].sum().item() * 10
        if self.num_template > 1:
            if (self.frame_id % self.update_intervals == 0) and (conf_score > self.update_threshold):
                z_patch_arr, _ = sample_target(image, self.state, self.params.template_factor,
                                               output_sz=self.params.template_size)
                template = self.preprocessor.process(z_patch_arr)
                self.template_list.append(template)
                if len(self.template_list) > self.num_template:
                    self.template_list.pop(1)

        if self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1+w), int(y1+h)), color=(0,0,255), thickness=2)
            cv2.imshow('vis', image_BGR)
            cv2.waitKey(1)

        result = {'target_bbox': self.state, 'best_score': conf_score}
        if sweep:
            result['sweep_results'] = depth_results
        if latency_ms is not None:
            result['encoder_decoder_latency_ms'] = latency_ms
        return result

    def _decode_output_to_box(self, out_dict, resize_factor, H, W):
        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        if self.seq_format == 'corner':
            pred_boxes = box_xyxy_to_cxcywh(pred_boxes)
        if self.seq_format == 'whxy':
            pred_boxes = pred_boxes[:, [2, 3, 0, 1]]
        pred_boxes = pred_boxes / (self.bins-1)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        return clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=1)

    @staticmethod
    def _xywh_iou(box1, box2):
        x1, y1, w1, h1 = [float(v) for v in box1]
        x2, y2, w2, h2 = [float(v) for v in box2]
        if not np.isfinite([x1, y1, w1, h1, x2, y2, w2, h2]).all():
            return float('nan')
        iw = max(0.0, min(x1+w1, x2+w2) - max(x1, x2))
        ih = max(0.0, min(y1+h1, y2+h2) - max(y1, y2))
        union = max(0.0, w1)*max(0.0, h1) + max(0.0, w2)*max(0.0, h2) - iw*ih
        return iw*ih/union if union > 0 else 0.0

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return SEQTRACK
