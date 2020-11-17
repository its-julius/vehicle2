import numpy as np
import torch

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deepsort_torch.feature_extractor import Extractor


class DeepSort:
    def __init__(self, model_path, max_cosine_dist=0.2, min_confidence=0.3,
                 nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70,
                 n_init=3, nn_budget=100, use_cuda=True, model_type='torch'):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        if model_type == 'tensorflow' or model_type == 'tf':
            self.encoder = gdet.create_box_encoder(model_path, batch_size=1)
            self.model_type = 'tf'
        elif model_type == 'torch' or model_type == 'pytorch':
            self.extractor = Extractor(model_path, use_cuda=use_cuda)
            self.model_type = 'torch'
        elif model_type == 'darknet':
            raise ValueError('I love darknet, but it is not supported yet.')
        else:
            raise ValueError('model type is not supported yet.')

        metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_dist, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox, confidence, cls, img, bbox_type='xyxy'):
        self.height, self.width = img.shape[:2]

        if bbox_type == 'xyxy' or bbox_type == 'tlbr':
            bbox_tlwh = self._xyxy_to_tlwh(bbox)
        elif bbox_type == 'xywh':
            bbox_tlwh = self._xywh_to_tlwh(bbox)
        elif bbox_type == 'tlwh':
            bbox_tlwh = bbox
        else:
            raise ValueError('bbox type unknown.')

        if self.model_type == 'tf':
            features = self.encoder(img, bbox_tlwh)
        elif self.model_type == 'torch':
            features = self._get_features(img, bbox_tlwh)
        else:
            raise ValueError('model type is not supported yet.')

        # Generate Detection
        detections = [Detection(b, conf, c, f) for b, conf, c, f in
                      zip(bbox_tlwh, confidence, cls, features)]
        # Run Non-Maximum Suppression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update Tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # Output
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox_result = track.to_tlbr()
            id_result = track.track_id
            cls_result = track.cls
            outputs.append(np.array([bbox_result[0], bbox_result[1], bbox_result[2], bbox_result[3],
                                     id_result, cls_result], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    def _xywh_to_tlwh(self, xywh):
        if isinstance(xywh, np.ndarray):
            tlwh = xywh.copy()
        elif isinstance(xywh, torch.Tensor):
            tlwh = xywh.clone()
        else:
            raise ValueError('xywh type unknown.')
        tlwh[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        tlwh[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        return tlwh

    def _xyxy_to_tlwh(self, xyxy):
        if isinstance(xyxy, np.ndarray):
            tlwh = xyxy.copy()
        elif isinstance(xyxy, torch.Tensor):
            tlwh = xyxy.clone()
        else:
            raise ValueError('xyxy type unknown.')
        tlwh[:, 2] = abs(xyxy[:, 2] - xyxy[:, 0])
        tlwh[:, 3] = abs(xyxy[:, 1] - xyxy[:, 3])
        return tlwh

    def _get_features(self, img, bbox_tlwh):
        im_crops = []
        for box in bbox_tlwh:
            im = img[box[1]: box[1] + box[3], box[0]: box[0] + box[2]]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features
