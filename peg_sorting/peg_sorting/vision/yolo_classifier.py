"""Optional YOLOv8 shape classifier (pip install ultralytics).

Enable in vision.yaml:  shape_backend: yolo,  yolo.model_path: <best.pt>
The colour + depth pipeline still finds the pegs and their 3-D position; YOLO
only provides the shape label (matched to each colour blob by box overlap).
Train a model with tools/capture_dataset.py + tools/train_yolo.py.
"""


class YoloShapeClassifier:
    def __init__(self, cfg):
        from ultralytics import YOLO
        path = cfg.get('model_path')
        if not path:
            raise ValueError('vision.yaml: yolo.model_path is empty')
        self.model = YOLO(path)
        self.class_names = cfg.get('class_names') or None
        self.conf = float(cfg.get('conf', 0.25))

    def detect(self, bgr):
        out = []
        res = self.model.predict(bgr, conf=self.conf, verbose=False)[0]
        names = res.names
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            cls = int(b.cls[0])
            name = names.get(cls, str(cls)) if isinstance(names, dict) else names[cls]
            out.append({'shape': name.lower(), 'confidence': float(b.conf[0]),
                        'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1))})
        return out
