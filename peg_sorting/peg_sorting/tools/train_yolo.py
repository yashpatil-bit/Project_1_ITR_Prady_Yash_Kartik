"""Train a YOLOv8 peg-shape detector (pip install ultralytics; a GPU helps but is not required).

    ros2 run peg_sorting train_yolo --data ~/peg_dataset/data.yaml --epochs 60
Then set in vision.yaml:  shape_backend: yolo   yolo.model_path: <printed best.pt path>
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.expanduser('~/peg_dataset/data.yaml'))
    ap.add_argument('--model', default='yolov8n.pt')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--project', default=os.path.expanduser('~/peg_yolo_runs'))
    a, _ = ap.parse_known_args()
    from ultralytics import YOLO
    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, project=a.project, name='pegs')
    best = os.path.join(a.project, 'pegs', 'weights', 'best.pt')
    print('\nDone. Best weights (newest run folder may be pegs2, pegs3...): %s' % best)


if __name__ == '__main__':
    main()
