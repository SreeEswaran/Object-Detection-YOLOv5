import torch
import cv2
from models.yolov5 import YOLOv5

def infer(image_path, model_path):
    model = YOLOv5()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Infer using YOLOv5 model.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    args = parser.parse_args()
    infer(args.image_path, args.model_path)
