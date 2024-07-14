import torch
import yaml
from torch.utils.data import DataLoader
from models.yolov5 import YOLOv5
from utils.dataset import YOLODataset

def evaluate(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    val_dataset = YOLODataset(config['val']['images'], config['val']['labels'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = YOLOv5()
    model.load_state_dict(torch.load('models/yolov5_model.pth'))
    model.eval()

    total, correct = 0, 0
    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images)    
    print(f'Accuracy: {correct / total}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate YOLOv5 model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    evaluate(args.config)
