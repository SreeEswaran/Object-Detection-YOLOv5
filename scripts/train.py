import torch
import yaml
from torch.utils.data import DataLoader
from models.yolov5 import YOLOv5
from utils.dataset import YOLODataset

def train(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    train_dataset = YOLODataset(config['train']['images'], config['train']['labels'])
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)

    model = YOLOv5()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    criterion = torch.nn.BCELoss()

    model.train()
    for epoch in range(config['train']['epochs']):
        for images, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{config['train']['epochs']}], Loss: {loss.item()}")

    torch.save(model.state_dict(), 'models/yolov5_model.pth')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train YOLOv5 model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    train(args.config)
