# Object-Detection-YOLOv5

This project demonstrates object detection using YOLOv5. The model is trained on a custom dataset and can detect objects in new images. YOLOv5 is a state-of-the-art object detection model known for its speed and accuracy, making it suitable for real-time applications.

## Dataset

You can use any dataset that follows the YOLO format. For this project, we assume the dataset is organized as follows:


### Download Dataset

You can download a sample dataset from [Roboflow's Public Object Detection Datasets](https://public.roboflow.com/object-detection). Extract the dataset into the `data/` directory.

## Model Architecture

YOLOv5 is a convolutional neural network (CNN) designed for object detection. It divides the image into a grid and predicts bounding boxes and class probabilities for each grid cell.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/SreeEswaran/Object-Detection-YOLOv5.git
    cd Object-Detection-YOLOv5
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download and preprocess the dataset as described above.

## Usage

### Training

To train the model, run:
```bash
python scripts/train.py --config config.yaml
```

### To evaluate the model, run:
```bash
python scripts/evaluate.py --config config.yaml
```
### To infer, run
```bash
python scripts/infer.py --image_path path/to/your/image.jpg --model_path models/yolov5_model.pth
```




