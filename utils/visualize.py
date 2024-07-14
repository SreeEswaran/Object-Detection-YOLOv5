import cv2
import matplotlib.pyplot as plt

def visualize(image, outputs):
    for box in outputs:
        x1, y1, x2, y2 = box[:4]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    plt.imshow(image)
    plt.show()
