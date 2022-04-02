import cv2
import torch
import torch.nn as nn
from tensorflow import expand_dims, convert_to_tensor, uint8
import numpy as np
import matplotlib.pylab as plt
import torchvision.models as models
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')


def get_model(path):
    model = models.alexnet(pretrained=True)
    model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2)
        )
    model.load_state_dict(torch.load(path))
    return model

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def drow_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=10)
    except:
        pass

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


def process(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, height-10),
        (0, height/2),
        (height-10, height/2),
        (height-10, height-10)
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = region_of_interest(canny_image,
                    np.array([region_of_interest_vertices], np.int32),)
    lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi/180,
                            threshold=70,
                            lines=np.array([]),
                            minLineLength=30,
                            maxLineGap=100)
    image_with_lines = drow_the_lines(image, lines)
    return image_with_lines


def getProbability(image, detectors):
    prob = []
    for detector in detectors:
        output = detector(image)
        p = F.softmax(output)
        prob.append(p[0])
    return prob


if __name__ == '__main__':

    cap = cv2.VideoCapture('test.mp4')

    width = 224
    height = 224

    path = ["./person.pth", "./animal.pth", "./roadCones.pth", "./zebra.pth"]
    detectors = []

    for p in path:
        detectors.append(get_model(p))

    while cap.isOpened():
        ret, frame = cap.read()

        img = cv2.resize(frame, (width , height ))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_tensor = convert_to_tensor(rgb, dtype=uint8)
        rgb_tensor = expand_dims(rgb_tensor , 0)

        pred = getProbability(rgb_tensor, detectors)

        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = cv2.putText(frame, f'Zebra Crossing: {pred[3]:.2f}', (20, 20), font, 
                   1, (255, 255, 255), 2, cv2.LINE_AA)
        
        frame = process(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()