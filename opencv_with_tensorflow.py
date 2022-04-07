import warnings

import cv2
import numpy as np
from torch import Tensor, permute, load
import Alexnet
import torch.nn as nn
import torch.nn.functional as F
# from jetbot import Robot
# from torchsummary import summary
from math import atan2, degrees

warnings.filterwarnings('ignore')

def get_model(path, out):
    model = Alexnet.alexnet(pretrained=True)

    model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out)
        )

    model.load_state_dict(load(path))
    return model

def getAngle(x1, y1, x2, y2, height, width):
    
    tX, tY = x2, y2
    center = x1, y1
    g, b = (0, width/2), (height, width/2)
    cxg, cyg = g
    cxb, cyb = b
    cx, cy = cxb, cyb
    dx, dy = g[0] - b[0], g[1] - b[1]
    

    if cxg >= cxb and cyg <= cyb:
        rads = atan2(dy, dx)
        intHeadingDeg = degrees(rads)
        intHeadingDeg = intHeadingDeg - 90

    elif cxg >= cxb and cyg >= cyb:
        rads = atan2(dx, dy)
        intHeadingDeg = degrees(rads)
        intHeadingDeg = (intHeadingDeg * -1)

    elif cxg <= cxb and cyg >= cyb:
        rads = atan2(dx, -dy)
        intHeadingDeg = degrees(rads)
        intHeadingDeg = intHeadingDeg + 180

    elif cxg <= cxb and cyg <= cyb:
        rads = atan2(dx, -dy)
        intHeadingDeg = degrees(rads) + 180

    if intHeadingDeg > 180:
        intHeadingDeg = intHeadingDeg-360
    
    if intHeadingDeg > 0:
        intHeading = intHeadingDeg - 180
    else:
        intHeading = intHeadingDeg + 180

    dx = center[0] - tX
    dy = center[1] - tY

    if tX >= center[0] and tY <= center[1]:
        rads = atan2( dy, dx)
        degs = degrees(rads)
        degs = degs - 90

    elif tX >= center[0] and tY >= center[1]:
        rads = atan2(dx, dy)
        degs = degrees(rads)
        degs = (degs * -1)

    elif tX <= center[0] and tY >= center[1]:
        rads = atan2(dx, -dy)
        degs = degrees(rads)
        degs = degs + 180

    elif tX <= center[0] and tY <= center[1]:
        rads = atan2(dx, -dy)
        degs = degrees(rads) + 180

    if tX >= center[0] and tY <= center[1]:
        rads = atan2( dy, dx)
        degs = degrees(rads)
        degs = degs - 90

    elif tX >= center[0] and tY >= center[1]:
        rads = atan2(dx, dy)
        degs = degrees(rads)
        degs = (degs * -1)

    elif tX <= center[0] and tY >= center[1]:
        rads = atan2(dx, -dy)
        degs = degrees(rads)
        degs = degs + 180

    elif tX <= center[0] and tY <= center[1]:
        rads = atan2(dx, -dy)
        degs = degrees(rads) + 180

    if degs > 180:
        degs = degs-360

    shortestAngle = degs - intHeadingDeg
    if shortestAngle > 180:
        shortestAngle -= 360
    elif shortestAngle < -180:
        shortestAngle += 360

    return shortestAngle


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
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
    return image_with_lines, lines



def getProbability(image, detectors):
    prob = []
    
    output = F.softmax(detectors[0](image))
    if max(output) > 0.7:

        p = np.argmax(output.detach().numpy())
        if(p == 0):
            prob.append('person')
        elif(p == 1):
            prob.append('animal')
        else:
            prob.append('roadCones')
    else:
        prob.append('Nothing')

    output = F.softmax(detectors[1](image))
    if max(output) > 0.7:

        p = np.argmax(output.detach().numpy())
        if(p == 0):
            prob.append('Stop')
        elif(p == 1):
            prob.append('Trafic Light - Blue')
        elif(p == 1):
            prob.append('Trafic Light - Red')
        else:
            prob.append('Trafic Light - Green')
    else:
        prob.append('Nothing')

    output = detectors[2](image)
    p = F.softmax(output)
    prob.append(p[0])
    return prob


def moveBot(lines, halt, frame):
    if halt:
        print("STOP")
        # robot.stop()
        pass
    else:
        mini = 90
        try:
            n = len(lines)
            sum = 0
            for line in lines:
                for x1, x2, y1, y2 in line:
                    sum += getAngle(x1, y1, x2, y2, width, height)
                    # shortestAngle = getAngle(x1, y1, x2, y2, width, height)
            shortestAngle = sum / n
            print(shortestAngle)
            speed = shortestAngle / 90
            if shortestAngle < -25:
                frame = cv2.putText(frame, f'LEFT', (20, 50), font, 1, (25, 25, 25), 2, cv2.LINE_AA)            
                # robot.left(0.5)
            elif shortestAngle > 25:
                frame = cv2.putText(frame, f'RIGHT', (20, 50), font, 1, (25, 25, 25), 2, cv2.LINE_AA)            
            else:
                frame = cv2.putText(frame, f'STRAIGHT', (20, 50), font, 1, (25, 25, 25), 2, cv2.LINE_AA)    

        except: pass        
    return frame


# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('test.mp4')

width = 500
height = 500

path = ["./PAC.pth", "./ST.pth","./zebra.pth"]
detectors = []

detectors.append(get_model(path[0], 3))
detectors.append(get_model(path[1], 4))
detectors.append(get_model(path[2], 2))


while cap.isOpened():
    ret, frame = cap.read()

    frame = cv2.resize(frame, (width , height ))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_tensor = Tensor(rgb)
    rgb_tensor = permute(rgb_tensor,(2,0,1))
    pred = getProbability(rgb_tensor, detectors)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # frame = cv2.putText(frame, f'D: {pred[0]}', (20, 50), font, 1, (25, 25, 25), 2, cv2.LINE_AA)
    # frame = cv2.putText(frame, f'Zebra: {pred[2]:.2f}', (20, 50), font, 1, (25, 25, 25), 2, cv2.LINE_AA)
    
    frame, lines = process(frame)

    frame = moveBot(lines, False, frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
