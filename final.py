import numpy as np
import cv2
import matplotlib.pyplot as plt

image=cv2.imread('C:/Users/Sushanti/autoz/lane.png')

converted = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

# white color mask
lower = np.uint8([  0, 200,   0])
upper = np.uint8([255, 255, 255])
white_mask = cv2.inRange(converted, lower, upper)
# yellow color mask
lower = np.uint8([ 10,   0, 100])
upper = np.uint8([ 40, 255, 255])
yellow_mask = cv2.inRange(converted, lower, upper)

mask = cv2.bitwise_or(white_mask, yellow_mask)
white_yellow_lines = cv2.bitwise_and(image, image, mask = mask)
#white_yellow_images = list(map(recognize, image))

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
smoothing = cv2.GaussianBlur(image, (11, 11), 0)

canny_edges = cv2.Canny(smoothing, 50, 50)

pts = np.array([[0,1280],[600,150],[800,150],[smoothing.shape[1],1280]], np.int32)
x=cv2.polylines(canny_edges,[pts],True,(0,0,0))
mask = np.array([pts], dtype=np.int32)
image2 = np.zeros((720,1280), np.int8)
p=cv2.fillPoly(image2, [mask],255)
maskimage2 = cv2.inRange(image2, 1, 255)
roi = cv2.bitwise_and(x, x, mask=maskimage2)

lines = cv2.HoughLines(roi,1,np.pi/180,10)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    hough=cv2.line(roi,(x1,y1),(x2,y2),(255,255,255),2)

cv2.imshow('houghlines3.jpg',hough)

cv2.imshow('img',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

