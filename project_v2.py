import numpy as np
import cv2
import os
import math
from matplotlib import pyplot as plt
 
def intCheck(string):
    try: 
        int(string)
        return True
    except ValueError:
        return False
 
def qrCodeRead(img):
    #Graying the image
 
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgLap = cv2.Laplacian(image_gray, cv2.CV_32F)

    imgAbsol = np.absolute(imgLap)
     
    imgInt = np.uint8(imgAbsol)

    avg_thresh = np.mean(imgInt) + np.std(imgInt)
 
    _, binaryImg = cv2.threshold(imgInt, thresh = avg_thresh, maxval = 255, type = cv2.THRESH_BINARY)

    structEl = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

    closing = cv2.morphologyEx(binaryImg, cv2.MORPH_CLOSE, structEl)

    dilate = cv2.dilate(closing, structEl, iterations = 3)
    erode = cv2.erode(dilate, structEl, iterations = 4)

    opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, structEl)
 
    boundary = cv2.morphologyEx(erode,cv2.MORPH_GRADIENT,structEl)
 
    _, contours, _ = cv2.findContours(boundary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 
    cv2.drawContours(image_gray, contours, -1, 255, 3)

    c = max(contours, key = cv2.contourArea)
    # print(contours)
 
    x, y, width, height = cv2.boundingRect(c)

    print (str(width) + " " + str(height))

    # if width >= (height - 5) or width <= (height + 5):
    #     print("QR")
 
    cv2.rectangle(image_gray,(x, y),(x + width, y + height), (0,255,0), 2)
    
    crop_img = img[y:y + height, x:x + width]
 
    cv2.imshow("Gray", image_gray)
    cv2.waitKey(0)
 
def main():
    file = raw_input("Enter file name: ")
    # file = "ZintPortable.png"
    # file = "qr2.png"
    file = "barcode1.jpg"
    # file = "qr-phone.jpg"
    img = cv2.imread("Images/" + file)
    qrCodeRead(img)

if __name__ == "__main__":
    main()