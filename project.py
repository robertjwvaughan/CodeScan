import numpy as np
import cv2
import os
import math
from matplotlib import pyplot as plt
 
#How it works
#using gradients, blurs, threshold and morphology, making the desired object(barcode/qr code)
#the biggest white object in the image, so findContours can locate it easily
#this will only work if the barcode is the biggest collection of edges in the image
 
def scanImage(img):
    #convert image to grayscale to reduce image complexity
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     
    #using laplacian operator to accentuate edges, 
    #which barcodes and qr codes are full of making it easier to differentiate
    imgLap = cv2.Laplacian(imgGray, cv2.CV_32F)
     
    #using only absolute values to remove negatives
    imgAbs = np.absolute(imgLap)
     
    #converting to unsigned 8 bit integers so future functions can work with it(finding contours)
    imgAbsInt = np.uint8(imgAbs)
     
    #blur the image so the barcode/qr cide area will be a sort of blob, making it easier to threshold
    imgBlur = cv2.blur(imgAbsInt, (5,5))
     
    #calculating threshold to use based on the average of the image and its standard deviation
    threshold = np.mean(imgBlur) + np.std(imgBlur)
     
    #apply a binary threshold on the image
    _, imgThresh = cv2.threshold(imgBlur, threshold, 255, cv2.THRESH_BINARY)
     
    #get a rectangular structuring element to use in the following morphological function
    structEl = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    
    #closing image to remove the small gaps in the barcode/qr code
    imgClose = cv2.morphologyEx(imgThresh, cv2.MORPH_CLOSE, structEl)
     
    #at this point barcode/qr code should be the biggest contour in the image
    #find extreme outer contours(RETR_EXTERNAL), and only return the required edges to get the shape(CHAIN_APPROX_SIMPLE_)
    _, contours, _ = cv2.findContours(imgClose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     
    #finding the largest contour
    largestArea = 0
    largestContour = None
     
    for contour in contours:
        area = cv2.contourArea(contour)
        if  area > largestArea:
            largestArea = area
            largestContour = contour
     
    #find the smallest rectangle that will encompass all of the largest contour
    rotRect = cv2.minAreaRect(largestContour)
     
    #the centre, w/h, rot in rotRect
    return rotRect
 
#same as above function but shows and prints images at every step
def scanImageStepByStep(img):
    print(img)
    showImage("image", img)
     
    #convert image to grayscale to reduce image complexity
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(imgGray)
    showImage("image", imgGray)
     
    #using laplacian operator to accentuate edges, 
    #which barcodes and qr codes are full of making it easier to differentiate
    imgLap = cv2.Laplacian(imgGray, cv2.CV_32F)
    print(imgLap)
    showImage("image", imgLap)
     
    #using only absolute values to remove negatives
    imgAbs = np.absolute(imgLap)
    print(imgAbs)
    showImage("image", imgAbs)
     
    #converting to unsigned 8 bit integers so future functions can work with it(finding contours)
    imgAbsInt = np.uint8(imgAbs)
    print(imgAbsInt)
    showImage("image", imgAbsInt)
     
    #blur the image so the barcode/qr cide area will be a sort of blob, making it easier to threshold
    imgBlur = cv2.blur(imgAbsInt, (5,5))
    print(imgBlur)
    showImage("image", imgBlur)
     
    #calculating threshold to use based on the average of the image and its standard deviation
    threshold = np.mean(imgBlur) + np.std(imgBlur)
    print("Threshold: " + str(threshold))
     
    #apply a binary threshold on the image
    _, imgThresh = cv2.threshold(imgBlur, threshold, 255, cv2.THRESH_BINARY)
    print(imgThresh)
    showImage("image",imgThresh)
     
    #get a rectangular structuring element to use in the following morphological function
    structEl = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    print("Structuring Element: " + str(structEl))
     
    #closing image to remove the small gaps in the barcode
    imgClose = cv2.morphologyEx(imgThresh, cv2.MORPH_CLOSE, structEl)
    print(imgClose)
    showImage("image", imgClose)
     
    #at this point barcode/qr code should be the biggest contour in the image
    #find extreme outer contours(RETR_EXTERNAL), and only return the required edges to get the shape(CHAIN_APPROX_SIMPLE_)
    _, contours, _ = cv2.findContours(imgClose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     
    #finding the largest contour
    largestArea = 0
    largestContour = None
     
    for contour in contours:
        area = cv2.contourArea(contour)
        if  area > largestArea:
            largestArea = area
            largestContour = contour
     
    #find the smallest rectangle that will encompass all of the largest contour
    rotRect = cv2.minAreaRect(largestContour)
    print(rotRect)
     
    #return the image with box drawn
    return rotRect
 
def drawBox(img, rotRect):
    print(rotRect)
    #convert the values(point, size, rotation) to points that can be 
    #used in the drawContours function
    points = [cv2.boxPoints(rotRect).astype(int)]
    #draw contours
    cv2.drawContours(img,points,-1,(0,0,255),1)
    return img
     
#shows image and hold window open
def showImage(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
 
def intCheck(string):
    try: 
        int(string)
        return True
    except ValueError:
        return False
 
def qrCodeRead(img):
    #Graying the image
 
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    # Smooths the image (for poorer quality images)
    # gaussian_img = cv2.GaussianBlur(image_gray,(5,5),0)
 
    # Defines edges more
    imgLap = cv2.Laplacian(image_gray, cv2.CV_32F)
     
    # Removing negative values
    imgAbsol = np.absolute(imgLap)
     
    imgInt = np.uint8(imgAbsol)
    
    avg_thresh = np.mean(imgInt) + np.std(imgInt)
 
    _, binaryImg = cv2.threshold(imgInt, thresh = avg_thresh, maxval = 255, type = cv2.THRESH_BINARY)

    structEl = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    closing = cv2.morphologyEx(binaryImg, cv2.MORPH_CLOSE, structEl)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, structEl)
 
    boundary = cv2.morphologyEx(opening,cv2.MORPH_GRADIENT,structEl)
 
    _, contours, _ = cv2.findContours(boundary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 
    cv2.drawContours(image_gray, contours, -1, 255, 3)

    c = max(contours, key = cv2.contourArea)
    print(contours)
 
    x, y, width, height = cv2.boundingRect(c)
 
    crop_img = img[y:y + height, x:x + width]
 
    #cv2.rectangle(img,(x, y),(x + width, y + height), (0,255,0), 2)
 
    cv2.imshow("Gray", crop_img)
    cv2.waitKey(0)
 
def main():
    #get list of files in the images folder
     
    file = raw_input("Enter file name: ")
    img = cv2.imread("Images/" + "images (1).jpg")
    img_choice = raw_input("Barcode (1) / QR (2): ")
    if (intCheck(img_choice)):
        if (int(img_choice) == 1):
            rotRect = scanImage(img)
            #img, rotRect = straighten(img, rotRect)
            img = drawBox(img, rotRect)
            showImage(file, img)
        elif (int(img_choice) == 2):
            qrCodeRead(img)
         
    #for showing step by step
    # img = cv2.imread("Images/" + "barcodediag.jpg")
    # rotRect = scanImageStepByStep(img)
    # img = drawBox(img, rotRect)
    # showImage("img",img)
 
if __name__ == "__main__":
    main()