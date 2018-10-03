import numpy as np
import cv2
import os
import math

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
	
	#closing image to remove the small gaps in the barcode/qr code
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

#TODO fix, does not work
def straighten(img, rotRect):
	#concept here is kind of flawed with the rotation value given from minAreaRect()
	#if the barcode is exactly perpendicular
	#rotation will be 0 but it won't be the write orientation
	#also the points given won't necessarily start at the barcodes top left
	#if rotRect[2] == 0 and rotRect[1][0] < rotRect[1][1]:
	#	rotRect = ((rotRect[0][0],rotRect[0][1]),(rotRect[1][0],rotRect[1][1]),90)

	#get shape of original image
	h,w,c = np.shape(img)
	
	#calculate diagonal
	wh = int(math.hypot(w, -h))
	
	#create new image
	newImg = np.zeros((wh, wh, c), np.uint8)
	
	#calculate coords for centering orig image in new image
	x1 = (wh - w)/2
	y1 = (wh - h)/2
	
	#put orig inside new img
	newImg[y1:y1+h, x1:x1+w] = img
	
	#get centre
	c = (wh/2,wh/2)
	
	#getting rotation matrix so we can rotate based on the barcodes rotation
	rotMx = cv2.getRotationMatrix2D(c, rotRect[2], 1)
	img = cv2.warpAffine(newImg,rotMx,(wh, wh))
	
	#translating in to new coordinates system
	pos = np.zeros((1,2))
	pos[0,0] = rotRect[0][0]
	pos[0,1] = rotRect[0][1]
	#pos[2,0] = 1
	
	print("POS: ", pos)

	print("ROT MX: ", rotMx)
	
	newCoords = np.dot(pos, rotMx)
	print("NEW COORDS: ", newCoords)
	
	rect = ((newCoords[0][0]+x1,newCoords[0][1]+y1),(rotRect[1][0],rotRect[1][1]),0)
	
	return img, rect

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
	
def main():
	#get list of files in the images folder
	
	file = raw_input("Enter file name: ")
	img = cv2.imread("Images/" + file)
	img_choice = raw_input("Barcode (1) / QR (2): ")
	if (intCheck(img_choice)):
		if (int(img_choice) == 1):
			rotRect = scanImage(img)
			#img, rotRect = straighten(img, rotRect)
			img = drawBox(img, rotRect)
			showImage(file, img)
		elif (int(img_choice) == 2):
			print "Hello"

		
	#for showing step by step
	# img = cv2.imread("Images/" + "barcodediag.jpg")
	# rotRect = scanImageStepByStep(img)
	# img = drawBox(img, rotRect)
	# showImage("img",img)

if __name__ == "__main__":
    main()