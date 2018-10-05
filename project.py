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
	
	#using only absolute values to make negative values positive
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
	
	#using only absolute values to make negative values positive
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
	#rotation will be 0 but it won't be the right orientation
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
	
def decodeBarcode(img):
		#UPC-A codes
		sideGuard = '101'
		midGuard = '01010'
		left = ['0001101', '0011001', '0010011', '0111101', '0100011', '0110001', '0101111', '0111011', '0110111', '0001011']
		right = ['1110010', '1100110', '1101100', '1000010', '1011100', '1001110', '1010000', '1000100', '1001000', '1110100']
		
		#get dimensions
		h, w, c = np.shape(img)
		
		#convert to grey and threshold it
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		_, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

		#closing image to clean up the barcode, remove numbers so they don't interfere with decoding
		structEl = cv2.getStructuringElement(cv2.MORPH_RECT, (1,h ))
		img = cv2.morphologyEx(img, cv2.MORPH_CLOSE  , structEl)
		
		#calculate the exact boundaries/characteristics of the barcode for decoding		
		startX, endX, startY, endY, barWidth = findBounds(img, h, w)
		
		binaryCode = ''
		
		#iterator over the barcode using above variables
		for column in range(startX, endX, barWidth):
			#select bar
			line = img[startY: endY, column:column+barWidth]
			
			#find the avg value for the pixels in the selected bar
			avg = np.mean(line)
			
			#if in the upper half of values binary 0, else binary 1
			if avg > int((255/2)):
				binaryCode += '0'
			else:
				binaryCode += '1'
		
		print("CODE(" + str(len(binaryCode)) + "): " + binaryCode)

		#checking if guard bars are in the correct place, otherwise don't bother checking the rest
		#only works for perfectly aligned and read.
		if binaryCode[0:3] == sideGuard:
			print("match left side guard")
		
			if binaryCode[45:50] == midGuard:
				print("match mid guard")
		
				if binaryCode[92:95] == sideGuard:
					print("match right side guard")
				else:
					print("No match")
					return
			else:
				print("No match")
				return
		else:
			print("No match")
			return
		
		#if the guard bars are in the correct location then the following should split the left and right side 
		#exactly for decodoing
		leftSide = binaryCode[3:45]
		rightSide = binaryCode[50:92]
		
		finalCode = ''
		
		#convert the left side binary to the numbers they represent
		#seven sections per number, so iterate by 7
		for section in range(0,len(leftSide)+1, 7):
			for lCode in left:
				if leftSide[section:section+7] == lCode:
					finalCode += str(left.index(lCode))
					break
					
		#convert the right side binary to the numbers they represent
		for section in range(0,len(rightSide)+1, 7):
			for rCode in right:
				if rightSide[section:section+7] == rCode:
					finalCode += str(right.index(rCode))
					break
					
		#hardcoded value for perf.jpg for quickly checking match
		if finalCode == '705632085943':
			print("CODE MATCHES")
		
		print(finalCode)
			
		showImage("decoding", img)

def findBounds(img, h , w):
	startX = 0 
	endX = 0
	startY = 0 
	endY = 0
	barWidth = 1

	#iterating over columns finding mean values until it find one that is on avg black
	for x in range(0,w):
		#cut the image
		column = img[0:h,x:x+1]
		
		#if the mean is below 128 assume black, mark as start, and start recording the width of bars
		if np.mean(column) < 128:
			startX = x
			
			#keep iterating from the start counting how wide the bar is
			for x2 in range(x,w):
				column2 = img[0:h,x2+1]

				#once it goes back to white on avg break, stop counting bar width
				if np.mean(column2) > 128:
					break
				else:
					barWidth +=1
			break
			
	#startX, endX, startY, endY, barWidth 
	return startX, 831, 0, 488, barWidth

def main():
	#just hardcoding manually aligned image until we get it to auto align for decoding
	# decodeBarcode(cv2.imread("Images/perf.jpg"))
	# return
	
	file = raw_input("Enter file name: ")
	img = cv2.imread("Images/" + file)
	img_choice = raw_input("Barcode (1) / QR (2): ")
	if (intCheck(img_choice)):
		if (int(img_choice) == 1):
			rotRect = scanImageStepByStep(img)
			#img, rotRect = straighten(img, rotRect)
			img = drawBox(img, rotRect)
			showImage(file, img)
			
		elif (int(img_choice) == 2):
			print "Hello"

if __name__ == "__main__":
    main()