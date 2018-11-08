import numpy as np
import cv2
import os
import math
import easygui
from matplotlib import pyplot as plt
 
def intCheck(string):
    try: 
        int(string)
        return True
    except ValueError:
        return False
		
def nearestOddInteger(val):
	return int(np.ceil(np.std(val)) // 2 * 2 + 1)
	
def barcodeCheck(img):
	height, width, _ = np.shape(img)

	if width > (height + 5) or width < (height - 5):
		print("Barcode: " + str(width) + " " + str(height))
		return True
	else:
		print("QR: " + str(width) + " " + str(height))
		return False

def qrCodeRead(img): 
	image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	height, width, _ = np.shape(img)
	print(np.shape(img))
	
	blur_k = (5,5)

	print(blur_k)
	image_gray = cv2.GaussianBlur(image_gray, blur_k, 0)
	showImage("blur", image_gray)

	canny = cv2.Canny(image_gray, threshold1=255-nearestOddInteger(image_gray), threshold2=255)

	showImage("canny", canny)
	_, binary_img = cv2.threshold(canny,127,255,cv2.THRESH_BINARY)
	
	showImage("bin", binary_img)
	
	structK = (4, 4)

	merge = int((math.sqrt(width * height)) * .02)
	print(merge)
	structEl = cv2.getStructuringElement(cv2.MORPH_RECT, (merge,merge))

	dilation = cv2.dilate(binary_img, structEl, iterations = 1)
	
	showImage("dilate", dilation)

	structK = (4,4)
	structEl = cv2.getStructuringElement(cv2.MORPH_RECT, structK)
	boundary = cv2.morphologyEx(dilation,cv2.MORPH_GRADIENT,structEl)
 
	_, contours, _ = cv2.findContours(boundary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 
	cv2.drawContours(image_gray, contours, -1, 255, 3)

	c = max(contours, key = cv2.contourArea)
 
	x, y, width, height = cv2.boundingRect(c)
	rotRect = cv2.minAreaRect(c)

	cv2.rectangle(img,(x, y),(x + width, y + height), (0,255,0), 2)
	
	crop_img = img[y:y + height, x:x + width]
	
	showImage("cropped", crop_img)
 
	return crop_img, rotRect
		
def decodeBarcode(img):	
	#get dimensions
	h, w, c = np.shape(img)
	
	#convert to grey and threshold it
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
	showImage("img before", img)
	#closing image to clean up the barcode, remove numbers so they don't interfere with decoding
	structEl = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h/4))
	morphed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, structEl)
	
	if np.mean(morphed) > 240:
		print(np.mean(morphed))
		structEl = cv2.getStructuringElement(cv2.MORPH_RECT, (w/4, 1))
		morphed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, structEl)
		
	showImage("img after", morphed)
	
	
	#get the binary code for each section of the barcode
	#black bar 1 white bar 0
	binaryCode = getBinary(morphed, h, w)
	

	#convert the binary code found in to the acutal barcode numbers
	finalCode = convertBinary(binaryCode)

	if finalCode is not None:		
		#display code found
		print("CODE: " + finalCode)
	else:
		print("Error code not read correctly")
		

def findBounds(img, h , w):
	startX = 0 
	endX = 0
	startY = 0 
	endY = 0
	barWidth = 1
	
	#calc startx and barWidth
	#iterating over columns finding mean values until it find one that is on avg black
	for x in range(0,w):
		#cut the image in single pixel wide, full height bars, across the image
		column = img[0:h,x:x+1]
			
		#if the mean is below 128 assume black, mark as start, and start recording the width of bars
		if np.mean(column) < 230:
			startX = x
			
			#keep iterating from the start counting how wide the bar is
			for x2 in range(x,w):
				column2 = img[0:h,x2+1]

				#once it goes back to white on avg break, stop counting bar width
				if np.mean(column2) > 230:
					break
				else:
					barWidth +=1
			break
		
	#calc endX, working backwwards from width to 0 finding where the end of the barcode is
	for x in range(w-1, 0, -1):
		column = img[0:h, x:x+1]
		
		#when found save it and break
		if np.mean(column) < 230:
			endX = x
			break
			
	#calc start y
	for y in range(0, h):
		row = img[y:y+1, startX:endX]
		
		if np.mean(row) < 175:
			startY = y
			break;
			
	#calc end y
	for y in range(h-1, 0, -1):
		row = img[y:y+1,startX:endX]
	
		if np.mean(row) < 175:
			endY = y
			break;
			

	c = (((endX-startX)/2,(endY-startY)/2))

	#if not oriented correctly rotate 90 degress
	if endX - startX < endY - startY:
		#crop the found area
		crop_img = img[startY:endY, startX:endX]
		h, w = np.shape(crop_img)
		
		#create new white image to fit the cropped image so rotating doesn't lose quality
		wh = int(math.hypot(w, -h))
		newImg = np.zeros((wh,wh), np.uint8)
		newImg.fill(255)
		
		x1 = (wh-w)/2
		y1 = (wh-h)/2
		
		newImg[y1:y1+h, x1:x1+w] = crop_img
		
		c = (wh/2,wh/2)
		
		#rotate
		rotMx = cv2.getRotationMatrix2D(c, 90, 1)
		rotImg = cv2.warpAffine(newImg, rotMx, (wh,wh), borderValue = (255,255,255))

		#find bounds in new image and overwrite previous bounds
		h,w = np.shape(rotImg)
		startX, endX, startY, endY, barWidth, img = findBounds(rotImg, h , w)
			
	print(startX, endX, startY, endY, barWidth)
	showImage("bounds", img)
	return startX, endX, startY, endY, barWidth, img
	
def getBinary(img, h, w):
	#calculate the exact boundaries/characteristics of the barcode for decoding		
	startX, endX, startY, endY, barWidth, img = findBounds(img, h, w)

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
			
	return binaryCode
	
def convertBinary(binaryCode):
	#UPC-A codes
	left = ['0001101', '0011001', '0010011', '0111101', '0100011', '0110001', '0101111', '0111011', '0110111', '0001011']
	right = ['1110010', '1100110', '1101100', '1000010', '1011100', '1001110', '1010000', '1000100', '1001000', '1110100']

	
	if errorCheck(binaryCode) is not True:
		print("Return")
		return
		
	#if the guard bars are in the correct location then the following should split the left and right side 
	#exactly for decodoing
	leftSide = binaryCode[3:45]
	rightSide = binaryCode[50:92]
	
	finalLeft = convertSide(leftSide, left)
	finalRight = convertSide(rightSide, right)
	
	if finalLeft == '' or finalRight == '':
		rightSide, leftSide = leftSide[::-1], rightSide[::-1]
		
		finalLeft = convertSide(leftSide, left)
		finalRight = convertSide(rightSide, right)
		
	finalCode = finalLeft + finalRight
	return finalCode
	
#converts a side of the barcode from binary to decimal using provided list
def convertSide(side, bin):
	final = ''
	
	for section in range(0,len(side)+1, 7):
		for code in bin:
			if side[section:section+7] == code:
				final += str(bin.index(code))
				break

	return final

#checks guard bars are in the correct place and code is correct length
def errorCheck(binaryCode):
	sideGuard = '101'
	midGuard = '01010'

	print(len(binaryCode), binaryCode)
	if len(binaryCode) != 95:
			print("incorrect length found")
			return False
			
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
				return False
		else:
			print("No match")
			return False
	else:
		print("No match")
		return False
			
	return True
 
#shows image and hold window open
def showImage(title, image):
	cv2.imshow(title, image)
	cv2.waitKey(0)
	
#aligns the image based on the rotation in the provided rotated rectangle
def align(crop_img, rotRect):
	h, w, c = np.shape(crop_img)

	#create new white image for aligning
	wh = int(math.hypot(w, - h))
	newImg = np.zeros((wh, wh, c), np.uint8)
	newImg.fill(255)
	

	x1 = (wh-w)/2
	y1 = (wh-h)/2

	# put orig inside new img
	newImg[y1:y1+h, x1:x1+w] = crop_img

	# get centre
	c = (wh/2,wh/2)

	#rotate
	rotMx = cv2.getRotationMatrix2D(c, rotRect[2], 1)
	rotImg = cv2.warpAffine(newImg, rotMx, (wh,wh), borderValue = (255,255,255))

	return rotImg

def main():
	filename = easygui.fileopenbox()
	img = cv2.imread(filename)

	img, rotRect = qrCodeRead(img)

	if barcodeCheck(img):
		aligned = align(img, rotRect)
		decodeBarcode(aligned)


if __name__ == "__main__":
    main()