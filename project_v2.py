`import numpy as np
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
	
def decodeBarcode(img):	
	#get dimensions
	h, w, c = np.shape(img)
	
	#convert to grey and threshold it
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    ###MO

    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black
    # gray = cv2.bitwise_not(img)
    
    # # threshold the image, setting all foreground pixels to
    # # 255 and all background pixels to 0
    # thresh = cv2.threshold(gray, 0, 255,
    #     cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # # Taking coords of every pixel that are > 0
    # # minAreaRect will determine the angle needed to align everything
    # coords = np.column_stack(np.where(thresh > 0))
    # angle = cv2.minAreaRect(coords)[-1]
    
    # # the `cv2.minAreaRect` function returns values in the
    # # range [-90, 0); as the rectangle rotates clockwise the
    # # returned angle trends to 0 -- in this special case we
    # # need to add 90 degrees to the angle
    # if angle < -45:
    #     angle = -(90 + angle)
    
    # # otherwise, just take the inverse of the angle to make
    # # it positive
    # else:
    #     angle = -angle

    # # rotate the image to deskew it
    # (h, w) = img.shape[:2]
    # center = (w // 2, h // 2)
    # M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # rotated = cv2.warpAffine(img, M, (w, h),
    #     flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # # draw the correction angle on the image so we can validate it
    # cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
    #     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # # show the output image
    # print("[INFO] angle: {:.3f}".format(angle))
    # cv2.imshow("Input", img)
    # cv2.imshow("Rotated", rotated)
    # cv2.waitKey(0)

    ###MO

	#closing image to clean up the barcode, remove numbers so they don't interfere with decoding
	structEl = cv2.getStructuringElement(cv2.MORPH_RECT, (1,h/2 ))
	img = cv2.morphologyEx(img, cv2.MORPH_CLOSE  , structEl)
	
	#get the binary code for each section of the barcode
	#black bar 1 white bar 0
	binaryCode = getBinary(img, h, w)
	
	print("CODE(" + str(len(binaryCode)) + "): " + binaryCode)

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
		
	#calc endX, working backwwards from width to 0 finding where the end of the barcode is
	for x in range(w-1, 0, -1):
		column = img[0:h, x:x+1]
		
		#when found save it and break
		if np.mean(column) < 128:
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
			
	cv2.rectangle(img, (startX, startY), (endX, endY), 0, 3)
	showImage("area", img)
			
	#startX, endX, startY, endY, barWidth 
	return startX, endX, startY, endY, barWidth
	
def getBinary(img, h, w):
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
			
	return binaryCode
	
def convertBinary(binaryCode):
	#UPC-A codes
	left = ['0001101', '0011001', '0010011', '0111101', '0100011', '0110001', '0101111', '0111011', '0110111', '0001011']
	right = ['1110010', '1100110', '1101100', '1000010', '1011100', '1001110', '1010000', '1000100', '1001000', '1110100']

	if errorCheck(binaryCode) is not True:
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
				
	print(finalCode)
	return finalCode
	

def errorCheck(binaryCode):
	sideGuard = '101'
	midGuard = '01010'

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
 
def main():
	img = cv2.imread("Images/upca.jpg")
	decodeBarcode(img)
	showImage("Original", img)
	return

	file = raw_input("Enter file name: ")
	img = cv2.imread("Images/" + "images (1).jpg")
	qrCodeRead(img)

if __name__ == "__main__":
    main()