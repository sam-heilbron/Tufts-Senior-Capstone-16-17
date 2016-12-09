#!/usr/bin/python

import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np

from enums import Color, Datatypes

class ImageParser(object):
	def __init__(self):
		self.__classifier 	= None
		self.__pp			= None
		self.__image		= None

	def load(self, classifierFilePath, dataImagePath):
		self.__classifier, self.__pp = joblib.load(classifierFilePath)
		self.__image = cv2.imread(dataImagePath)

	def parse(self, outputImagePath):
		grayscaleImage = self._getGrayscale()
		imageThreshold = self._getThreshold(grayscaleImage)
		contours = self._getContours(imageThreshold)

		contourRectangles = [cv2.boundingRect(contour) for contour in contours]
		self._parseRectangles(contourRectangles, imageThreshold)

		cv2.imwrite(outputImagePath, self.__image);

	def _getGrayscale(self):
		""" Convert to grayscale and apply Gaussian filtering """
		grayImage = cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY)

		""" 
			Could also open and close image to fill holes 
			and remove small noise pixels
		"""
		return cv2.GaussianBlur(
					src 	= grayImage, 
					ksize 	= (5, 5), 
					sigmaX 	= 0)

	def _getThreshold(self, grayscaleImage):
		# Threshold the image
		"""
		static threshold works better when there are small lines that get 'missed'
		but really the adaptiveThreshold is better
		"""
		ret_, imageThreshold = cv2.threshold(
								grayscaleImage, 
								90, 
								255, 
								cv2.THRESH_BINARY_INV)

		return imageThreshold
		"""

		return cv2.adaptiveThreshold(
					src 			= grayscaleImage,
					maxValue 		= 255,
					adaptiveMethod 	= cv2.ADAPTIVE_THRESH_MEAN_C,
					thresholdType 	= cv2.THRESH_BINARY_INV,
					blockSize 		= 5,
					C 				= 7)
		"""

	def _getContours(self, imageThreshold):
		# Find contours in the image
		im_, contours, hierarchy_ = cv2.findContours(
										imageThreshold.copy(), 
										cv2.RETR_EXTERNAL, 
										cv2.CHAIN_APPROX_SIMPLE)
		return contours

	def _predict(self, roi):
		# Calculate the HOG features
		roi_hog_fd = hog(
						roi, 
						orientations	= 9, 
						pixels_per_cell	= (7, 7), 
						cells_per_block	= (1, 1), 
						visualise		= False)
		roi_hog_fd = self.__pp.transform(
								np.array([roi_hog_fd], Datatypes.FLOAT64))

		return self.__classifier.predict(roi_hog_fd)[0]

	## region of interest
	def _getROI(self, x, y, width, height, imageThreshold):
		# Make the rectangular region around the digit
		length 	= int(height * 1.6)
		pt1 	= int(y + height // 2 - length // 2) # floor division
		pt2 	= int(x + width // 2 - length // 2)
			
		roi 	= imageThreshold[pt1:pt1 + length, pt2:pt2 + length]
			
		# Resize the image
		##http://stackoverflow.com/questions/31996367/opencv-resize-fails-on-large-image-with-error-215-ssize-area-0-in-funct
		# dsize = (28, 28)
		roi = cv2.resize(
					src 			= roi, 
					dsize 			= (28,28), 
					interpolation	= cv2.INTER_AREA)

		"""
			INTER_NEAREST - a nearest-neighbor interpolation
			INTER_LINEAR - a bilinear interpolation (used by default)
			INTER_AREA - resampling using pixel area relation. 
				It may be a preferred method for image decimation, 
				as it gives moire’-free results. But when the image is zoomed, 
				it is similar to the INTER_NEAREST method.
			INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
			INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood
		"""

		return cv2.dilate(
					src 		= roi, 
					kernel	 	= (3, 3),
					iterations 	= 1)

	def _parseRectangles(self, rectangles, imageThreshold):
		for x, y, width, height in rectangles:
			""" Draw the rectangles """
			cv2.rectangle(
				img 		= self.__image, 
				pt1 		= (x, y), 
				pt2 		= (x + width, y + height), 
				color 		= Color.GREEN, 
				thickness 	= 2) 

			""" Make prediction """
			roi = self._getROI(x, y, width, height, imageThreshold)
			number = self._predict(roi)

			""" Label image with prediction """
			cv2.putText(
				img 				= self.__image, 
				text 				= str(int(number)), 
				org 				= (x, y + height),
				fontFace 			= cv2.FONT_HERSHEY_DUPLEX, 
				fontScale 			= 1, 
				color 				= Color.RED,
				thickness 			= 2,
				bottomLeftOrigin 	= False)