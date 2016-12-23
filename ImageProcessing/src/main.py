#!/usr/bin/env python

#   main.py


# http://answers.opencv.org/question/27411/use-opencv-to-detect-text-blocks-send-to-tesseract-ios/
# http://stackoverflow.com/questions/23028037/detecting-an-object-words-in-an-image

import os, sys
from sklearn.externals import joblib
from sklearn.svm import LinearSVC

from datasets import MLDataset
from parsers import ImageParser

def main(argv):
	trainedDigitsFile = "digits_cls.pkl"
	imageName = "Crop-9.jpg"
	#imageName = "digits_table_filled.png"
	#imageName = "digits2.jpg"
	
	inputImagePath = os.path.join("images/raw/", imageName)
	outputImagePath = os.path.join("images/labeled/", imageName)
	
	#train(trainedDigitsFile)
	#print("Finished training data")

	parse(trainedDigitsFile, inputImagePath, outputImagePath)
	print("Finished labeling image")

	return

def train(outputFile):
	dataset = MLDataset()
	dataset.load("MNIST Original")
	labels, hogFeatures, pp = dataset.getAll()

	# Create an linear SVM object
	clf = LinearSVC()

	# Perform the training
	clf.fit(hogFeatures, labels)

	# Save the classifier
	joblib.dump((clf, pp), outputFile, compress=3)

def parse(classifierFilePath, inputImagePath, outputImagePath):
	parser = ImageParser()
	parser.load(classifierFilePath, inputImagePath)
	parser.parse(outputImagePath)


if __name__ == '__main__':
	main(sys.argv)