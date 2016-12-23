#!/usr/bin/env python

#   main.py


# http://answers.opencv.org/question/27411/use-opencv-to-detect-text-blocks-send-to-tesseract-ios/
# http://stackoverflow.com/questions/23028037/detecting-an-object-words-in-an-image

import os, sys
from sklearn.externals import joblib
from sklearn.svm import LinearSVC

from datasets import NISTDataset
from classifiers import LinearSVM, LinearSVM_HOG

def main(argv):
	linearClassifierDigitsFile = "svm_digits_cls.pkl"
	hogLinearClassifierDigitsFile = "hog_svm_digits_cls.pkl"

	linearClassifierPath = os.path.join("classifiers/", 
		linearClassifierDigitsFile)
	hogLinearClassifierPath = os.path.join("classifiers/", 
		hogLinearClassifierDigitsFile)

	print("Loading dataset")
	nistDataset = NISTDataset()
	linearClassifier = LinearSVM(nistDataset)

	print("training linear classifier")
	linearClassifier.trainAndSave(linearClassifierPath)
	print("evaluating linear classifier")
	linearClassifier.testScore()


	hogLinearClassifier = LinearSVM_HOG(nistDataset)

	print("training linear classifier with preprocessing ")
	hogLinearClassifier.trainAndSave(hogLinearClassifierPath)
	print("evaluating linear classifier with preprocessing ")
	hogLinearClassifier.testScore()


if __name__ == '__main__':
	main(sys.argv)