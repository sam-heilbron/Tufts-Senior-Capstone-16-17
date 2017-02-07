#!/usr/bin/env python

#   main.py


# http://answers.opencv.org/question/27411/use-opencv-to-detect-text-blocks-send-to-tesseract-ios/
# http://stackoverflow.com/questions/23028037/detecting-an-object-words-in-an-image

import os, sys
from sklearn.externals import joblib
from sklearn.svm import LinearSVC

from datasets import NISTDataset, UCILetterDataset, Chars74KDataset
from classifiers import LinearSVM, LinearSVM_HOG

def main(argv):
	linearClassifierDigitsFile = "svm_digits_cls.pkl"
	linearClassifierLetterFile = "svm_letter_cls.pkl"
	hogLinearClassifierDigitsFile = "hog_svm_digits_cls.pkl"
	hogLinearClassifierLetterFile = "hog_svm_letter_cls.pkl"


	print("loading charsDataset")
	charsDataset = Chars74KDataset()

	return

	print("Loading UCI dataset")
	letterDataset = UCILetterDataset()

	print("Loading Nist dataset")
	nistDataset = NISTDataset()


	linearClassifier = LinearSVM(letterDataset, linearClassifierLetterFile)
	print("training letter linear classifier")
	linearClassifier.trainAndSave()
	linearClassifier.trainAndSave()
	print("evaluating letter linear classifier")
	linearClassifier.testScore()
 

	linearClassifier = LinearSVM(nistDataset, linearClassifierDigitsFile)
	print("training number linear classifier")
	linearClassifier.trainAndSave()
	print("evaluating number linear classifier")
	linearClassifier.testScore()

	return

	hogLinearClassifier = LinearSVM_HOG(nistDataset, hogLinearClassifierDigitsFile)
	print("training number linear classifier with preprocessing ")
	hogLinearClassifier.trainAndSave()
	print("evaluating number linear classifier with preprocessing ")
	hogLinearClassifier.testScore()


if __name__ == '__main__':
	main(sys.argv)