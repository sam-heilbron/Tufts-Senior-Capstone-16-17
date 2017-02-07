#!/usr/bin/python
import os

from sklearn.externals import joblib
from sklearn.svm import LinearSVC

from skimage.feature import hog
from sklearn import preprocessing
import numpy as np

from enums import Datatypes

class Classifier(object):
	def __init__(
			self,
			datasetClass, 
			preProcessing, 
			classifier,
			outputFile):
		self.datasetClass 		= datasetClass
		self.preProcessing 		= preProcessing
		self.classifier 		= classifier
		self.outputFile			= outputFile

	# Save the classifier
	def _saveToFile(self):
		joblib.dump(
			(self.classifier, self.preProcessing), 
			os.path.join("classifiers/", self.outputFile), 
			compress=3)


class LinearSVM(Classifier):
	""" Linear SVM without preprocessing

	 "one-vs-the-rest” multi-class strategy:
	 	-training n_class models
	 	-Each model decides whether it is apart of that class or not. 
	"""


	def __init__(self, datasetClass, outputFile):
		Classifier.__init__(
			self,
			datasetClass 	= datasetClass, 
			preProcessing 	= None,
			classifier 		= LinearSVC(),
			outputFile 		= outputFile)

	##########################   PUBLIC   ###########################
	def trainAndSave(self):
		features, labels = self.datasetClass.getTrainingData()

		self.classifier = self.classifier.fit(features, labels)

		# Save the classifier
		self._saveToFile()

	def testScore(self):
		testFeatures, testLabels = self.datasetClass.getTestData()

		score = self.classifier.score(testFeatures, testLabels)
		print("Linear SVM recognition rate: %f" % score)




class LinearSVM_HOG(Classifier):
	""" Linear SVM with HOG feature extraction (preprocessing) 

	 "one-vs-the-rest” multi-class strategy:
	 	-training n_class models
	 	-Each model decides whether it is apart of that class or not. 

	"""

	def __init__(self, datasetClass, outputFile):
		Classifier.__init__(
			self,
			datasetClass 	= datasetClass, 
			preProcessing 	= preprocessing.StandardScaler(),
			classifier 		= LinearSVC(),
			outputFile 		= outputFile)

	##########################   PUBLIC   ###########################

	def trainAndSave(self):
		features, labels = self.datasetClass.getTrainingData()

		normalizedFeatures = self._normalizeFeatures(
									self._extractHOGFeatures(features))

		self.classifier = self.classifier.fit(normalizedFeatures, labels)

		# Save the classifier
		self._saveToFile()
		

	def predict(self, roi):
		# Calculate the HOG features
		roiHOGFeatures = self._getHOG(roi)

		roiHOGFeatures = self.preProcessing.transform(
								np.array([roiHOGFeatures], Datatypes.FLOAT64))

		return self.classifier.predict(roiHOGFeatures)[0]


	def testScore(self):
		testFeatures, testLabels = self.datasetClass.getTestData()

		normalizedFeatures = self._normalizeFeatures(
								self._extractHOGFeatures(testFeatures))

		score = self.classifier.score(normalizedFeatures, testLabels)
		print("Linear SVM with HOG preprocessing recognition rate: %f" % score)

	########################   PROTECTED   ##########################

	def _extractHOGFeatures(self, features):
		""" Extract the hog features """
		hogFeatures = []
		for feature in features:
			hogFeatures.append(
				self._getHOG(feature.reshape((28, 28))))

		return np.array(hogFeatures, Datatypes.FLOAT64)

	def _getHOG(self, img):
		return hog(
				img, 
				orientations	= 9, 
				pixels_per_cell	= (7, 7), 
				cells_per_block	= (1, 1), 
				visualise		= False)

	def _normalizeFeatures(self, hogFeatures):
		""" Normalize the HOG Features """
		self.preProcessing = self.preProcessing.fit(hogFeatures)
		return self.preProcessing.transform(hogFeatures)


