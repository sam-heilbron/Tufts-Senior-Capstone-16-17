#!/usr/bin/python

from sklearn.externals import joblib
from sklearn.svm import LinearSVC

from skimage.feature import hog
from sklearn import preprocessing
import numpy as np

from enums import Datatypes

class LinearSVM(object):
	""" Linear SVM without preprocessing

	 "one-vs-the-rest” multi-class strategy:
	 	-training n_class models
	 	-Each model decides whether it is apart of that class or not. 
	"""

	def __init__(self, datasetClass):
		self.__datasetClass 		= datasetClass
		self.__preProcessing 		= None
		self.__classifier 			= None

	##########################   PUBLIC   ###########################
	def trainAndSave(self, outputFile):
		features, labels = self.__datasetClass.getTrainingData()

		# Create an linear SVM object
		clf = LinearSVC()

		# Perform the training
		clf.fit(features, labels)
		self.__classifier = clf

		# Save the classifier
		joblib.dump(
			(self.__classifier, self.__preProcessing), 
			outputFile, 
			compress=3)

	def testScore(self):
		testFeatures, testLabels = self.__datasetClass.getTestData()

		score = self.__classifier.score(testFeatures, testLabels)
		print("Linear SVM recognition rate: %f" % score)






class LinearSVM_HOG(object):
	""" Linear SVM with HOG feature extraction (preprocessing) 

	 "one-vs-the-rest” multi-class strategy:
	 	-training n_class models
	 	-Each model decides whether it is apart of that class or not. 

	"""

	def __init__(self, datasetClass):
		self.__datasetClass 		= datasetClass
		self.__preProcessing 		= None
		self.__classifier 			= None

	##########################   PUBLIC   ###########################

	def trainAndSave(self, outputFile):
		features, labels = self.__datasetClass.getTrainingData()

		normalizedFeatures = self._normalizeFeatures(
									self._extractHOGFeatures(features))

		# Create an linear SVM object
		clf = LinearSVC()

		# Perform the training
		clf.fit(normalizedFeatures, labels)
		self.__classifier = clf

		# Save the classifier
		joblib.dump(
			(self.__classifier, self.__preProcessing), 
			outputFile, 
			compress=3)
		

	def predict(self, roi):
		# Calculate the HOG features
		roiHOGFeatures = self._getHOG(
							roi, 
							orientations	= 9, 
							pixels_per_cell	= (7, 7), 
							cells_per_block	= (1, 1), 
							visualise		= False)
		roiHOGFeatures = self.__preProcessing.transform(
								np.array([roiHOGFeatures], Datatypes.FLOAT64))

		return self.__classifier.predict(roiHOGFeatures)[0]


	def testScore(self):
		testFeatures, testLabels = self.__datasetClass.getTestData()

		normalizedFeatures = self._normalizeFeatures(
								self._extractHOGFeatures(testFeatures))

		score = self.__classifier.score(normalizedFeatures, testLabels)
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
		self.__preProcessing = preprocessing.StandardScaler().fit(hogFeatures)
		return self.__preProcessing.transform(hogFeatures)


