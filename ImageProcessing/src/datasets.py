#!/usr/bin/python

from sklearn import datasets
from skimage.feature import hog
from sklearn import preprocessing
import numpy as np
from collections import Counter

from enums import Datatypes

class MLDataset(object):
	def __init__(self):
		self.__features 	= None
		self.__label 		= None
		self.__hogFeatures 	= None # https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
		self.__pp 			= None

	def load(self, dataFile = "MNIST Original"):
		dataset = datasets.fetch_mldata(dataFile)

		self._extractLabels(dataset.target)
		self._extractFeatures(dataset.data)

		self._normalizeFeatures(
			self._extractHOGFeatures())
		
	def _extractLabels(self, target):
		self.__labels = np.array(target, Datatypes.INT)

	def _extractFeatures(self, data):
		self.__features = np.array(data, Datatypes.INT16) 

	def _extractHOGFeatures(self):
		""" Extract the hog features """

		list_hog_fd = []
		for feature in self.__features:
			fd = hog(
					feature.reshape((28, 28)), 
					orientations	= 9, 
					pixels_per_cell	= (7, 7), 
					cells_per_block	= (1, 1), 
					visualise		= False)
			list_hog_fd.append(fd)

		return np.array(list_hog_fd, Datatypes.FLOAT64)

	def _normalizeFeatures(self, hogFeatures):
		# Normalize the features
		self.__pp = preprocessing.StandardScaler().fit(hogFeatures)
		self.__hogFeatures = self.__pp.transform(hogFeatures)

	def getCount(self):
		return Counter(self.__labels)

	def getAll(self):
		return self.__labels, self.__hogFeatures, self.__pp