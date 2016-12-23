#!/usr/bin/python





from sklearn import datasets
from collections import Counter
from enums import Datatypes
import numpy as np

from enums import Datatypes

class Dataset(object):
	def __init__(self):
		self.__trainData 	= None
		self.__testData 	= None

	def _setTrainData(self, features, labels):
		self.__trainData = features, labels

	def _setTestData(self, features, labels):
		self.__testData = features, labels

	def getTrainingData(self):
		return self.__trainData

	def getTestData(self):
		return self.__testData

	def getTrainingCount(self):
		features, labels = self.__trainData
		return Counter(labels)

	def getTestCount(self):
		features, labels = self.__testData
		return Counter(labels)


class IntDataset(Dataset):
	def _extractLabels(self, target):
		return np.array(target, Datatypes.INT)

	def _extractFeatures(self, data):
		return np.array(data, Datatypes.INT16)


class NISTDataset(IntDataset):
	def __init__(self):
		self.load()
		
	def load(self, dataFile = "MNIST Original"):
		dataset = datasets.fetch_mldata(dataFile)

		labels = self._extractLabels(dataset.target)
		features = self._extractFeatures(dataset.data)

		self._setTrainData(features[:60000], labels[:60000])
		self._setTestData(features[60000:], labels[60000:])