#!/usr/bin/python





from sklearn import datasets
from collections import Counter
from enums import Datatypes
from enum import Enum
import numpy as np


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


class MLDataset(Dataset):
	def load(self, dataFile, lastTrainDataRow):
		dataset		= datasets.fetch_mldata(dataFile)

		labels 		= MLDataset._extractLabels(dataset.target)
		features 	= MLDataset._extractFeatures(dataset.data)

		self._setTrainData(features[:lastTrainDataRow], labels[:lastTrainDataRow])
		self._setTestData(features[lastTrainDataRow:], labels[lastTrainDataRow:])

	@staticmethod
	def _extractLabels(target): return np.array(target, Datatypes.INT)

	@staticmethod
	def _extractFeatures(data): return np.array(data, Datatypes.INT16)



class NISTDataset(MLDataset):
	def __init__(self):
		MLDataset.load(self, "MNIST Original", 60000)


class UCILetterDataset(MLDataset):
	def __init__(self):
		MLDataset.load(self, "letter", 15000)

class Chars74KDataset(MLDataset):
	def __init__(self):
		MLDataset.load(self, "Chars74K English img", 50000)
