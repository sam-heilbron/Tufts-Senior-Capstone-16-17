#!/usr/bin/python
"""
from sklearn.svm import LinearSVC

class SVMClassifier(object):
	def __init__(self):
		self.__classifier = LinearSVC()

	def fit(self, hogFeatures, labels):
		self.__classifier.fit(hogFeatures, labels)
		return self.__classifier


	dataset = MLDataset("MNIST Original")
	dataset.load()
	labels, hogFeatures, pp = dataset.getAll()

	# Create an linear SVM object
	clf = LinearSVC()

	# Perform the training
	clf.fit(hogFeatures, labels)

	# Save the classifier
	joblib.dump((clf, pp), outputFile, compress=3)

"""