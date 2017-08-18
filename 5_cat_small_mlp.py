#!/usr/bin/python
# -*- coding: utf-8 -*-
import os, sys
import io
import codecs
import pickle
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
import codecs
import pickle
from sklearn.externals import joblib
import codecs
from collections import defaultdict
import csv
import sys
import csv
import codecs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score
import pickle
from sklearn.externals import joblib
from sklearn.datasets import load_files
import math
from sklearn import datasets, linear_model
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
reload(sys)
sys.setdefaultencoding("utf-8")
accident_docs = joblib.load("accident_docs.pkl")
education_docs = joblib.load("education_docs.pkl")
crime_docs = joblib.load("crime_docs.pkl")
economics_docs = joblib.load("economics_docs.pkl")
art_docs = joblib.load("art_docs.pkl")
entertainment_docs = joblib.load("entertainment_docs.pkl")
environment_docs = joblib.load("environment_docs.pkl")
international_docs = joblib.load("international_docs.pkl")
opinion_docs = joblib.load("opinion_docs.pkl")
politics_docs = joblib.load("politics_docs.pkl")
science_docs = joblib.load("science_tech_docs.pkl")
sports_docs = joblib.load("sports_docs.pkl")
file_names = ["accident_features.txt","education_features.txt","crime_features.txt","economics_features.txt","art_features.txt","entertainment_features.txt","environment_features.txt","international_features.txt","opinion_features.txt","politics_features.txt","science_tech_features.txt","sports_features.txt"]
all_features = {}
inv_all_features = {}
count = 0
for file_name in file_names:
	doc = codecs.open("F:\\Rafi\\My_Study\\Thesis\\Back_up_features_2\\"+file_name,encoding='utf-8')
	lines = doc.readlines()
	for line in lines:
		words = []
		words = line.split(",")
		if words[0] not in all_features:
			all_features[words[0]] = count
			inv_all_features[count] = words[0] 
			count = count + 1
row_num = len(accident_docs)+len(education_docs)+len(crime_docs)+len(economics_docs)+len(art_docs)+len(environment_docs)+len(entertainment_docs)+len(international_docs)+len(opinion_docs)+len(politics_docs)+len(science_docs)+len(sports_docs)
total_matrix = [[0 for x in range(10200)] for y in range(row_num)]
import re
regex = r'[^ ]+'
doc_count = 0
labels = []
for acc_doc in accident_docs:
	acc_doc_words = []
	acc_doc_words = re.findall(regex,acc_doc)
	for acc_doc_word in acc_doc_words:
		if acc_doc_word in all_features:
			total_matrix[doc_count][all_features[acc_doc_word]] = 1
	labels.append(0)
	doc_count += 1
for crime_doc in crime_docs:
	crime_doc_words = []
	crime_doc_words = re.findall(regex,crime_doc)
	for crime_doc_word in crime_doc_words:
		if crime_doc_word in all_features:
			total_matrix[doc_count][all_features[crime_doc_word]] = 1
	labels.append(1)
	doc_count += 1
for economics_doc in economics_docs:
	economics_doc_words = []
	economics_doc_words = re.findall(regex,economics_doc)
	for economics_doc_word in economics_doc_words:
		if economics_doc_word in all_features:
			total_matrix[doc_count][all_features[economics_doc_word]] = 1
	labels.append(2)
	doc_count += 1
for education_doc in education_docs:
	education_doc_words = []
	education_doc_words = re.findall(regex,education_doc)
	for education_doc_word in education_doc_words:
		if education_doc_word in all_features:
			total_matrix[doc_count][all_features[education_doc_word]] = 1
	labels.append(3)
	doc_count += 1
for art_doc in art_docs:
	art_doc_words = []
	art_doc_words = re.findall(regex,art_doc)
	for art_doc_word in art_doc_words:
		if art_doc_word in all_features:
			total_matrix[doc_count][all_features[art_doc_word]] = 1
	labels.append(4)
	doc_count += 1
for entertainment_doc in entertainment_docs:
	entertainment_doc_words = []
	entertainment_doc_words = re.findall(regex,entertainment_doc)
	for entertainment_doc_word in entertainment_doc_words:
		if entertainment_doc_word in all_features:
			total_matrix[doc_count][all_features[entertainment_doc_word]] = 1
	labels.append(5)
	doc_count += 1
for environment_doc in environment_docs:
	environment_doc_words = []
	environment_doc_words = re.findall(regex,environment_doc)
	for environment_doc_word in environment_doc_words:
		if environment_doc_word in all_features:
			total_matrix[doc_count][all_features[environment_doc_word]] = 1
	labels.append(6)
	doc_count += 1
for international_doc in international_docs:
	international_doc_words = []
	international_doc_words = re.findall(regex,international_doc)
	for international_doc_word in international_doc_words:
		if international_doc_word in all_features:
			total_matrix[doc_count][all_features[international_doc_word]] = 1
	labels.append(7)
	doc_count += 1
for opinion_doc in opinion_docs:
	opinion_doc_words = []
	opinion_doc_words = re.findall(regex,opinion_doc)
	for opinion_doc_word in opinion_doc_words:
		if opinion_doc_word in all_features:
			total_matrix[doc_count][all_features[opinion_doc_word]] = 1
	labels.append(8)
	doc_count += 1
for politics_doc in politics_docs:
	politics_doc_words = []
	politics_doc_words = re.findall(regex,politics_doc)
	for politics_doc_word in politics_doc_words:
		if politics_doc_word in all_features:
			total_matrix[doc_count][all_features[politics_doc_word]] = 1
	labels.append(9)
	doc_count += 1
for science_doc in science_docs:
	science_doc_words = []
	science_doc_words = re.findall(regex,science_doc)
	for science_doc_word in science_doc_words:
		if science_doc_word in all_features:
			total_matrix[doc_count][all_features[science_doc_word]] = 1
	labels.append(10)
	doc_count += 1
for sports_doc in sports_docs:
	sports_doc_words = []
	sports_doc_words = re.findall(regex,sports_doc)
	for sports_doc_word in sports_doc_words:
		if sports_doc_word in all_features:
			total_matrix[doc_count][all_features[sports_doc_word]] = 1
	labels.append(11)
	doc_count += 1
for i in range(5):
	mlp_model8 = MLPClassifier(hidden_layer_sizes=(5500, ), activation='relu',  max_iter=500, random_state=1)
	x_train, x_test, y_train, y_test = train_test_split(total_matrix, labels, test_size=0.1)
	mlp_model8.fit(x_train,y_train)
	predicted2 = mlp_model8.predict(x_test)
	count2 = 0
	for i in range(len(predicted2)):
		if (predicted2[i]-y_test[i])==0:
			count2 += 1
	print(float(count2)/float(len(predicted2)))