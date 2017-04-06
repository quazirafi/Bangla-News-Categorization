from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
categories = ['alt.atheism',
 'comp.graphics',
# 'misc.forsale',
# 'rec.autos',
# 'rec.motorcycles',
'rec.sport.baseball',
'sci.electronics',
#'sci.med',
#'sci.space',
'soc.religion.christian',
'talk.politics.guns']
#'talk.politics.mideast']
print(categories)
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
print('categories imported')
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
print('dataFitTrained')
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
vectors_test = vectorizer.transform(newsgroups_test.data)
print('fit_transformed')
from sklearn import svm
from sklearn.metrics import accuracy_score
model = svm.SVC(kernel='linear', C=1, gamma=1, probability=True)
model.fit(vectors, newsgroups_train.target)
pred = model.predict(vectors_test)
print(accuracy_score(newsgroups_test.target, pred)) 