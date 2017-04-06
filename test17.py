import pickle
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import load_files
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn import metrics
path = r'F:\Rafi\My_Study\MyTestProject\src\d2v\corpus14'
dataset = load_files(path, shuffle= False, decode_error='ignore', random_state=None,load_content=True)
trainData,testData,trainTarget,testTarget = train_test_split(dataset.data,dataset.target,train_size  = 0.9, test_size=0.1,random_state=42);
vectorizer = CountVectorizer(ngram_range=(1,1),lowercase = True, analyzer='word', max_features = 20000,token_pattern='[^ ]+')
cv = vectorizer.fit_transform(trainData)
tfidf = TfidfTransformer(norm="l2")
cv = tfidf.fit_transform(cv)
print(cv.shape)
from sklearn import svm
from sklearn.metrics import accuracy_score
model = svm.SVC(kernel='rbf', C=1, gamma=1)
model.fit(cv, trainTarget)
new_doc_tfidf_matrix = vectorizer.transform(testData)
new_doc_tfidf_matrix = tfidf.transform(new_doc_tfidf_matrix)
print(new_doc_tfidf_matrix.shape)
predicted = model.predict(new_doc_tfidf_matrix)
print(accuracy_score(testTarget, predicted))
