from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split

path = r'F:\Rafi\My_Study\Thesis\op'

dataset = load_files(path, shuffle= False, decode_error='ignore', random_state=None,load_content=True)
trainData,testData,trainTarget,testTarget = train_test_split(dataset.data,dataset.target,train_size  = 0.6, test_size=0.4,random_state=42);
vectorizer=TfidfVectorizer(use_idf=True)
trainData=vectorizer.fit_transform(trainData)
print(trainData.shape)
from sklearn import svm
from sklearn.metrics import accuracy_score

model = svm.SVC(kernel='linear', C=1, gamma=1, probability=True)
model.fit(trainData, trainTarget)
new_doc_tfidf_matrix = vectorizer.transform(testData)
predicted = model.predict(new_doc_tfidf_matrix)
probs = model.predict_proba(new_doc_tfidf_matrix)
fh = open('testfile.txt','w')  
for prob in probs:
	fh.write(format(prob[0], '.4f'))
	fh.write(' ')
	fh.write(format(prob[1], '.4f'))
	fh.write("\n")
fh.close() 
print(accuracy_score(testTarget, predicted)) 