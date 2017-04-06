from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split

path = r'F:\Rafi\My_Study\MyTestProject\src\d2v\corpus14'

dataset = load_files(path, shuffle= False, decode_error='ignore', random_state=None,load_content=True)
trainData,testData,trainTarget,testTarget = train_test_split(dataset.data,dataset.target,train_size  = 0.9, test_size=0.1,random_state=42);
vectorizer=TfidfVectorizer(use_idf=True)
trainData=vectorizer.fit_transform(trainData)
print(trainData.shape)
features = vectorizer.get_feature_names()	
fh = open('testfile.txt','w')  
for ftr in features: 
	fh.write(ftr)
	fh.write('\n') 
fh.close() 


from sklearn import svm
from sklearn.metrics import accuracy_score

model = svm.SVC(kernel='rbf', C=2, gamma=1) 
model.fit(trainData, trainTarget)
new_doc_tfidf_matrix = vectorizer.transform(testData)
predicted = model.predict(new_doc_tfidf_matrix)
print(accuracy_score(testTarget, predicted)) 