from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split

path = r'F:\Rafi\My_Study\MyTestProject\src\d2v\corpus14'

dataset = load_files(path, shuffle= False, decode_error='ignore', random_state=None,load_content=True)
trainData,testData,trainTarget,testTarget = train_test_split(dataset.data,dataset.target,train_size  = 0.9, test_size=0.1,random_state=42);
vectorizer=TfidfVectorizer( use_idf=True)
trainData=vectorizer.fit_transform(trainData)
trainData=trainData.toarray()
clf= MultinomialNB()
clf.fit(trainData,trainTarget)

testData= vectorizer.transform(testData)
testData=testData.toarray()
#pr = clf.predict(x_Test)
#print('Prediction', clf.predict(x_Test))
acuracy= clf.score(testData,testTarget)
print("Acuracy is",acuracy)

#for item in document:
#	with io.open(item,'r',encoding='utf-8') as f:
#		text=f.read()
#	with io.open('test2.txt','w',encoding='utf-8') as f1:
#		 f1.write(text)
        
	     	
