from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn import metrics
path = r'F:\Rafi\My_Study\MyTestProject\src\d2v\corpus14'

dataset = load_files(path, shuffle= False, decode_error='ignore', random_state=None,load_content=True)
trainData,testData,trainTarget,testTarget = train_test_split(dataset.data,dataset.target,train_size  = 0.9, test_size=0.1,random_state=42);
vectorizer=TfidfVectorizer(use_idf=True)
trainData=vectorizer.fit_transform(trainData)
print(trainData.shape)
from sklearn import svm
from sklearn.metrics import accuracy_score
target_names = ['accident','art','crime','economics','education','entertainment','environment','international','opinion','politics','science_tech','sports']
#target_names = ['opinion','politics']
model = svm.SVC(kernel='linear', C=1, gamma=1, probability=True)
model.fit(trainData, trainTarget)
new_doc_tfidf_matrix = vectorizer.transform(testData)
predicted = model.predict(new_doc_tfidf_matrix)
probs = model.predict_proba(new_doc_tfidf_matrix)
fh = open('LastYear2.txt','w')  
fh.write('accident')
fh.write(' ')
fh.write('art')
fh.write(' ')
fh.write('crime')
fh.write(' ')
fh.write('economics')
fh.write(' ')
fh.write('education')
fh.write(' ')
fh.write('entertainment')
fh.write(' ')
fh.write('environment')
fh.write(' ')
fh.write('international')
fh.write(' ')
fh.write('opinion')
fh.write(' ')
fh.write('politics')
fh.write(' ')
fh.write('science_tech')
fh.write(' ')
fh.write('sports')
fh.write(' ')
fh.write('predicted')
fh.write(' ')
fh.write('target')
fh.write("\n")
for i in range(0,len(predicted)):
	if predicted[i]!=testTarget[i]:
		fh.write(format(probs[i][0], '.2f'))
		fh.write(' ')
		fh.write(format(probs[i][1], '.2f'))
		fh.write(' ')
		fh.write(format(probs[i][2], '.2f'))
		fh.write(' ')
		fh.write(format(probs[i][3], '.2f'))
		fh.write(' ')
		fh.write(format(probs[i][4], '.2f'))
		fh.write(' ')
		fh.write(format(probs[i][5], '.2f'))
		fh.write(' ')
		fh.write(format(probs[i][6], '.2f'))
		fh.write(' ')
		fh.write(format(probs[i][7], '.2f'))
		fh.write(' ')
		fh.write(format(probs[i][8], '.2f'))
		fh.write(' ')
		fh.write(format(probs[i][9], '.2f'))
		fh.write(' ')
		fh.write(format(probs[i][10], '.2f'))
		fh.write(' ')
		fh.write(format(probs[i][11], '.2f'))
		fh.write(' ')
		fh.write(str(predicted[i]))
		fh.write(' ')
		fh.write(str(testTarget[i]))
		fh.write("\n")
fh.close() 
print(accuracy_score(testTarget, predicted)) 
print(metrics.classification_report(testTarget, predicted,target_names=target_names))