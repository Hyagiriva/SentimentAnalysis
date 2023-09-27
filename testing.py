

#Importing Essentials
import pandas as pd
import smtplib 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

path = 'data/output.tsv'
data = pd.read_table(path,header=None,skiprows=1,names=['Sentiment','Review'])
X = data.Review
y = data.Sentiment
#Using CountVectorizer to convert text into tokens/features
vect = CountVectorizer(stop_words='english', ngram_range = (1,1), max_df = .80, min_df = 4)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size= 0.2)
#Using training data to transform text into counts of features for each message
vect.fit(X_train)
X_train_dtm = vect.transform(X_train) 
X_test_dtm = vect.transform(X_test)

#Accuracy using Naive Bayes Model
NB = MultinomialNB()
NB.fit(X_train_dtm, y_train)
y_pred = NB.predict(X_test_dtm)
nbaccuracy=metrics.accuracy_score(y_test,y_pred)*100
print('\nNaive Bayes')
print('Accuracy Score: ',nbaccuracy,'%',sep='')
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')

#Accuracy using Logistic Regression Model
LR = LogisticRegression()
LR.fit(X_train_dtm, y_train)
y_pred = LR.predict(X_test_dtm)
lraccuracy=metrics.accuracy_score(y_test,y_pred)*100
print('\nLogistic Regression')
print('Accuracy Score: ',lraccuracy,'%',sep='')
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')

#Accuracy using SVM Model
SVM = LinearSVC()
SVM.fit(X_train_dtm, y_train)
y_pred = SVM.predict(X_test_dtm)
svmaccuracy=metrics.accuracy_score(y_test,y_pred)*100
print('\nSupport Vector Machine')
print('Accuracy Score: ',svmaccuracy,'%',sep='')
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')

#Accuracy using KNN Model
KNN = KNeighborsClassifier(n_neighbors = 3)
KNN.fit(X_train_dtm, y_train)
y_pred = KNN.predict(X_test_dtm)
knnaccuracy=metrics.accuracy_score(y_test,y_pred)*100
print('\nK Nearest Neighbors (NN = 3)')
print('Accuracy Score: ',knnaccuracy,'%',sep='')
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')

#Naive Bayes Analysis
tokens_words = vect.get_feature_names()
print('\nAnalysis')
print('No. of tokens: ',len(tokens_words))
counts = NB.feature_count_
df_table = {'Token':tokens_words,'Negative': counts[0,:],'Positive': counts[1,:]}
tokens = pd.DataFrame(df_table, columns= ['Token','Positive','Negative'])
positives = len(tokens[tokens['Positive']>tokens['Negative']])
print('No. of positive tokens: ',positives)
print('No. of negative tokens: ',len(tokens_words)-positives)
#Check positivity/negativity of specific tokens
token_search = ['awesome']
print('\nSearch Results for token/s:',token_search)
print(tokens.loc[tokens['Token'].isin(token_search)])
#Analyse False Negatives (Actual: 1; Predicted: 0)(Predicted negative review for a positive review) 
print(X_test[ y_pred < y_test ])
#Analyse False Positives (Actual: 0; Predicted: 1)(Predicted positive review for a negative review) 
print(X_test[ y_pred > y_test ])

#Custom Test: Test a review on the best performing model (Logistic Regression)
trainingVector = CountVectorizer(stop_words='english', ngram_range = (1,1), max_df = .80, min_df = 5)
trainingVector.fit(X)
X_dtm = trainingVector.transform(X)
if nbaccuracy > lraccuracy and nbaccuracy > svmaccuracy and nbaccuracy > knnaccuracy:
    LR_complete = MultinomialNB()
    LR_complete.fit(X_dtm, y)
elif lraccuracy >nbaccuracy and lraccuracy > svmaccuracy and lraccuracy > knnaccuracy:
    LR_complete = LogisticRegression()
    LR_complete.fit(X_dtm, y)
elif svmaccuracy > nbaccuracy and svmaccuracy > lraccuracy and svmaccuracy > knnaccuracy:
    LR_complete = LinearSVC()
    LR_complete.fit(X_dtm, y)
else:
    LR_complete = KNeighborsClassifier()
    LR_complete.fit(X_dtm, y)

#Input Review
print('\nTest a custom review message')
print('Enter review to be analysed: ', end=" ")
test = []
test.append(input())
test_dtm = trainingVector.transform(test)
predLabel = LR_complete.predict(test_dtm)
tags = ['Negative','Positive']
#Display Output
print('The review is predicted',tags[predLabel[0]])
if tags[predLabel[0]] == 'Negative':
      try: 
    #Create your SMTP session 
            smtp = smtplib.SMTP('smtp.gmail.com', 587) 

   #Use TLS to add security 
            smtp.starttls() 

    #User Authentication 
            smtp.login("hyagirivaaravindan19@gmail.com","tmuqdadkvapohtvk")

    #Defining The Message 
            message ="There is a negative tweet about your company."

    #Sending the Email
            smtp.sendmail("hyagirivaaravindan19@gmail.com", "hyagiriva2000@gmail.com",message) 

    #Terminating the session 
            smtp.quit() 
            print ("Email sent successfully!") 

      except Exception as ex: 
            print("Something went wrong....",ex)
