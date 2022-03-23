#make necessary imports
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from newsapi import NewsApiClient
import re
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

newsapi = NewsApiClient(api_key='34b0879255f547818b431e10afab8641')

#get the news headlines dataset
#0 -> prices remains same or declined
#1 -> prices goes up
url = 'https://raw.githubusercontent.com/krishnaik06/Stock-Sentiment-Analysis/master/Data.csv'
df = pd.read_csv(url, encoding = "ISO-8859-1")

#split the dataset into training and testing
train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']

#Removing punctuations
data = train.iloc[:, 2:27]
data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

#Renaming column names for ease of access
list1 = [i for i in range(25)]
new_Index = [str(i) for i in list1]
data.columns = new_Index

#Convertng headlines to lower case
for index in new_Index:
    data[index] = data[index].str.lower()

#merge headlines of same day
' '.join(str(x) for x in data.iloc[1, 0:25])

#make an array of news headlines only
headlines = []
for row in range(0, len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row, 0:25]))

#implement BAG OF WORDS
countvector = CountVectorizer(ngram_range = (1, 1)) #each word is treated separately
traindataset = countvector.fit_transform(headlines)

#implement RandomForest Classifier
randomclassifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy')
randomclassifier.fit(traindataset, train['Label'])

#predict for the Test Dataset
test_transform = []
for row in range(0, len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row, 2:27]))

test_dataset = countvector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)

#check accuracy of the model
matrix = confusion_matrix(test['Label'], predictions)
print(matrix)

score = accuracy_score(test['Label'], predictions)
print(score)