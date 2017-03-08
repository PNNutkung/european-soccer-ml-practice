import csv
import pandas as pd
import sqlite3
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

matches = pd.read_csv('output.csv')
cats = []
for i in range(0, len(matches)):
    if matches.ix[i]['goal_diff'] > 0:
        cats.append('win')
    elif matches.ix[i]['goal_diff'] < 0:
        cats.append('lose')
    else:
        cats.append('draw')

matches['category'] = cats
print("Win:", len(matches[(matches.category == 'win')]))
print("Lose:", len(matches[(matches.category == 'lose')]))
print("Draw:", len(matches[(matches.category == 'draw')]))

numitems = len(matches)
percenttrain = 0.85
numtrain = int(numitems*percenttrain)
numtest = numitems - numtrain
print('Training set', numtrain, 'items')
print('Test set', numtest, 'items')
matchesTrain = matches[0:numtrain]
matchesTest = matches[numtrain:numitems]

features = ['match_api_id','home_team_api_id','away_team_api_id','goal_diff']
neighbors = 5
classifier = KNeighborsClassifier(neighbors)
classifier.fit(matchesTrain[features], matchesTrain['category'])
predictions = classifier.predict(matchesTest[features])

numtrain = len(matchesTrain)
numtest = len(matchesTest)
correct = 0
for i in range(0, numtest):
    if predictions[i] == matchesTest.ix[numtrain+i]['category']: correct += 1

print('Percent correct:', float(correct)/float(numtest)*100)
