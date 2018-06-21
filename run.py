from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.metrics import accuracy_score


# 4 different classifier models

clf_tree = tree.DecisionTreeClassifier()
clf_svm = svm.SVC()
clf_NB = GaussianNB()
clf_KNN = neighbors.KNeighborsClassifier()



# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# Training models
clf_tree = clf_tree.fit(X, Y)
clf_svm = clf_svm.fit(X, Y)
clf_NB = clf_NB.fit(X, Y)
clf_KNN = clf_KNN.fit(X, Y)


# Test dataset
X_=[[184,84,45],[197,91,48],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
Y_=['male','male','male','female','female','female','male','male']

# Model predictions

prediction_tree = clf_tree.predict(X_)
prediction_svm = clf_svm.predict(X_)
prediction_NB = clf_tree.predict(X_)
prediction_KNN = clf_KNN.predict(X_)

# Accuracies
a_tree = accuracy_score(Y_,prediction_tree)
a_svm = accuracy_score(Y_,prediction_svm)
a_NB = accuracy_score(Y_,prediction_NB)
a_KNN = accuracy_score(Y_,prediction_KNN)


# Print most accurate model
accuracies = {'Decision Tree':a_tree, 'SVM':a_svm, 'Naive Bayes':a_NB, 'KNN':a_KNN}
print('Best result using {} of accuracy {}'.format(max(accuracies, key=accuracies.get),max(accuracies.values())))
