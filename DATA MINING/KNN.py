import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold, cross_val_score

dataset = pd.read_csv('knn.csv')
print(dataset)

# prepare datasets to be fed in the regression model
#predict grade increase given start year and frequency
CV =  dataset.grade_increase.reshape((len(dataset.grade_increase), 1))
data = (dataset.ix[:,'frequency':'start_year'].values).reshape((len(dataset.grade_increase), 2))

# Create a KNN object
KNN = KNeighborsClassifier(n_neighbors=3)

# Train the model using the training sets
KNN.fit(data, CV)

#predict the class for each data point
predicted = KNN.predict(data)
print("Predictions: \n", np.array([predicted]).T)

# predict the probability/likelihood of the prediction
print("Probability of prediction: \n",KNN.predict_proba(data))

print("Neighbors and their Distance: \n",KNN.kneighbors(data, return_distance=True))

print("Accuracy score for the model: \n", KNN.score(data,CV))

print(metrics.confusion_matrix(CV, predicted, labels=["Yes","No"]))

# Calculating 5 fold cross validation results
model = KNeighborsClassifier()
kf = KFold(len(CV), n_folds=5)
scores = cross_val_score(model, data, CV, cv=kf)
print("Accuracy of every fold in 5 fold cross validation: ", abs(scores))
print("Mean of the 5 fold cross-validation: %0.2f" % abs(scores.mean()))