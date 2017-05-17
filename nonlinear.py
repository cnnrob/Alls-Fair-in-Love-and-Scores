import pandas as pd
import matplotlib.pyplot as plt
import math
import pylab as P
import numpy as np

from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold, cross_val_score, cross_val_predict
from sklearn import preprocessing

dataset = pd.read_csv('MultiRegression.csv')
print(dataset)

#plot a scatter plot
#the data is exponential
plt.figure(1)
plt.scatter(dataset.Age,  dataset.Grade_Change, color='blue')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# CV_no_ln = dataset.Frequency.reshape((len(dataset.Frequency), 1))
# CV_ln = (dataset.Frequency.map(math.log)).reshape((len(dataset.Frequency), 1))
# data = dataset.Abs_grades.reshape((len(dataset.Abs_grades), 1))
#
# # Create linear regression object on the log transformed data
# regr = linear_model.LinearRegression()
#
# # Train the model using the training sets
# regr.fit(data, CV_ln)
#
# # get the predictions on the training data
# predicted_results_ln = regr.predict(data)
# predicted_results = np.exp(predicted_results_ln)
# print(predicted_results)
#
# # show in non-linear domain
# plt.figure(2)
# plt.scatter(data, predicted_results, color='green', s=75)
# plt.scatter(data, CV_no_ln, color='black')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()
#
# print("Results with fitting a linear model to log transformed data:")
# # The coefficients (m, b) of ln y = mx + t
# print('Coefficients (m): \n', regr.coef_)
# print('Intercept (b): \n', regr.intercept_)
#
# # The mean square error MSE or the mean residual sum of square of errors should be less
# MSE = mean_squared_error(CV_ln,predicted_results_ln)
# RMSE = math.sqrt(MSE)
# # Explained variance score: 1 is perfect prediction
# R2 = r2_score(CV_ln,predicted_results_ln)
#
# print("Mean residual sum of squares =", MSE)
# print("RMSE =", RMSE)
# print("R2 =", R2)