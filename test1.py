import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import math
import pylab as P
import numpy as np

from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

dataset = pd.read_csv('MultiRegression.csv')
print(dataset)

#pre-processing features
Frequency =  dataset.Frequency.reshape((len(dataset.Frequency), 1))
Start_Year = dataset.Start_Year.reshape((len(dataset.Start_Year),1))

AgeDiff = dataset.AgeDiff.reshape((len(dataset.AgeDiff),1))
PartnerAge = dataset.Partner_Age.reshape((len(dataset.Partner_Age),1))
Age = dataset.Age.reshape((len(dataset.Age),1))

#encode categorical variables
Gender =  dataset.Gender.reshape((len(dataset.Gender), 1))
enc = preprocessing.OneHotEncoder()
enc.fit(Gender)
gendertransform = enc.transform(Gender).toarray()

enc = preprocessing.OneHotEncoder()
enc.fit(Age)
agetransform = enc.transform(Age).toarray()

enc = preprocessing.OneHotEncoder()
enc.fit(PartnerAge)
p_agetransform = enc.transform(PartnerAge).toarray()

enc = preprocessing.OneHotEncoder()
enc.fit(AgeDiff)
agedifftransform = enc.transform(AgeDiff).toarray()

STEM = dataset.STEM.reshape((len(dataset.STEM),1))
enc = preprocessing.OneHotEncoder()
enc.fit(STEM)
stemtransform = enc.transform(STEM).toarray()

Distance = dataset.Distance.reshape((len(dataset.Distance),1))
enc = preprocessing.OneHotEncoder()
enc.fit(Distance)
disttransform = enc.transform(Distance).toarray()

Religion = dataset.Religious.reshape((len(dataset.Religious),1))
enc = preprocessing.OneHotEncoder()
enc.fit(Religion)
religiontransform = enc.transform(Religion).toarray()

enc = preprocessing.OneHotEncoder()
enc.fit(Start_Year)
starttransform = enc.transform(Start_Year).toarray()

# prepare datasets to be fed in the regression model
CV =  dataset.Grade_Change.reshape((len(dataset.Grade_Change), 1))

# THIS IS WHAT YOU EDIT FOR VARIABLES TO INCLUDE
data = np.concatenate((Frequency,gendertransform, starttransform,stemtransform,agedifftransform, religiontransform,p_agetransform,disttransform, agetransform), axis=1)
np.savetxt("foo.csv",data,delimiter=",")
print("The processed dataset: ", np.concatenate((data, CV), axis=1))

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(data, CV)

# get the predictions on the training data
predicted_results = regr.predict(data)

print("Results:")
# The coefficients (m, b) of y = mx + b
print('Coefficients (m1, m2, m3): \n', regr.coef_)
print('Intercept (b): \n', regr.intercept_)

print("Mean residual sum of squares = %.2f"
      % np.mean((regr.predict(data) - CV) ** 2))
print('R2 = %.3f' % regr.score(data,CV))