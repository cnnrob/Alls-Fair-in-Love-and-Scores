import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn import preprocessing

dataset = pd.read_csv('446-data-logisticregression.csv')
print(dataset)


Age = dataset.Age.reshape((len(dataset.Age),1))
enc = preprocessing.OneHotEncoder()
enc.fit(Age)
Agebin = enc.transform(Age).toarray()

School = dataset.SchoolNum.reshape((len(dataset.SchoolNum),1))
enc = preprocessing.OneHotEncoder()
enc.fit(School)
Schoolbin = enc.transform(School).toarray()

Program = dataset.ProgramNum.reshape((len(dataset.ProgramNum),1))
enc = preprocessing.OneHotEncoder()
enc.fit(Program)
Programbin = enc.transform(Program).toarray()

Religion = dataset.IsReligious.reshape((len(dataset.IsReligious),1))
enc = preprocessing.OneHotEncoder()
enc.fit(Religion)
Relbin = enc.transform(Religion).toarray()

Gender = dataset.IsFemale.reshape((len(dataset.IsFemale),1))
enc = preprocessing.OneHotEncoder()
enc.fit(Gender)
Genderbin = enc.transform(Gender).toarray()

Distance = dataset.HourBin.reshape((len(dataset.HourBin),1))
enc = preprocessing.OneHotEncoder()
enc.fit(Distance)
Distbin = enc.transform(Distance).toarray()

P_Age = dataset.Partner_Age.reshape((len(dataset.Partner_Age),1))
enc = preprocessing.OneHotEncoder()
enc.fit(P_Age)
Partnerbin = enc.transform(P_Age).toarray()

Frequency = dataset.Frequency.reshape((len(dataset.Frequency),1))
enc = preprocessing.OneHotEncoder()
enc.fit(Frequency)
FreqBin = enc.transform(Frequency).toarray()

LiveTog = dataset.TogetherBin.reshape((len(dataset.TogetherBin),1))
enc = preprocessing.OneHotEncoder()
enc.fit(LiveTog)
LiveTogBin = enc.transform(LiveTog).toarray()

Startyear = dataset.Start_Year.reshape((len(dataset.Start_Year),1))
enc = preprocessing.OneHotEncoder()
enc.fit(Startyear)
Startbin = enc.transform(Startyear).toarray()



data = np.concatenate((Programbin,FreqBin,Schoolbin,LiveTogBin), axis=1)



CV = dataset.GradeDecrease.reshape((len(dataset.GradeDecrease),1))
#CV = dataset.Grade_Bin.reshape((len(dataset.Grade_Bin),1))

# Create a KNN object
LogReg = LogisticRegression()

# Train the model using the training sets
LogReg.fit(data, CV)

# the model
print('Coefficients (m): \n', LogReg.coef_)
print('Intercept (b): \n', LogReg.intercept_)

#predict the class for each data point
predicted = LogReg.predict(data)
print("Predictions: \n", np.array([predicted]).T)

# predict the probability/likelihood of the prediction
#print("Probability of prediction: \n",LogReg.predict_proba(data))

print("Accuracy score for the model: \n", LogReg.score(data,CV))
print(metrics.confusion_matrix(CV, predicted, labels=["Yes","No"]))
#print(metrics.confusion_matrix(CV, predicted, labels=["Greater than 4%","0 to 4%", "No change","-4% to 0", "Less than -4%"]))