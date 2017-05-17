import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing

dataset = pd.read_csv('446-dataforNB.csv')

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

# prepare datasets to be fed into the naive bayes model
#predict grade change bin given school and program
CV =  dataset.Grade_Bin.reshape((len(dataset.Grade_Bin), 1))


data = np.concatenate((Programbin,Schoolbin,Agebin,FreqBin,Partnerbin,Distbin), axis=1)



#data = np.concatenate((FreqBin,LiveTogBin,Distbin,Genderbin, Programbin,Schoolbin,Agebin,Startyear,Partnerbin, Relbin), axis=1)
NB = MultinomialNB()

NB.fit(data,CV)


#predict the class for each data point
predicted = NB.predict(data)
print("Predictions:\n",np.array([predicted]).T)

# predict the probability/likelihood of the prediction
prob_of_pred = NB.predict_proba(data)
print("Probability of each class for the prediction: \n",prob_of_pred)

print("Accuracy of the model: ",NB.score(data,CV))

print("The confusion matrix:\n", metrics.confusion_matrix(CV, predicted, ['Greater than 4%','No change','Less than -4%','0 to 4%','-4% to 0']))
