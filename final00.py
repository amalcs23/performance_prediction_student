#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("data_new.csv")
type(data)



from sklearn.preprocessing import LabelEncoder
processed_data=pd.DataFrame()
lb_make = LabelEncoder()
processed_data["ge"] = lb_make.fit_transform(data["ge"])
processed_data["cst"] = lb_make.fit_transform(data["cst"])
processed_data["ms"] = lb_make.fit_transform(data["ms"])
processed_data["fmi"] = lb_make.fit_transform(data["fmi"])
processed_data["fs"] = lb_make.fit_transform(data["fs"])
processed_data["fo"] = lb_make.fit_transform(data["fo"])
processed_data["mo"] = lb_make.fit_transform(data["mo"])
processed_data["nf"] = lb_make.fit_transform(data["nf"])
processed_data["sh"] = lb_make.fit_transform(data["sh"])
processed_data["ss"] = lb_make.fit_transform(data["ss"])
processed_data["me"] = lb_make.fit_transform(data["me"])



predict_class=data.columns[-1]
predictions=data[predict_class]
predictions


predict_class = predictions.apply(lambda x: 0 if (x == "Poor") else(1 if x =="Average" else 2))
predict_class


from sklearn.model_selection import train_test_split

np.random.seed(1234)

X_train, X_test, y_train, y_test = train_test_split(processed_data, predict_class, train_size=0.80, random_state=1)


# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


#SVM Classifier

import sklearn
from sklearn import svm

from sklearn.model_selection import cross_val_score

C = 1.0
svc = svm.SVC(kernel='linear',C=C,gamma=2)
svc.fit(X_train, y_train)

from sklearn.metrics import fbeta_score
predictions_test = svc.predict(X_test)
print(predictions_test)

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix  
# Model Accuracy: how often is the classifier correct?
print(confusion_matrix(y_test,predictions_test))



print(classification_report(y_test,predictions_test))


print('Accuracy is',metrics.accuracy_score(y_test,predictions_test))


input_data = pd.DataFrame([[1,0,0,3,2,0,1,0,0,0,0]])
input_data

predictions_test = svc.predict(input_data)
print(predictions_test)


