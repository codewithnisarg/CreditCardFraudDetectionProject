#%%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


#%%

data = pd.read_csv("card.csv")
data.head()
data.info()
data.describe()

data.shape

classes = data['Class'].value_counts()
print(classes)
sns.countplot(x="Class",data=data)
plt.title("No. of fraudulent vs. No. of non- fradulent")


normal = round((classes[0]/data['Class'].count()*100),2)
print(normal)

fraud = round((classes[1]/data['Class'].count()*100),2)
print(fraud)


#%%

"""Graphs"""

sns.countplot(x="Class",data=data)
plt.title("No. of fraudulent vs. No. of non- fradulent")



#%%

#Now let us check that time column is related to class column or not.

#let us craete new variables by fraud and non- fraud transactions.

data_fraud = data[data['Class']==1]
data_non_fraud = data[data['Class']==0]

#Distribution Plot

ax = sns.distplot(data_fraud['Time'],label="Fraud",hist=False)
ax = sns.distplot(data_non_fraud['Time'],label="Non - Fraud",hist=False)

#here we can not see any pattern with time so we can drop it .

data.drop('Time',axis=1)

data.shape

data.info()

#sns.pairplot(data)

#%%%

#now let us split data 

X = data.drop("Class",axis=1)
X.shape
X.info()

y = data['Class']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=0)

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()

X_train['Amount'] = sc.fit_transform(X_train[['Amount']])
X_test['Amount'] = sc.transform(X_test[['Amount']])

#%%

#here we have target variable imbalance so we have to make it balance otherwise
#our data model will predict only non fraud .

from imblearn.over_sampling import ADASYN

ada = ADASYN(random_state=0)
X_train_adasyn, y_train_adasyn = ada.fit_resample(X_train, y_train)


from collections import Counter

# Befor sampling class distribution
print('Before sampling class distribution:-',Counter(y_train))
# new class distribution 
print('New class distribution:-',Counter(y_train_adasyn))


#%%

from sklearn.linear_model import LogisticRegression
# Impoting metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# Importing libraries for cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#%%

# Creating KFold object with 3 splits
folds = KFold(n_splits=10, shuffle=True, random_state=4)


# Specify params
params = {"C": [0.01, 0.1, 1, 10, 100, 1000]}

# Specifing score as roc-auc
model_cv = GridSearchCV(estimator = LogisticRegression(),
                        param_grid = params, 
                        scoring= 'roc_auc', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True) 

# Fit the model
model_cv.fit(X_train_adasyn, y_train_adasyn)
best_score = model_cv.best_score_
best_C = model_cv.best_params_['C']

print(" The highest test roc_auc is {0} at C = {1}".format(best_score, best_C))
#The highest test roc_auc is 0.9953533903855286 at C = 10

#%%

logistic_bal_adasyn = LogisticRegression(C=10)

# Fit the model on the train set
logistic_bal_adasyn_model = logistic_bal_adasyn.fit(X_train_adasyn, y_train_adasyn)

# Predictions on the train set
#y_train_pred = logistic_bal_adasyn_model.predict(X_test_adasyn)
y_test_pred = logistic_bal_adasyn_model.predict(X_test)

# Confusion matrix
#confusion = metrics.confusion_matrix(y_train_adasyn, y_train_pred)
#print(confusion)
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))

print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))


