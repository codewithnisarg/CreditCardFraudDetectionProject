import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

data = pd.read_csv("C:\\Users\\dell\\Desktop\\creditcard.csv")

#Checking the data
print("First 5 row of our data :"+"\n",data.head())

print("Total Number of column : "+"\n",data.columns)

print(data.info())

print(data.describe())
print(data['Class'].describe())
print("Value count for Class attribute : ","\n",data['Class'].value_counts())

sns.countplot(x=data['Class'], palette = 'pastel')
plt.title('Creditcard', fontsize = 15)
plt.xlabel('Not-Fruad or Fruad', fontsize = 15)
plt.ylabel('count', fontsize = 15)

print(data['Time'].describe())
print("Time : "+"\n",data['Amount'].value_counts())


#looking for null values in the data

print(data.isnull().sum()) #there is no null values in the data

#visulising the data

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()

#standard scaling
data['std_Amount'] = scaler.fit_transform(data['Amount'].values.reshape (-1,1))

#removing Amount
data = data.drop("Amount", axis=1)

#From the above information we see that
#the distribution of fraud and notfraud data are hightly imbalanced.

# so we need to perform undersampling technique :

import imblearn
from imblearn.under_sampling import RandomUnderSampler 

undersample = RandomUnderSampler(sampling_strategy=0.5)

cols = data.columns.tolist()
cols = [c for c in cols if c not in ["Class"]]
target = "Class"

X = data[cols]
Y = data[target]

#undersample
X_under, Y_under = undersample.fit_resample(X, Y)
from pandas import DataFrame
test = pd.DataFrame(Y_under, columns = ['Class'])

#visualizing undersampling results
fig, axs = plt.subplots(ncols=2, figsize=(13,4.5))
sns.countplot(x="Class", data=data, ax=axs[0])
sns.countplot(x="Class", data=test, ax=axs[1])

fig.suptitle("Class repartition before and after undersampling")
a1=fig.axes[0]
a1.set_title("Before")
a2=fig.axes[1]
a2.set_title("After")

corr = data.corr()
plt.figure(num=None, dpi=80, facecolor='w', edgecolor='k')
corrMat = plt.matshow(corr, fignum = 1)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.gca().xaxis.tick_bottom()
plt.colorbar(corrMat)
plt.title('Correlation Matrix for ', fontsize=15)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_under, Y_under, test_size=0.2, random_state=1)

from sklearn.neural_network import MLPClassifier

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

#train the model
model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(100,100), random_state=2)
mlp = model.fit(X_train, y_train)

print(model.get_params(deep=True))

#predictions
y_pred_mlp = model.predict(X_test)

#scores
print("Accuracy MLP:",metrics.accuracy_score(y_test, y_pred_mlp))
print("Precision MLP:",metrics.precision_score(y_test, y_pred_mlp))
print("Recall MLP:",metrics.recall_score(y_test, y_pred_mlp))
print("F1 Score MLP:",metrics.f1_score(y_test, y_pred_mlp))

#CM matrix
matrix_mlp = confusion_matrix(y_test, y_pred_mlp)
cm_mlp = pd.DataFrame(matrix_mlp, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud'])

sns.heatmap(cm_mlp, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Confusion Matrix MLP"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

from sklearn.model_selection import KFold, cross_val_score

cvs1 = cross_val_score(MLPClassifier(), X_under, Y_under, scoring='accuracy', cv=KFold(n_splits=10))
print("MLP Classifier - Accuracy : ",cvs1)
