import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("F:\\TRENT COURSE\\DM\\PROJECT\\creditcard.csv")
dataset.head()
dataset.info()

x = dataset.iloc[:,1:30].values
y = dataset.iloc[:,30].values
print("Input Range : " , x.shape)
print("Output Range : " , y.shape)
print("Class Labels : \n",y)

dataset.isnull().values.any()

set_class = pd.value_counts(dataset['Class'] , sort = True)
set_class.plot(kind = 'bar' , rot =0)
plt.title("Count Disrtibution ")
plt.xlabel("Classes")
plt.ylabel("No of occurances")
plt.xticks(range(2))

fraud_data = dataset[dataset["Class"] == 1]
normal_data = dataset[dataset["Class"] == 0]
print(fraud_data.shape , normal_data.shape)
fraud_data.Amount.describe()

#finding coreleation
import sklearn.model_selection as train_test_split
corrmat = dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (20,20))
g = sns.heatmap(dataset[top_corr_features].corr(),annot = True , cmap = 'RdYlGn')

xtrain , xtest , ytrain , ytest = train_test_split(x, y , test_size = 0.30 , random_state = 0)

print('xtrain.shape :' , xtrain.shape)
print("xtest.shape :" , xtest.shape)
print("ytrain.shape :" , ytrain.shape)
print("ytest.shape : " , ytest.shape)

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
xtrain =  stdsc.fit_transform(xtrain)
xtest = stdsc.transform(xtest)
print("Training Set after Standardised : \n" , xtrain[0])

from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy' , random_state = 0)
dt_classifier.fit(xtrain , ytrain )
y_pred_decision_tree = dt_classifier.predict(xtest)
print("y_pred_decision_tree : \n" ,y_pred_decision_tree)

from sklearn.metrics import confusion_matrix
con_decision = confusion_matrix(ytest , y_pred_decision_tree)
print("confusion matrix : \n" , con_decision)
accuracy = ((con_decision[0][0]  + con_decision[0][1]) / con_decision.sum())*100
print(accuracy)
error_rate = ((con_decision[0][1] + con_decision[1][0]) / con_decision.sum())*100
print(error_rate)







