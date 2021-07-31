
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor



#Load Dataset from csv file
data = pd.read_csv('Credit-card-dataset/creditcard.csv')
# print(data.head)
data = data.sample(frac=0.1, random_state=1)
#print(len(data.columns))

# plot histogram of each parameter
#data.hist(figsize=(20, 20))
#plt.show() 

fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

outlier_fraction = len(fraud) / float(len(valid))
#print(outlier_fraction)

#print(len(fraud))
#print(len(valid))

corrmat = data.corr()
fig = plt.figure(figsize=(12, 9))

sns.heatmap(corrmat, vmax = .8, square = True)
#plt.show()

columns = data.columns.tolist()

columns = [c for c in columns if c not in ['Class']]    
target =  "Class"

x =data[columns]
y = data[target]

#random state
state = 1

#define the outlier detection mothods
classifiers =  {
    "Isolation Forest": IsolationForest(max_samples=len(x), contamination= outlier_fraction,random_state=state ),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination= outlier_fraction )
}

n_outlier = len(fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(x)
        score_pred = clf.negative_outlier_factor_
    else:
        clf.fit(x)
        score_pred = clf.decision_function(x)
        y_pred = clf.predict(x)

    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] =1

    n_erros = (y_pred != y).sum()

    #Run classifier metrics
    print('{}: {}'.format(clf_name, n_erros))
    print(accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))
