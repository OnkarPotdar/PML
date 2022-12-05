###knn_class_scl
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\human-resources-analytics")
########### Breast Cancer ################

hr = pd.read_csv("HR_comma_sep.csv")

dum_hr = pd.get_dummies(hr, drop_first=True)

X = dum_hr.drop('left',axis=1)
y = dum_hr['left']

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=2022,
                                                    test_size=0.3,
                                                    stratify=y)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
scl_trn = scaler.fit_transform(X_train)
scl_tst = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(scl_trn, y_train)

# ROC AUC
y_pred_prob = knn.predict_proba(scl_tst)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

#### Min Max

scaler = MinMaxScaler()
scl_trn = scaler.fit_transform(X_train)
scl_tst = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(scl_trn, y_train)

# ROC AUC
y_pred_prob = knn.predict_proba(scl_tst)[:,1]

print(roc_auc_score(y_test, y_pred_prob))

######## Pipeline ##############
from sklearn.pipeline import Pipeline
scaler = MinMaxScaler()
knn = KNeighborsClassifier(n_neighbors=3)
pipe = Pipeline([('scl_mm', scaler), ('knn_model', knn)])
pipe.fit(X_train, y_train)

# ROC AUC
y_pred_prob = pipe.predict_proba(X_test)[:,1]

print(roc_auc_score(y_test, y_pred_prob))




