## bernaulli_nb.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Telecom")

telecom = pd.read_csv("Telecom.csv")
dum_tel = pd.get_dummies(telecom, drop_first=True)

X = dum_tel.drop('Response_Y', axis=1)
y = dum_tel['Response_Y']

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=2022,
                                                    test_size=0.3,
                                                    stratify=y)
nb = BernoulliNB()
# Calculate all the apriori probabilities
nb.fit(X_train, y_train)

y_pred_prob = nb.predict_proba(X_test)[:,1]

print(roc_auc_score(y_test, y_pred_prob))

### on test
tst_tel = pd.read_csv("testTelecom.csv")
dum_tst = pd.get_dummies(tst_tel, drop_first=True)

predict_probs = nb.predict_proba(dum_tst)
predictions = nb.predict(dum_tst)
np.sum(predictions)

####### K-Folds CV
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
nb = BernoulliNB()
results = cross_val_score(nb, X, y, scoring='roc_auc', cv=kfold)
print(results)
print(results.mean())

# Building the model on whole data
nb.fit(X,y)
predict_probs = nb.predict_proba(dum_tst)
predictions = nb.predict(dum_tst)
np.sum(predictions)

############# Cancer Data #######################

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Cancer")

cancer = pd.read_csv("Cancer.csv", index_col=0)
dum_cancer = pd.get_dummies(cancer, drop_first=True)

X = dum_cancer.drop('Class_recurrence-events', axis=1)
y = dum_cancer['Class_recurrence-events']

####### K-Folds CV
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
nb = BernoulliNB()
results = cross_val_score(nb, X, y, scoring='roc_auc', cv=kfold)
print(results)
print(results.mean())

