##gaussian_nb
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Sonar")

sonar = pd.read_csv("Sonar.csv")
dum_sonar = pd.get_dummies(sonar, drop_first=True)

X = dum_sonar.drop('Class_R', axis=1)
y = dum_sonar['Class_R']

####### K-Folds CV
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
nb = GaussianNB()
results = cross_val_score(nb, X, y, scoring='roc_auc', cv=kfold)
print(results)
print(results.mean())

## HR
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\human-resources-analytics")
hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)

X = dum_hr.drop('left', axis=1)
y = dum_hr['left']

####### K-Folds CV
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
nb = GaussianNB()
results = cross_val_score(nb, X, y, scoring='roc_auc', cv=kfold)
print(results)
print(results.mean())

############################### Bankruptcy #############################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Bankruptcy")

bruptcy = pd.read_excel("Bankruptcy.xlsx", sheet_name=2, index_col=0)
X = bruptcy.drop(['D','YR'], axis=1)
y = bruptcy['D']

####### K-Folds CV with Naive Bayes
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
nb = GaussianNB()
results = cross_val_score(nb, X, y, scoring='roc_auc', cv=kfold)
print(results.mean())


####### Grid Search with k-NN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

scaler = StandardScaler()
knn = KNeighborsClassifier()
pipe = Pipeline([('scl_std', scaler), ('knn_model', knn)])

params = {'knn_model__n_neighbors':np.arange(1,31,2)}
gcv = GridSearchCV(pipe, param_grid=params,scoring='roc_auc',
                          cv = kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

pd_gcv = pd.DataFrame(gcv.cv_results_)

############ on test data

best_model = gcv.best_estimator_

tst_brupt = pd.read_csv("testBankruptcy.csv",index_col=0)

y_pred = best_model.predict(tst_brupt)
############## Image Segmentation Data #################

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Image Segmentation")

from sklearn.preprocessing import LabelEncoder

imag_seg = pd.read_csv('Image_Segmention.csv')

X = imag_seg.drop('Class', axis=1)
y = imag_seg['Class']

le = LabelEncoder()
le_y = le.fit_transform(y)
print(le.classes_)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)

scaler = StandardScaler()
knn = KNeighborsClassifier()
pipe = Pipeline([('scl_std', scaler), ('knn_model', knn)])

params = {'knn_model__n_neighbors':np.arange(1,31)}
gcv = GridSearchCV(pipe, param_grid=params,scoring='neg_log_loss',
                          cv = kfold)
gcv.fit(X, le_y)
print(gcv.best_params_)
print(gcv.best_score_)

pd_gcv = pd.DataFrame(gcv.cv_results_)

####### K-Folds CV with Naive Bayes
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
nb = GaussianNB()
results = cross_val_score(nb, X, le_y, scoring='neg_log_loss', cv=kfold)
print(results.mean())

#### on test data
tst_img = pd.read_csv("tst_img.csv")
best_model = gcv.best_estimator_

y_pred = best_model.predict(tst_img)
le.inverse_transform(y_pred)
