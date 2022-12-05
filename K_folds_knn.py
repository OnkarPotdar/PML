
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
########### Breast Cancer ################

b_cancer=pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Wisconsin\BreastCancer.csv", 
                     index_col=0)

dum_b_cancer = pd.get_dummies(b_cancer, drop_first=True)
X = dum_b_cancer.drop('Class_Malignant',axis=1)
y = dum_b_cancer['Class_Malignant']

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
knn = KNeighborsClassifier(n_neighbors=3)
results = cross_val_score(knn, X, y ,scoring='roc_auc',
                          cv = kfold)
print(results)
print(results.mean())

roc = []
ks = np.arange(1,30,2)
for i in ks:
    knn = KNeighborsClassifier(n_neighbors=i)
    results = cross_val_score(knn, X, y ,scoring='roc_auc',
                              cv = kfold)
    roc.append(results.mean())

i_max = np.argmax(roc)
print("Best k =", ks[i_max], "with score =", np.max(roc))    

############# HR

df=pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\human-resources-analytics\HR_comma_sep.csv")

dum_df = pd.get_dummies(df, drop_first=True)
X = dum_df.drop('left',axis=1)
y = dum_df['left']

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)

roc = []
ks = np.arange(1,30,2)
for i in ks:
    knn = KNeighborsClassifier(n_neighbors=i)
    results = cross_val_score(knn, X, y ,scoring='roc_auc',
                              cv = kfold)
    roc.append(results.mean())

i_max = np.argmax(roc)
print("Best k =", ks[i_max], "with score =", np.max(roc))    

############### Grid Search CV
from sklearn.model_selection import GridSearchCV

b_cancer=pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Wisconsin\BreastCancer.csv", 
                     index_col=0)
dum_b_cancer = pd.get_dummies(b_cancer, drop_first=True)
X = dum_b_cancer.drop('Class_Malignant',axis=1)
y = dum_b_cancer['Class_Malignant']

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
knn = KNeighborsClassifier()

params = {'n_neighbors':np.arange(1,30,2)}
gcv = GridSearchCV(knn, param_grid=params,scoring='roc_auc',
                          cv = kfold, verbose=3)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

pd_gcv = pd.DataFrame(gcv.cv_results_)

############### Image Segementation
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Image Segmentation")

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

pd_gcv = pd.DataFrame(gcv.cv_results_)
