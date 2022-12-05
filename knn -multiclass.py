###knn -multiclass
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Image Segmentation")

################ Image Segmentation #################
imag_seg = pd.read_csv('Image_Segmention.csv')

X = imag_seg.drop('Class', axis=1)
y = imag_seg['Class']

le = LabelEncoder()
le_y = le.fit_transform(y)
print(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, le_y,random_state=2022,
                                                    test_size=0.3,
                                                    stratify=le_y)
scaler = MinMaxScaler()
knn = KNeighborsClassifier(n_neighbors=3)
pipe = Pipeline([('scl_mm', scaler), ('knn_model', knn)])
pipe.fit(X_train, y_train)

# Log loss
y_pred_prob = pipe.predict_proba(X_test)

print(log_loss(y_test, y_pred_prob))

losses = []

for i in np.arange(1,31):
    knn = KNeighborsClassifier(n_neighbors=i)
    pipe = Pipeline([('scl_mm', scaler), ('knn_model', knn)])
    pipe.fit(X_train, y_train)

    # Log loss
    y_pred_prob = pipe.predict_proba(X_test)

    print("k=",i,"Log Loss=", log_loss(y_test, y_pred_prob))
    losses.append(log_loss(y_test, y_pred_prob))

np.min(losses)
np.argmin(losses)

print("Best K =", np.argmin(losses)+1, "with log loss=",np.min(losses))

best_k = np.argmin(losses)+1

tst_img = pd.read_csv("tst_img.csv")
knn = KNeighborsClassifier(n_neighbors=best_k)
pipe = Pipeline([('scl_mm', scaler), ('knn_model', knn)])
pipe.fit(X_train, y_train)

y_pred = pipe.predict(tst_img)
print(le.classes_)

le.inverse_transform(y_pred)

################## Vehicle #######################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Vehicle Silhouettes")

################ Image Segmentation #################
vehicle = pd.read_csv('Vehicle.csv')

X = vehicle.drop('Class', axis=1)
y = vehicle['Class']

le = LabelEncoder()
le_y = le.fit_transform(y)
print(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, le_y,random_state=2022,
                                                    test_size=0.3,
                                                    stratify=le_y)
scaler = MinMaxScaler()
losses = []

for i in np.arange(1,31):
    knn = KNeighborsClassifier(n_neighbors=i)
    pipe = Pipeline([('scl_mm', scaler), ('knn_model', knn)])
    pipe.fit(X_train, y_train)

    # Log loss
    y_pred_prob = pipe.predict_proba(X_test)

    print("k=",i,"Log Loss=", log_loss(y_test, y_pred_prob))
    losses.append(log_loss(y_test, y_pred_prob))

print("Best K =", np.argmin(losses)+1, "with log loss=",np.min(losses))
