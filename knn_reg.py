## knn_reg
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Medical Cost Personal")

df = pd.read_csv("insurance.csv")
dum_df = pd.get_dummies(df, drop_first=True)

X = dum_df.drop('charges', axis=1)
y = dum_df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=2022,
                                                    test_size=0.3)
scaler = StandardScaler()
knn = KNeighborsRegressor(n_neighbors=3)
pipe = Pipeline([('scl_std', scaler), ('knn_model', knn)])
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred)) 


## try k = 1 to 30
r2 = []

for i in np.arange(1,31):
    knn = KNeighborsRegressor(n_neighbors=i)
    pipe = Pipeline([('scl_std', scaler), ('knn_model', knn)])
    pipe.fit(X_train, y_train)

    # Log loss
    y_pred = pipe.predict(X_test)

    print("k=",i,"R2=", r2_score(y_test, y_pred))
    r2.append(r2_score(y_test, y_pred))

np.max(r2)
np.argmax(r2)

print("Best K =", np.argmax(r2)+1, "with r2 =",np.max(r2))

best_k = np.argmax(r2)+1

# Predicting on test set with no response variable
tst_insure = pd.read_csv("tst_insure.csv")
dum_tst = pd.get_dummies(tst_insure, drop_first=True)
knn = KNeighborsRegressor(n_neighbors=best_k)
pipe = Pipeline([('scl_std', scaler), ('knn_model', knn)])
pipe.fit(X_train, y_train)

y_pred = pipe.predict(dum_tst)

#################### Concrete Strength #####################


os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Concrete Strength")

df = pd.read_csv("Concrete_Data.csv")

X = df.drop('Strength', axis=1)
y = df['Strength']

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=2022,
                                                    test_size=0.3)
scaler = StandardScaler()

## try k = 1 to 30
r2 = []

for i in np.arange(1,31):
    knn = KNeighborsRegressor(n_neighbors=i)
    pipe = Pipeline([('scl_std', scaler), ('knn_model', knn)])
    pipe.fit(X_train, y_train)

    # Log loss
    y_pred = pipe.predict(X_test)

    print("k=",i,"R2=", r2_score(y_test, y_pred))
    r2.append(r2_score(y_test, y_pred))

np.max(r2)
np.argmax(r2)

print("Best K =", np.argmax(r2)+1, "with r2 =",np.max(r2))

best_k = np.argmax(r2)+1

# Predicting on test set with no response variable
tst_conc = pd.read_csv("testConcrete.csv")
knn = KNeighborsRegressor(n_neighbors=best_k)
pipe = Pipeline([('scl_std', scaler), ('knn_model', knn)])
pipe.fit(X_train, y_train)

y_pred = pipe.predict(tst_conc)
