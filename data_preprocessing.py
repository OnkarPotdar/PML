### data preprocessing.....
import pandas as pd
import numpy as np
import os


os.chdir(r"C:\Training\Academy\Statistics (Python)\Datasets")

exp_salary = pd.read_csv("Exp_Salaries.csv")
print(exp_salary.info())

dum_exp = pd.get_dummies(exp_salary, drop_first=True)

jobsal = pd.read_csv("JobSalary2.csv")

# Count the NAs in each column
jobsal.isnull().sum()

# Total NAs
jobsal.isnull().sum().sum()

# Dropping NA rows
jobsal.dropna()
from sklearn.impute import SimpleImputer
# Constant Imputation
imputer = SimpleImputer(strategy="constant",fill_value=50)
imp_data=imputer.fit_transform(jobsal)
imp_pd_data = pd.DataFrame(imp_data, columns=jobsal.columns)

# Mean Imputation
imputer = SimpleImputer(strategy="mean")
imp_data=imputer.fit_transform(jobsal)
imp_pd_data = pd.DataFrame(imp_data, columns=jobsal.columns)

# Median Imputation
imputer = SimpleImputer(strategy="median")
imp_data=imputer.fit_transform(jobsal)
imp_pd_data = pd.DataFrame(imp_data, columns=jobsal.columns)

### Chem Proc
chem = pd.read_csv("ChemicalProcess.csv")

chem.isnull().sum()
chem.isnull().sum().sum()

# Median Imputation
imputer = SimpleImputer(strategy="median")
imp_data=imputer.fit_transform(chem)
imp_pd_data = pd.DataFrame(imp_data, columns=chem.columns)

imp_pd_data.isnull().sum().sum()

########## Scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

milk = pd.read_csv("milk.csv", index_col=0)
# Standard Scaler
scl_std = StandardScaler()
scl_std.fit(milk)
# Means
print(scl_std.mean_)
# Std Dev
print(scl_std.scale_)

trns_scl = scl_std.transform(milk)
# or
trns_scl = scl_std.fit_transform(milk)
type(trns_scl)

trns_scl_pd = pd.DataFrame(trns_scl,columns=milk.columns,
                           index=milk.index)

trns_scl_pd.mean()
trns_scl_pd.std()

# MinMax Scaler
scl_mm = MinMaxScaler()
scl_mm.fit(milk)
# Min
print(scl_mm.data_min_)
# Max
print(scl_mm.data_max_)

trns_scl = scl_mm.transform(milk)
# or
trns_scl = scl_mm.fit_transform(milk)
type(trns_scl)

trns_scl_pd = pd.DataFrame(trns_scl,columns=milk.columns,
                           index=milk.index)

trns_scl_pd.min()
trns_scl_pd.max()
