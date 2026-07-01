# Import KNeighborsClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

churn_df = pd.read_csv("churn.csv")
#print(churn_df.head())
y = churn_df["Churn"].values
X = churn_df[["Account_Length", "CustServ_Calls"]].values

#show de characteictics of features
print(y.shape, X.shape)
