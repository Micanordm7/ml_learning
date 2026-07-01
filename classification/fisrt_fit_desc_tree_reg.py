from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor


housing = fetch_california_housing(as_frame=True)

data, target = housing.data, housing.target

""" Viewing the the content of the dataset 
#showing the data
#print(data)

#showing the data
#print(target)

#description of the dataset
print(housing.DESCR)
"""

""" TRAINING THE MODEL"""
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(data, target)

""" Model VERIFICATION"""
print(regressor.score(data, target))




