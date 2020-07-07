import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  train_test_split

data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train.reshape(-1,1))
X_test = X_scaler.transform(X_test)
y_test = y_scaler.transform(y_test.reshape(-1,1))
regressor = SGDRegressor(loss='squared_loss')
scores = cross_val_score(regressor, X_train, y_train.ravel(), cv=5)
print('Cross validation r-squared scores: %s' % scores)
print('Average cross validation r-squared score: %s' % np.mean(scores))
regressor.fit(X_train, y_train.ravel())
print('Test set r-squared score %s' % regressor.score(X_test, y_test))
