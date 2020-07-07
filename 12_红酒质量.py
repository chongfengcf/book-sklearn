import pandas as pd
import matplotlib.pylab as plt

df = pd.read_csv('data/winequality-red.csv', sep=';')
print(df.describe())

'''评估质量与酒精的关系'''
plt.scatter(df['alcohol'], df['quality'])
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Alcohol Against Quality')
plt.show()

print(df.corr())

'''拟合和评估模型'''
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv('data/winequality-red.csv', sep=';')
X = df[list(df.columns)[:-1]]
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predictions = regressor.predict(X_test)
print('R-squared: %s' % regressor.score(X_test, y_test))

'''交叉验证'''
from sklearn.model_selection import cross_val_score
regressor = LinearRegression()
scores = cross_val_score(regressor, X, y, cv=5)
print(scores.mean())
print(scores)

'''结果画图'''
plt.scatter(y_test, y_predictions)
plt.xlabel('True')
plt.ylabel("Predicted")
plt.title('Predicted Against True')
plt.show()
