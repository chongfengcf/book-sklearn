from numpy.linalg import inv
from numpy import dot, transpose

'''手动求解Y=XB 求B参数值'''
X = [[1, 6, 2], [1, 8, 1], [1, 10, 0], [1, 14, 2], [1, 18, 0]]
y = [[7], [9], [13], [17.5], [18]]
print(dot(inv(dot(transpose(X), X)), dot(transpose(X), y)))

'''使用最小二乘函数求解'''
from numpy.linalg import lstsq
print(lstsq(X, y, rcond=None)[0])

'''多元线性回归'''
from sklearn.linear_model import LinearRegression
X = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
y = [[7], [9], [13], [17.5], [18]]
model = LinearRegression()
model.fit(X, y)
X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y_test = [[11], [8.5], [15], [18], [11]]
predictions = model.predict(X_test)
for i, prediction in enumerate(predictions):
    print('Predicted: %s, Target: %s,' % (prediction, y_test[i]))
    print('R-squared: %.2f' % model.score(X_test, y_test))