import numpy as np
import matplotlib.pyplot as plt

X = np.array([[6], [8], [10], [14], [18]]).reshape(-1, 1)
y = [7, 9, 13, 17.5, 18]

'''数据可视化'''
plt.figure()
plt.title('Pizza price plotted against diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.plot(X, y, 'k.')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.show()

'''sklearn实现'''
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

test_pizza = np.array([[12]])
predicted_price = model.predict(test_pizza)[0]
print('A 12" pizza should const: $%.2f' % predicted_price)

'''损失函数(残差平方和RSS)'''
print('Residual sum of squares: %.2f' % np.mean((model.predict(X) - y) ** 2))

'''计算方差'''
x_bar = X.mean()
print(x_bar)

# 注意我们在计算样本方差的时候将样本的数量减去1
# 这项技巧称为贝塞尔校正，它纠正了对样本中总体方差估计的偏差

variance = ((X - x_bar) ** 2).sum() / (X.shape[0] - 1)
print(variance)

# numpy自带计算方差方法 ddof可以设置贝塞尔校正
print(np.var(X, ddof=1))

'''计算协方差'''
# 转换为ndarray类型
y = np.array(y)
y_bar = y.mean()
# 将X转置，因为所有的操作都必须是行向量
# transpose()数组转置
covariance = np.multiply((X - x_bar).transpose(), y - y_bar).sum() / (X.shape[0] - 1)
print(covariance)

# numpy自带计算方法
# cov()返回协方差矩阵，我们需要cov(a,b)
# [ cov(a,a) cov(a,b) ]
# [ cov(b,a) cov(b,b) ]
print(np.cov(X.transpose(), y)[0][1])

'''计算模型的参数'''
# y = a + bx;
# b = cov(x,y) / var(x)
# a = y平均 - b * x平均
b = covariance / variance;
a = y_bar - b * x_bar;
print('模型的参数a=%.2f,b=%.2f' % (a, b))

'''模型评价'''
# R方等于皮尔森积差相关系数(PPMCC)的平方
# 这个比例不能大于1或者小于0
X_train = np.array([6, 8, 10, 14, 18]).reshape(-1, 1)
y_train = [7, 9, 13, 17.5, 18]

X_test = np.array([8, 9, 11, 16, 12]).reshape(-1, 1)
y_test = [11, 8.5, 15, 18, 11]

model =LinearRegression()
model.fit(X_train, y_train)
r_squared = model.score(X_test, y_test)
print(r_squared)