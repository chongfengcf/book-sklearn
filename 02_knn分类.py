import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([
    [158, 64],
    [170, 86],
    [183, 84],
    [191, 80],
    [155, 49],
    [163, 59],
    [180, 67],
    [158, 54],
    [170, 67]
])

y_train = ['male', 'male', 'male', 'male', 'female', 'female', 'female', 'female', 'female']

'''数据可视化'''
plt.figure()
plt.title('Human Heights and Weights by Sex')
plt.xlabel('Height in cm')
plt.ylabel('Weight in kg')
for i, x in enumerate(X_train):
    plt.scatter(x[0], x[1], c='k', marker='x' if y_train[i] == 'male' else 'D')
plt.grid(True)
plt.show()

'''手动预测对象'''
# 身高155 体重70
x = np.array([155, 70])
# 计算这个点到所有点的距离
distances = np.sqrt(np.sum((X_train - x)**2, axis=1))
# 按距离排序取前k个，这里的k为3
# argsort()是排序后返回索引
nearest_neighbor_indices = distances.argsort()[:3]
# 获得最近k个人的性别
# take()是按索引取数组中的值
nearest_neighbor_genders = np.take(y_train, nearest_neighbor_indices)

from collections import Counter
b = Counter(nearest_neighbor_genders)
# most_common()返回最数组中最频繁的项，即预测的结果
print(b.most_common(1)[0][0])

'''sklearn实现'''
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier

# LabelBinarizer()标签二值化
lb = LabelBinarizer()
# fit()做转换准备 transform()进行转换 fit_transform()同时调用前面两个方法
# 训练集应该用fit_transform
y_train_binarized = lb.fit_transform(y_train)
# 一般设置为奇数防止平局情况
K = 3

clf = KNeighborsClassifier(n_neighbors=K)
clf.fit(X_train, y_train_binarized.reshape(-1))
prediction_binarized = clf.predict(np.array([155, 70]).reshape(1, -1))[0]
# inverse_transform()逆向转换为字符串标签
predicted_label = lb.inverse_transform(prediction_binarized)
print(predicted_label)

'''批量预测'''
X_test = np.array([
    [168, 65],
    [180, 96],
    [160, 52],
    [169, 67]
])

y_test = ['male', 'male', 'female', 'female']
# 测试集应该用transform
y_test_binarized = lb.transform(y_test)
# .T是转置
print('Binarized labels: %s' % y_test_binarized.T[0])
predictions_binarized = clf.predict(X_test)
print('Binarized predictions: %s' % predictions_binarized)
print('Predicted labels: %s' % lb.inverse_transform(predictions_binarized))

'''模型评价'''
# 准确率(正确分类的比例)
from sklearn.metrics import accuracy_score
print('Accuracy: %s' % accuracy_score(y_test_binarized, predictions_binarized))

# 精准率(实际是男性/预测是男性)
from sklearn.metrics import precision_score
print('Precision %s' % precision_score(y_test_binarized, predictions_binarized))

# 召回率(预测是男性/实际是男性)
from sklearn.metrics import recall_score
print('Recall: %s' % recall_score(y_test_binarized, predictions_binarized))

# F1得分(精准率和召回率的调和平均值)
# 若精准率和召回率差异过大则评分会下降
from sklearn.metrics import f1_score
print('F1 score: %s' % f1_score(y_test_binarized, predictions_binarized))

# 马修斯相关系数(MCC) 完美分类器1 随机分类器0 完全错误分类器-1
from sklearn.metrics import matthews_corrcoef
print('Matthews correlation coefficient: %s' % matthews_corrcoef(y_test_binarized, predictions_binarized))

# sklearn自带综合平均价函数
from sklearn.metrics import classification_report
print(classification_report(y_test_binarized, predictions_binarized,
                            target_names=['male'], labels=[1]))