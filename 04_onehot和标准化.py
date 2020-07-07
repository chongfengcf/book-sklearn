from sklearn.feature_extraction import DictVectorizer

'''one-hot编码'''
onehot_encoder = DictVectorizer()
X = [
    {'city': 'New York'},
    {'city': 'San Francisco'},
    {'city': 'Chapel Hill'}
]

print(onehot_encoder.fit_transform(X).toarray())

'''特征标准化'''
# 等同于StandardScaler
from sklearn import preprocessing
import numpy as np
X = np.array([
    [0., 0., 5., 13., 9., 1.],
    [0., 0., 13., 15., 10., 15.],
    [0., 3., 15., 2., 0., 11.]
])
print(preprocessing.scale(X))

# 能更好的处理异常值
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)