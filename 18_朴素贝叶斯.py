import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

'''威斯康星数据集'''
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=31)

lr = LogisticRegression(max_iter=1000)
nb = GaussianNB()

lr_scores = []
nb_scores = []

train_sizes = range(10, len(X_train), 25)

for train_size in train_sizes:
    X_slice, _, y_slice, _ = train_test_split(
        X_train, y_train, train_size=train_size, stratify=y_train, random_state=31
    )
    nb.fit(X_slice, y_slice)
    nb_scores.append(nb.score(X_test, y_test))
    lr.fit(X_slice, y_slice)
    lr_scores.append(lr.score(X_test, y_test))

plt.plot(train_sizes, nb_scores, label='Naive Bayes')
plt.plot(train_sizes, lr_scores, linestyle='--', label='Logistic Regression')
plt.title('Naive Bayes and Logistic Regression Accuracies')
plt.xlabel('Number of training instances')
plt.ylabel('Test set accuracy')
plt.legend()
plt.show()

'''匹马印第安数据集'''
df = pd.read_csv('data/PimaIndiansdiabetes.csv', header=0)
y = df.iloc[:, -1:].values.ravel()
X = df.iloc[:, 0:-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

lr = LogisticRegression(max_iter=1000)
nb = GaussianNB()
lr_scores = []
nb_scores = []

train_sizes = range(10, len(X_train), 10)
for train_size in train_sizes:
    X_slice, _, y_slice, _ = train_test_split(
        X_train, y_train, train_size=train_size, stratify=y_train, random_state=11
    )
    nb.fit(X_slice, y_slice)
    nb_scores.append(nb.score(X_test, y_test))
    lr.fit(X_slice, y_slice)
    lr_scores.append(lr.score(X_test, y_test))

plt.plot(train_sizes, nb_scores, label='Naive Bayes')
plt.plot(train_sizes, lr_scores, linestyle='--', label='Logistic Regression')
plt.title('Naive Bayes and Logistic Regression Accuracies')
plt.xlabel('Number of training instances')
plt.ylabel('Test set accuracy')
plt.legend()
plt.show()