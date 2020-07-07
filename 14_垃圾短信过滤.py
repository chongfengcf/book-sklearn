import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv('data/SMSSpamCollection', delimiter='\t', header=None)
print(df.head())
print('Number of spam message: %s' % df[df[0] == 'spam'][0].count())
print('Number of ham message: %s' % df[df[0] == 'ham'][0].count())

'''逻辑回归'''
X = df[1].values
y = df[0].values
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
for i, prediction in enumerate(predictions[:5]):
    print('Predicted: %s, message: %s' % (prediction, X_test_raw[i]))

'''混淆矩阵'''
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# y_test = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
# y_pred = [0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
y_pred = predictions
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
plt.matshow(confusion_matrix)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

'''准确率'''
scores = cross_val_score(classifier, X_train, y_train, cv=5)
print('Accuracies: %s' % scores)
print('Mean accuracy: %s' % np.mean(scores))

'''精准率和召回率'''
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
precisions = cross_val_score(classifier, X_train, y_train, cv=5, scoring='precision')
print('Precision: %s' % np.mean(precisions))
recalls = cross_val_score(classifier, X_train, y_train, cv=5, scoring='recall')
print('Recall: %s' % np.mean(recalls))

'''F1'''
f1s = cross_val_score(classifier, X_train, y_train, cv=5, scoring='f1')
print('F1 score: %s' % np.mean(f1s))

'''ROC曲线'''
from sklearn.metrics import roc_curve, auc
predictions = classifier.predict_proba(X_test)
false_positice_rate ,recall, thresholds = roc_curve(lb.transform(y_test), predictions[:, 1])
roc_auc = auc(false_positice_rate, recall)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positice_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('recall')
plt.xlabel('Fall-out')
plt.show()