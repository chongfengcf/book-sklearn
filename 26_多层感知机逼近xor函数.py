from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

y = [0, 1, 1, 0]
X = [[0, 0], [0, 1], [1, 0], [1, 1]]

clf = MLPClassifier(solver='lbfgs', activation='logistic',
                    hidden_layer_sizes=(2,), random_state=20)
clf.fit(X, y)
predictions = clf.predict(X)
print('Accuracy: %s' %clf.score(X, y))
for i, p in enumerate(predictions):
    print('True: %s, predicted: %s' % (y[i], p))

print('Weights connecting the input layer and the hidden layer: \n%s' % clf.coefs_[0])
print('Hidden layer bias weights: \n%s' % clf.intercepts_[0])
print('Weights connecting the hidden layer and the output layer: \n%s' % clf.coefs_[1])
print('Output layer bias weight: \n%s' % clf.intercepts_[1])
