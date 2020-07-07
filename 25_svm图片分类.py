import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from PIL import Image

X = []
y = []
for path, subdirs, files in os.walk('data/chars74k/'):
    for filename in files:
        f = os.path.join(path, filename)
        target = filename[3: filename.index(('-'))]
        img = Image.open(f).convert('L').resize((30, 30),
                                                resample=Image.LANCZOS)
        X.append(np.array(img).reshape(900,))
        y.append(int(target))
X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=11)
pipline = Pipeline([
    ('clf', SVC(kernel='rbf', gamma=0.01, C=100))
])
parameters = {
    'clf__gamma': (0.01, 0.03, 0.1, 0.3, 1),
    'clf__C': (0.1, 0.3, 1, 3, 10, 30),
}

if __name__ == '__main__':
    grid_search = GridSearchCV(pipline, parameters, n_jobs=-1,
                               verbose=1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print('Best score: %0.3f' % grid_search.best_score_)
    print('Best parameters set:')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    predictions = grid_search.predict(X_test)
    print(classification_report(y_test, predictions))