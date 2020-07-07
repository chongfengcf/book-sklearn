import pandas as pd
df = pd.read_csv('data/movie-reviews.tsv', header=0, delimiter='\t')
print(df.count())
print(df.head())
print(df['Phrase'].head(10))
print(df['Sentiment'].describe())
print(df['Sentiment'].value_counts())
print(df['Sentiment'].value_counts() / df['Sentiment'].count())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


X, y = df['Phrase'], df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
pipline = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(max_iter=150))
])
parameters = {
    'vect__max_df': (0.25, 0.5),
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__use_idf': (True, False),
    'clf__C': (0.1, 1, 10)
}
grid_search = GridSearchCV(pipline, parameters, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search.fit(X_train, y_train)
print('Best score: %0.3f' % grid_search.best_score_)
print('Bset Parameters set:')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('t%s: %r' % (param_name, best_parameters[param_name]))

predictions = grid_search.predict(X_test)
print('Accuracy: %s' % accuracy_score(y_test, predictions))
print('Confusion Matrix:')
print(confusion_matrix(y_test, predictions))
print('Classfication Report:')
print(classification_report(y_test, predictions))
