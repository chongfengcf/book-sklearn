corpus = [
    'UNC played Duke in basketball',
    'Duck lost the basketball game'
]

'''文档编码'''
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)

corpus.append('I ate a sandwich')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)

'''文档距离比较'''
from sklearn.metrics.pairwise import euclidean_distances
X = vectorizer.fit_transform(corpus).todense()
print('Distance between 1st and 2nd documents:',
      euclidean_distances(X[0], X[1]))
print('Distance between 1st and 3rd documents:',
      euclidean_distances(X[0], X[2]))
print('Distance between 2nd and 3rd documents:',
      euclidean_distances(X[1], X[2]))

'''停用词过滤'''
# 使用基本英语停用词列表
vectorizer = CountVectorizer(stop_words='english')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)

'''词干提取和词形还原'''
corpus = [
    'He ate the sandwiches',
    'Every sandwich was eaten by him'
]

# 如果binary=True，非零的n将全部置为1
vectorizer = CountVectorizer(binary=True, stop_words='english')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)

import nltk
nltk.download('wordnet')

# 使用NLTK进行词干提取和词形还原
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('gathering', 'v'))
print(lemmatizer.lemmatize('gathering', 'n'))

# 不考虑词性
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
print(stemmer.stem('gathering'))

'''对语料库做词形还原'''
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag

wordnet_tags = ['n', 'v']
corpus = [
    'He ate the sandwiches',
    'Every sandwich was eaten by him'
]
stemmer = PorterStemmer()
print('Stemmed:', [[stemmer.stem(token) for token in word_tokenize(document)] for document in corpus])

def lemmatize(token, tag):
    if tag[0].lower in ['n', 'v']:
        return lemmatizer.lemmatize(token, tag[0].lower(0))
    return token

lemmatizer = WordNetLemmatizer()
tagged_corpus = [pos_tag(word_tokenize(document)) for document in corpus]
print('Lemmatized:', [[lemmatize(token, tag) for token, tag in document] for document in tagged_corpus])