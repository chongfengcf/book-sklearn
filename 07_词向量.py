import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)

# 展示词向量
embedding = model.word_vec('cat')
print('Dimensions: %s' % embedding.shape)
print(embedding)

# 展示不同词向量之间的距离
print(model.similarity('cat', 'dog'))
print(model.similarity('cat', 'sandwich'))

# 找相似词 positive是相关的词 negative是不相关的词
print(model.most_similar(positive=['puppy', 'cat'], negative=['kitten'], topn=1))

for i in model.most_similar(positive=['saddle', 'painter'], negative=['palette'], topn=3):
    print(i)