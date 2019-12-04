'''6_1 单词级的one-hot编码
2019_12_3
'''

import numpy as np

def Six_one():
    samples = ['The cat sat on the mat.', 'The dog ate my homework.']

    token_index = {}
    for sample in samples:
        for word in sample.split():
            if word not in token_index:
                token_index[word] = len(token_index) + 1

    #print(token_index)
    max_length = 10

    #创建一个三维数组
    results = np.zeros(shape=(len(samples),
                    max_length,
                    max(token_index.values()) + 1))

    #print(results)

    print(list(enumerate(samples)))
    #enumerate():函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = token_index.get(word)
            results[i, j, index] = 1.

    #print(results)
    print(results.shape)

'''6_2 字符级的one-hot编码
2019_12_3
'''
import string

def Six_two():
    samples = ['The cat sat on the mat.', 'The dog ate my homework.']
    #所有可打印的ASCII字符
    characters = string.printable
    token_index = dict(zip(range(1, len(characters) + 1), characters))
    print(token_index)
    max_length = 50
    results = np.zeros(shape=(len(samples),
                    max_length,
                    max(token_index.keys()) + 1))
    for i, sample in enumerate(samples):
        for j, character in enumerate(sample):
            index = token_index.get(character)
            results[i, j, index] = 1.
    #print(results)
    print(results.shape)

'''6_3 用Keras实现单词级的one-hot编码
2019_12_4
'''

from keras.preprocessing.text import Tokenizer

def Six_three():
    samples = ['The cat sat on the mat.', 'The dog ate my homework.']

    #创建一个分词器(tokenizer)，设置为只考虑前1000个最常见的单词
    tokenizer = Tokenizer(num_words=1000)
    #构建单词索引
    tokenizer.fit_on_texts(samples)

    #将字符串转换为整数索引组成的列表
    sequences = tokenizer.texts_to_sequences(samples)

    #也可以直接得到one-hot二进制表示。这个分词器也支持除one-hot编码外的其他向量化模式
    one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

    #找回单词索引
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

if __name__ == '__main__':
    #Six_one()
    #Six_two()
    Six_three()
    