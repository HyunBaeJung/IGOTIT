# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 01:44:28 2017

@author: Big Data Guru
"""

import os
from konlpy.tag import Okt  #twitter -> Okt로 변경. 이 녀석이 품사태깅해주는 라이브러리
import warnings # detected Windows; aliasing chunkize to chunkize_serial 에러 발생 시 추가 
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim 
import tensorflow as tf
import numpy as np
import codecs

# 품사태깅 할 데이터 가져올 경로
os.chdir("C:\\Users\\Playdata\\Desktop\\프로젝트\\Word2Vec\\Sentimental-Analysis\\Word2Vec\\Movie_rating_data")

# 데이터 읽어오는 함수 정의 tsv형식으로 저장되어 \t으로 구분
def read_data(filename):    
    with open(filename, 'r',encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]        
        data = data[1:]   # header 제외 #    
    return data 


train_data = read_data('ratings_train.txt') 
test_data = read_data('ratings_test.txt') 


pos_tagger = Okt() #twitter -> Okt로 변경

# 토큰화 함수 정의 
def tokenize(doc):

    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]


## training Word2Vec model using skip-gram 
## skip-gram 방법으로 워2벡모델 학습 
tokens = [tokenize(row[1]) for row in train_data]
model = gensim.models.Word2Vec(size=300,sg = 1, alpha=0.025,min_alpha=0.025, seed=1234)
model.build_vocab(tokens)
    

for epoch in range(30):           
    model.train(tokens,epochs = model.iter, total_examples=model.corpus_count,)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
# You must specify either total_examples or total_words~ 어쩌고 에러
# total_examples 등의 추가 인자가 없어서 발생하는 에러인 것 같음
# 에러가 발생한 gensim 의 버전은 3.4.0 이다. 버전업 되면서 인자나 사용법이 변경된 것 같다.
# 참고- http://daewonyoon.tistory.com/240


os.chdir("C:\\Users\\Playdata\\Desktop\\프로젝트\\Word2Vec\\Sentimental-Analysis\\Word2Vec")    
model.save('Word2vec.model')
model.most_similar('연기/Noun',topn = 20)  ## topn = len(model.wv.vocab)
# 입력받은 키워드와 가장 비슷한 임베딩을 가지는 단어를 topn개 만큼 출력