#encoding:utf_8
import numpy as np
import jieba
import jieba.posseg as psg
from scipy.sparse import csr_matrix
from sklearn.externals import joblib
from word_vector import Counter_Vectorizer,TfidfVectorizer
import sys, os
'''
text = '一次价值xxx元王牌项目；可充值xxx元店内项目卡一张；可以参与V动好生活百分百抽奖机会一次！预约电话：xxxxxxxxxxx'
cut = jieba.cut(text)
for term in cut:
    print term
seg_cut = psg.cut(text)
for word,flag in seg_cut:
    print word+" /"+flag

x = np.array([[1, 2, 3],[3, 4, 3],[1, 0, 2],[0, 0, 1]], float)
z = csr_matrix(x)
print z.data
print 'ok'
'''
print sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/'+'../'))
