# encoding:utf-8
import jieba
import jieba.posseg as pseg
import sklearn.feature_extraction.text
import json
from scipy import sparse, io
from sklearn.externals import joblib

# 非 tf-idf 词向量
class Counter_Vectorizer(sklearn.feature_extraction.text.CountVectorizer):
    def build_analyzer(self):
        def analyzer(doc):
            # 去标点
            words = pseg.cut(doc)
            new_doc = ''.join(w.word for w in words if w.flag != 'x')
            words = jieba.cut(new_doc)
            return words
        return analyzer
# 用tf-idf生成词向量
class TfidfVectorizer(sklearn.feature_extraction.text.TfidfVectorizer):
    def build_analyzer(self):
        # 生成词向量前需要进行切词
        def analyzer(doc):
            # 将标点符号去掉
            words = pseg.cut(doc)
            new_doc = ''.join(w.word for w in words if w.flag != 'x')
            words = jieba.cut(new_doc)
            return words
        return analyzer


# 生成词向量并进行存储
def vector_word():
    with open('RawData/train_content_5000.json', 'r') as f:
        content = json.load(f)
    with open('RawData/train_label_5000.json', 'r') as f:
        label = json.load(f)

    vec_tfidf = TfidfVectorizer(min_df=2, max_df=0.8,max_features=2000)
    tfidf = vec_tfidf.fit(content)
    # 存储分词模型
    # joblib.dump(tfidf,'model/word_vector_model_60w.pkl')
    data_tfidf = tfidf.transform(content)
    data_tfidf_dense = data_tfidf.todense()
    name_tfidf_feature = vec_tfidf.get_feature_names()
    io.mmwrite('XGBoost/word_vector/word_vector.mtx', data_tfidf)
    '''
    # 稀疏矩阵存储
    io.mmwrite('word_vector/word_vector.mtx', data_tfidf)

    with open('word_vector/train_label.json', 'w') as f:
        json.dump(label, f)
    # 存入特征词
    with open('word_vector/vector_type.json', 'w') as f:
        json.dump(name_tfidf_feature, f)
    '''

def dispose_new_doc():
    tfidf = joblib.load('word_vector_model.pkl')
    doc = '您好！紫荆x号本周日x日妇女节有活动，女士到场都有花送，小孩有礼物，下午x:xx还会有抽奖活动哦，有兴趣可过来玩噢！联系人:黄秀秀。x'
    transform_document = [doc]
    new_data_tfidf = (tfidf.transform(transform_document)).todense()
    print (new_data_tfidf)

if '__main__' == __name__:
    vector_word()
    print ('word_vector Finish')
    # dispose_new_doc()
