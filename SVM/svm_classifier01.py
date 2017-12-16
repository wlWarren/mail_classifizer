# encoding:utf-8
import sys,os
import time
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.pipeline import *
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split
import json
from scipy import sparse
import pandas as pd
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/'+'../'))
from word_vector import Counter_Vectorizer,TfidfVectorizer

class svm_classifier:
    def __init__(self,content,label):
        self.train_data,self.test_data,self.train_label,self.test_label = \
            train_test_split(content,label,test_size=0.2,random_state=0)
        # self.clf = svm.svc()
        # 创建模型
        self.model = Pipeline([('clf', SVC())])

    # 模型初始化
    def model_init(self):
        # 训练集切词
        tf_idf_Vector = TfidfVectorizer(min_df=2,max_df=0.8,max_features=2000)
        train_data = tf_idf_Vector.fit_transform(self.train_data)
        test_data = tf_idf_Vector.fit_transform(self.test_data)
        print 'TfidfVectorizer__fit_transform finish'
        # PCA 特征提取
        # pca = PCA(n_components=1000,copy=True,whiten=True)
        # pca.fit(data_tfidf.todense())
        # train_data_pca = sparse.csr_matrix(pca.transform(self.train_data))
        # test_data_pca = sparse.csr_matrix(pca.transform(self.test_data))
        # print 'PCA finish'

        # return train_data_pca,test_data_pca
        return train_data,test_data
    # 调节超参数
    def adjust_paras(self):
        # train_data_pca, test_data_pca = self.model_init()
        train_data,test_data = self.model_init()
        parameters_dict = [{'clf__kernel':['linear'],'clf__C':np.logspace(-5,5,11,base=2)},
                           {'clf__kernel': ['rbf'], 'clf__C': np.logspace(-5,5,11,base=2),
                            'clf__gamma':np.logspace(-5,5,11,base=2)}
                           ]
        # parameters_dict = {'clf__kernel': ['linear'], 'clf__C':[1,2]}

        cv = StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
        print 'shufflesplit finish'
        grid = GridSearchCV(self.model,parameters_dict,cv=cv,n_jobs=4)
        print 'start fitting'
        # grid.fit(self.train_data,self.train_label)
        grid.fit(train_data,self.train_label)
        print 'fitting finish'

        # 写入超参数
        hyper_paras = pd.DataFrame.from_dict(grid.cv_results_)
        with open('hyper_params.csv', 'w') as hyper_paras_f:
            hyper_paras.to_csv(hyper_paras_f)
        print 'hyper_params write finish'

        # 存储模型
        joblib.dump(self.model, 'SVM_model.pkl')
        print 'model write finish'

        # 打印最优模型参数
        print 'params details'
        best_paras = dict(grid.best_estimator_.get_params())
        for paras_name in best_paras.keys():
            print '\t%s : %r' % (paras_name,best_paras[paras_name])
        print 'the best score : %.2f' % grid.best_score_

        print "Grid scores on train set:"
        for params, mean_score, scores in grid.grid_scores_:
            print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params)

        # 打印测试信息
        print 'The scores are computed on test set'
        test_result = grid.predict(test_data)
        print metrics.classification_report(self.test_label,test_result)



    def model_train(self):
        # train_data, test_data = self.model_init()
        word_vector_model = joblib.load('model/word_vector_model_60w.pkl')
        print 'word_vector_model load finish'
        train_data = word_vector_model.transform(self.train_data)
        print 'train_data transform finish'
        test_data = word_vector_model.transform(self.test_data)
        print 'test_data transform finish'
        print 'start training SVM'
        start_time = time.time()
        model = svm.SVC(C=8.0,kernel='rbf',gamma=0.25)
        # 使用最优超参进行训练
        model.fit(train_data,self.train_label)
        print 'model train finish'
        print time.time()-start_time
        # 存储模型
        joblib.dump(model,'model/SVM_model_train_60w.pkl')
        print 'model write finish'
        # 测试模型
        train_result = model.predict(test_data)
        print metrics.classification_report(self.test_label,train_result)

if __name__ == '__main__':
    # 加载数据
    with open('../RawData/train_content_60w.json','r') as f:
        content = json.load(f)
    with open('../RawData/train_label_60w.json','r') as f:
        label = json.load(f)
    classifier = svm_classifier(content,label)
    # classifier.adjust_paras()
    classifier.model_train()