# encoding:utf-8
import numpy as np
from word_vector import Counter_Vectorizer,TfidfVectorizer
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


class svm_classifier:
    def __init__(self,content,label):
        self.train_data,self.test_data,self.train_label,self.test_label = \
            train_test_split(content,label,test_size=0.1,random_state=0)
        # self.clf = svm.svc()
        # 创建模型
        # self.model = Pipeline([('word_vec',TfidfVectorizer()),('medial',intermediate()),
        #                        ('pca',PCA()),('clf',SVC())])

        self.model = Pipeline([('word_vec', TfidfVectorizer(min_df=2,max_df=0.8)), ('clf', SVC())])
        # self.model = Pipeline([('word_vec', TfidfVectorizer(min_df=2,max_df=0.8,max_features=2000)),
        #                        ('clf', SVC(C=1,kernel='linear'))])

    def adjust_paras(self):
        parameters_dict = [{'word_vec__max_features':np.arange(1000,3000,500),
                            'clf__kernel':['linear'],'clf__C':np.logspace(-2,10,13)},
                           {'word_vec__max_features': np.arange(1000, 3000, 500),
                            'clf__kernel': ['rbf'], 'clf__C': np.logspace(-2, 10, 13),'clf__gamma':np.logspace(-9, 3, 13)}
                           ]
        cv = StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
        print 'shufflesplit finish'
        grid = GridSearchCV(self.model,parameters_dict,cv=cv,n_jobs=2)
        print 'start fitting'
        grid.fit(self.train_data,self.train_label)
        print 'fitting finish'
        # 获取最优模型参数
        best_paras = dict(grid.best_estimator_.get_params())
        for paras_name in best_paras.keys():
            print '\t%s : %r' % (paras_name,best_paras[paras_name])
            # 将最优参数设置到模型
            self.model.set_params(paras_name = best_paras[paras_name])
        print 'the best score : %.2f' % grid.best_score_

    def model_train(self):
        # 使用最优超参进行训练
        self.model.fit(self.train_data,self.train_label)
        print 'model train finish'
        # 存储模型
        joblib.dump(self.model,'model/SVM_model.pkl')
        print 'model write finish'
        # 测试模型
        train_result = self.model.predict(self.train_data)
        print metrics.classification_report(self.train_label,train_result)

if __name__ == '__main__':
    # 加载数据
    with open('RawData/train_content.json','r') as f:
        content = json.load(f)
    with open('RawData/train_label.json','r') as f:
        label = json.load(f)
    classifier = svm_classifier(content,label)
    classifier.adjust_paras()
    classifier.model_train()