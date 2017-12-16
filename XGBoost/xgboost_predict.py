# encoding:utf-8
from sklearn.externals import joblib
import sys,os
import sklearn.feature_extraction.text
# from word_vector import Counter_Vectorizer,TfidfVectorizer
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/'+'../'))
from word_vector import TfidfVectorizer

class xgboost_predict():

    def __init__(self,classifier_model,word_vector_model):
        self.classifier_model = classifier_model
        self.word_vector_model = word_vector_model

    def test_data_predict(self,doc):
        doc_tfidf = self.word_vector_model.transform(doc)
        predict_result = self.classifier_model.predict(doc_tfidf)
        with open('predict_result/svm_predict.txt','w') as f:
            for i in range(len(predict_result)):
                f.writelines(predict_result[i])
        print 'SVM predict_result write finish '

    def single_sample_predict(self,doc):
        doc = [doc]
        single_sample_tfidf = self.word_vector_model.transform(doc)
        predict_result = self.classifier_model.predict(single_sample_tfidf)
        print predict_result

    def predict(self,operate_methods,single_sample=None,docment=None):
        if operate_methods == 'single':
            if single_sample == None:
                print 'You hava to provide single sample.No sample has been given.'
            else:
                self.single_sample_predict(single_sample)
        if operate_methods == 'set':
            if docment == None:
                print 'You hava to provide document set.No document set has given.'
            else:
                self.single_sample_predict(docment)

if __name__ == '__main__':
    classifier_model = joblib.load('model/XGBoost_model_2w.pkl')
    print 'classifier_model load finish'
    word_vector_model = joblib.load('model/word_vector_model.pkl')
    print 'word_vector_model load finish'
    predict = xgboost_predict(classifier_model,word_vector_model)
    while(True):
        doc = raw_input()
        if doc == 'exit':
            break
        predict.predict('single',doc)
