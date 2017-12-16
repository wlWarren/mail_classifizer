# encoding:utf-8
from SVM import svm_predict
from flask import Flask,request,render_template
import sys,os
from sklearn.externals import joblib
import numpy as np
import jsonify
import sklearn.feature_extraction.text
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/'+'../'))
from word_vector import Counter_Vectorizer,TfidfVectorizer


app = Flask(__name__)
app.config.from_object('config')


@app.route('/',methods=['GET'])
def signin_form():
    return render_template('form.html')
#
# @app.route('/form',methods=['GET','POST'])
# def signin():
#     # username = request.form['username']
#     # password = request.form['password']
#     #print(request.form['note'])
#     #print request.form.get('note')
#     #console.log(request.form['username'])
#     #print(username)
#     # if username=='admin' and password=='1234':
#     #     return render_template('first.html', username=username)
#     #return render_template('form.html', message='Bad name or password', username=username)
#     classifier_model = joblib.load('K:/wwl/combine/SVM/model/SVM_model_train.pkl')
#     print ("classifier_model load finish")
#     word_vector_model = joblib.load('K:/wwl/combine/SVM/model/word_vector_model.pkl')
#     print ("word_vector_model load finish")
#     predict_result = svm_predict.svm_do(classifier_model, word_vector_model, request.form['note'])
#     print type(np.ndarray.tostring(predict_result))
#     return render_template('form.html', message=np.ndarray.tostring(predict_result))



@app.route('/form',methods=['GET','POST'])
def classify():
    classifier_model = joblib.load('K:/wwl/combine/SVM/model/SVM_model_train.pkl')
    print ("classifier_model load finish")
    word_vector_model = joblib.load('K:/wwl/combine/SVM/model/word_vector_model.pkl')
    print ("word_vector_model load finish")
    predict_result = svm_predict.svm_do(classifier_model, word_vector_model, request.form['note'])
    if cmp(np.ndarray.tostring(predict_result)[0], '0') == 0:
        result_text = "this is not garbage"
    elif cmp(np.ndarray.tostring(predict_result)[0], '1') == 0:
        result_text = "this is a garbage"
    # print result_text
    return render_template('form.html', message=result_text.decode())
    # return (np.ndarray.tostring(predict_result))
# @app.route('/form',methods=['POST'])
# def sendjson2():
#
#     # 接收前端发来的数据,转化为Json格式,我个人理解就是Python里面的字典格式
#     print request.get_data()
#     data = request.get_data()
#
#
#     # Output: {u'age': 23, u'name': u'Peng Shuang', u'location': u'China'}
#     # print data
#     # return jsonify(data)
#     return data

#
# if(__name__)=='__main__':
#     predict = svm_predict.svm_do()
#     app.run()



if __name__ == '__main__':
    app.run()
