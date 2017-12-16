from flask import Flask, render_template, request, redirect, url_for
from os import path
from werkzeug.utils import secure_filename
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html', title='Welcome')

@app.route('/inde')
def inde():
    return render_template('inde.html')

@app.route('/services')
def services():
    return 'Service'

@app.route('/about')
@app.route('/About/')
def about():
    return 'About'

@app.route('/user/<username>')
def user1(username):
    return 'User %s' % username

@app.route('/user/<int:user_id>')
def user2(user_id):
    return 'User %d' % user_id

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method=='POST':
        username = request.form['username']
        password = request.form['password']
    else:
        username = request.args['username']
    return render_template('login.html', method=request.method)

@app.route('/upload', methods = ['GET' , 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = path.abspath(path.dirname(__file__))
        upload_path = path.join(basepath, 'static/uploads')
        f.save(upload_path, secure_filename(f.filename))
        return redirect(url_for('upload'))
    return render_template('upload.html')
if __name__ == '__main__':
    app.run(debug=True)
