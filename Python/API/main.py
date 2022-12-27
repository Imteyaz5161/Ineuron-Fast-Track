
from flask import  Flask , request , jsonify

app = Flask(__name__)


@app.route('/xyz',methods=['GET','POST'])
def test():
    if(request.method == 'POST'):
        a = request.json['num1']
        b = request.json['num2']
        result = a + b
        return jsonify(str(result))

@app.route('/abc/sudh',methods=['POST'])
def test1():
    if(request.method == 'POST'):
        a = request.json['num3']
        b = request.json['num4']
        result = a + b
        return jsonify(str(result))

@app.route('/abc/sudh/kumar',methods=['POST'])
def test2():
    if(request.method == 'POST'):
        a = request.json['num4']
        b = request.json['num5']
        result = a + b
        return jsonify(str(result))

@app.route('/abcxyz',methods=['POST'])
def test3():
    if(request.method == 'POST'):
        a = request.json['sudh']
        b = request.json['kumar']
        result = a + b
        return jsonify(str(result))
from flask import Flask,request, jsonify

app= Flask(__name__)

@app.route('/xyz',methods=['GET','POST'])
def test():
     if(request.method==['POST']):
         a=request.json['num 1']
         b=request.json['num 2']
         result =a+b
         return jsonify(str(result))
if __name__=='__main__':
    app.run()

if __name__ == '__main__':
    app.run()


"""1 . write a function to fetch data from sql table via api 
2 . write a functoin to fetch a data from mongodb table """