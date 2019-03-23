from flask import Flask
from app.main.predict import predict
import json
from flask import request
app = Flask(__name__)
@app.route('/',methods=['POST'])
def index():
    val = json.loads(request.files('file'))
    return '<h1>{}</h1>'.format(val)
    # l = predict(test_path='.\\test', model_path='.\\app\\model\\mobilenet_my_finetune_model_ep15_d3_lr0.0001.h5')
    # l_j = json.dumps(l)
    # return l_j
# print(json.loads(l_j))
#     #return '<h1>Hello world!</h>'
if __name__== "__main__":
    app.run(host='0.0.0.0',port=5000)
