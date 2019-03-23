from flask import Flask
from keras.preprocessing import image
import numpy as np
from app.main.predict import predict
import json
from flask import request
import os
app = Flask(__name__)
upload_folder = './app/photos'
if os.path.exists(upload_folder):
    pass
else:
    os.mkdir(upload_folder)
@app.route('/',methods=['GET','POST'])
def index():
    upload_file = request.files('file')
    file_name = upload_file.filename
    upload_file.save(os.path.join(upload_folder,file_name))
    img = image.load_img(os.path.join(upload_folder,file_name), target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    result = predict(x,model_path='./app/model/mobilenet_my_finetune_model_ep15_d3_lr0.0001.h5')
    return json.dumps(result)
if __name__== "__main__":
    app.run(host='0.0.0.0',port=5000)
