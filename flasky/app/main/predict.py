from .Image_aug import ImgGen
from ..config import Config
from keras.models import load_model
import numpy as np
class_ = {0: '骨头', 1: '金属', 2: '电池', 3: '灯泡', 4: '化妆品', 5: '食品袋', 6: '果皮', 7: '报纸', 8: '塑料袋', 9: '纸巾'}
garbage_class = {'厨余垃圾': [0, 6], '可回收垃圾': [1, 7, 8], '有害垃圾': [3, 2, 4], '其它垃圾': [5, 9]}
def predict(test_path,model_path):
    img_gen = ImgGen(Config)
    model = load_model(model_path)
    container = []
    for image,file_name in img_gen.get_test_generator(test_path):
        goods = {}
        y = model.predict(image)
        result = np.argmax(y, 1)[0]
        p = np.max(y)
        goods['物品名称'] = file_name
        predicted = class_[result]
        goods['预测'] = predicted
        for key, value in garbage_class.items():
            if result in value:
                goods['类别'] = key
                goods['可能性'] = str(int(p*100)) +'%'
                container.append(goods)
                break
            else:
                continue
    return container