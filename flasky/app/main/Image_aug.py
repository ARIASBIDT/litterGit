#------------------------图像增强处理   旋转剪切  让模型训练结果更好---------------------#
from keras.preprocessing import image
import os
import numpy as np
class ImgGen():
    def __init__(self,Config):
        self.img_size = Config.IMG_SIZE
        self.batch_size = Config.BATCH_SIZE

    def process(self,shear_range=0.2,zoom_range=0.2,horizontal_flip=True):
        self.train_datagen = image.ImageDataGenerator(
            rescale=1. / 255,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip)
        self.test_datagen = image.ImageDataGenerator(rescale=1. / 255)

    def get_train_generator(self,train_path):
        return self.train_datagen.flow_from_directory(train_path,
                                                            target_size=(self.img_size, self.img_size),
                                                            batch_size=self.batch_size * 5,
                                                            class_mode='categorical')
    def get_val_generator(self,val_path):
        return self.test_datagen.flow_from_directory(
            val_path,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical')

    def get_test_generator(self,test_path):
        for file in os.listdir(test_path):
            image_file = os.path.join(test_path,file)
            img = image.load_img(image_file, target_size=(self.img_size,self.img_size))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0
            yield x,file