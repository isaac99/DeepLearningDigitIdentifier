import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator
from VGG16 import VGG16
from DataReader import DataReader as reader

def main():
    # print('running')
    # training_data_generator = ImageDataGenerator()
    # training_data = training_data_generator.flow_from_directory(directory="./input/train", target_size=(224, 224))
    # testing_data_generator = ImageDataGenerator()
    # testing_data = testing_data_generator.flow_from_directory(directory="./input/test", target_size=(224, 224))
    training_data = reader('input/test')

    pass

if __name__ == "__main__":
    main()