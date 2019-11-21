import numpy as np
from cv2 import cv2
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator
import argparse
from VGG16 import VGG16
from DataReader import DataReader as reader
import pickle

def main():
    # print('running')
    # training_data_generator = ImageDataGenerator()
    # training_data = training_data_generator.flow_from_directory(directory="./input/train", target_size=(224, 224))
    # testing_data_generator = ImageDataGenerator()
    # testing_data = testing_data_generator.flow_from_directory(directory="./input/test", target_size=(224, 224))

    # training_data32x32 = reader(folder='input/train', file='train_32x32', pickle_name='training32x32')
    # exit(0)

    training_data_reader = reader(folder='input/train', file='digitStruct.mat', pickle_name='training', development_mode=True)
    images = training_data_reader.number_images
    labels = training_data_reader.labels
    # print(labels[0])
    # cv2.imshow('test', images[0])
    # cv2.waitKey()
    # exit(0)



    # print(training_data.number_images[0][0])
    # cv2.imshow('test', training_data.number_images[0][1])
    # cv2.waitKey()


    # testing_data = reader('input/test')

    # print(training_data.data[:20, :])

    pass

if __name__ == "__main__":
    main()