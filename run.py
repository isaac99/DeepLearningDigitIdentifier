import numpy as np
from cv2 import cv2
import tensorflow as tf
from tensorflow import keras
# from keras_preprocessing.image import ImageDataGenerator
# from keras_applications import imagenet_utils
# from keras_applications.inception_v3 import preprocess_input
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import load_img

# from keras_preprocessing.image import img_to_array
# from keras_preprocessing.image import load_img
import argparse
from VGG16CV2019 import VGG16Custom, VGG16Transfer, VGG16Pretrained, segment_images
# import keras_applications.vgg16 as VGG16
# from keras_applications.vgg16 import VGG16
# from keras_applications.vgg16 import VGG16
# from keras_applications import vgg16

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

from DataReader import DataReader as reader
from DataReader import resize_if_necessary
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import log_loss
import os
from sklearn.utils import class_weight
from tensorflow.keras.models import load_model


def main():
    model = load_model('./models/vgg16Pretrained.h5')
    
    
    image1 = cv2.imread('./input/graded_input/1_graded.png')

    image1, regions1 = segment_images(image1)
    
    # print(regions1)
    # cv2.imshow('1', image1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    draw_numbers_for_regions(regions1, model, image1, 1)
    
    
    image2 = cv2.imread('./input/graded_input/2_graded.png')
    image2, regions2 = segment_images(image2)
    

    # cv2.imshow('2', image2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    draw_numbers_for_regions(regions2, model, image2, 2)

    
    image3 = cv2.imread('./input/graded_input/3_graded.png')
    image3, regions3 = segment_images(image3)
    # cv2.imshow('3', image3)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    draw_numbers_for_regions(regions3, model, image3, 3)

    
    image4 = cv2.imread('./input/graded_input/4_graded.png')
    image4, regions4 = segment_images(image4)
    # cv2.imshow('4', image4)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    draw_numbers_for_regions(regions4, model, image4, 4)



    image5 = cv2.imread('./input/graded_input/5_graded.png')
    image5, regions5 = segment_images(image5)
    # cv2.imshow('5', image5)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    draw_numbers_for_regions(regions5, model, image5, 5)
    
    
    # segment_video(model)
    
    pass


def draw_numbers_for_regions(regions, model, image, number):


    # print(regions)

    image_name = 'graded_images/'+ str(number) + '.png'

    for index, p in enumerate(regions):
        black_canvas = np.zeros((224, 224, 3), dtype=np.uint8)

        xmax, ymax = np.amax(p, axis=0)
        xmin, ymin = np.amin(p, axis=0)

        width = xmax - xmin
        height = ymax - ymin

        center_x = width // 2
        center_y = height // 2


        image_cutout = image[ymin:ymax, xmin:xmax]

        image_cutout = resize_if_necessary(image_cutout)
        black_canvas[0:image_cutout.shape[0], 0:image_cutout.shape[1]] = image_cutout


        black_canvas = black_canvas / 255
        black_canvas -= black_canvas.mean()
        black_canvas /= black_canvas.std()

        digit_image = preprocess_input(black_canvas)
        # temp_digit_image = cv2.normalize(digit_image, 0, 255, cv2.NORM_MINMAX)

        
        prediction_input = np.array([digit_image])

        prediction = model.predict(prediction_input)
        print(prediction)
        # print(np.argmax(prediction))
        p_index = np.argmax(prediction)

        if p_index == 10:
            prediction_final = None
        # elif p_index == 11:
        #     prediction_final = None
        else:
            prediction_final = p_index

        if prediction_final is not None:
            cv2.rectangle(image, (xmin,ymax), (xmax,ymin), (0, 255, 0), 1)
            cv2.putText(image,str(p_index), (xmin+center_x, ymax+ 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)

    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit(0)

    cv2.imwrite(image_name, image )

def draw_numbers_for_regions_video(regions, model, image, number):
    # print(regions)

    # image_name = 'graded_images/'+ str(number) + '.png'

    for index, p in enumerate(regions):
        black_canvas = np.zeros((224, 224, 3), dtype=np.uint8)

        xmax, ymax = np.amax(p, axis=0)
        xmin, ymin = np.amin(p, axis=0)

        width = xmax - xmin
        height = ymax - ymin

        center_x = width // 2
        center_y = height // 2

        """
        image_cutout = image[ymin:ymax, xmin:xmax]

        image_cutout = resize_if_necessary(image_cutout)
        black_canvas[0:image_cutout.shape[0], 0:image_cutout.shape[1]] = image_cutout


        black_canvas = black_canvas / 255
        black_canvas -= black_canvas.mean()
        black_canvas /= black_canvas.std()

        digit_image = preprocess_input(black_canvas)
        # temp_digit_image = cv2.normalize(digit_image, 0, 255, cv2.NORM_MINMAX)

        
        prediction_input = np.array([digit_image])

        prediction = model.predict(prediction_input)
        # print(prediction)
        # print(np.argmax(prediction))
        p_index = np.argmax(prediction)

        if p_index == 10:
            prediction_final = None
        # elif p_index == 11:
        #     prediction_final = None
        else:
            prediction_final = p_index

        if prediction_final is not None:
            cv2.rectangle(image, (xmin,ymax), (xmax,ymin), (0, 255, 0), 1)
            cv2.putText(image,str(p_index), (xmin+center_x, ymax+ 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        """
        cv2.rectangle(image, (xmin,ymax), (xmax,ymin), (0, 255, 0), 1)
        cv2.putText(image,str(2), (xmin+center_x, ymax+ 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit(0)

    # cv2.imwrite(image_name, image )

    return image

def segment_video(model):

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('graded_images/output_video.mp4', fourcc, 20.0, (224,224))
    video = cv2.VideoCapture("input/video/street_video.mp4")

    count = 0
    while video.isOpened():
        ret,frame = video.read()
        image, regions = segment_images(frame)

        frame = draw_numbers_for_regions_video(regions, model, image, 2)
        # out.write(frame)


        cv2.imshow('window-name', image)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
        # cv2.imwrite("frame%d.jpg" % count, frame)
        # key = cv2.waitKey(0)
        # while key not in [ord('q'), ord('k')]:
        #     key = cv2.waitKey(0)
        # # Quit when 'q' is pressed
        # if key == ord('q'):
        #     break


    out.release()
    video.release()
    cv2.destroyAllWindows()  # destroy all the opened windows

    pass

if __name__ == "__main__":
    main()