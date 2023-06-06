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
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import log_loss
import os
from sklearn.utils import class_weight



def main():
    # print('running')
    # training_data_generator = ImageDataGenerator()
    # training_data = training_data_generator.flow_from_directory(directory="./input/train", target_size=(224, 224))
    # testing_data_generator = ImageDataGenerator()
    # testing_data = testing_data_generator.flow_from_directory(directory="./input/test", target_size=(224, 224))

    # training_data32x32 = reader(folder='input/train', file='train_32x32', pickle_name='training32x32')
    # exit(0)

    

    # init = tf.global_variables_initializer()
    # saver = tf.train.Saver()
    # run = get_AMLRun()


    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--image", required=True,
        help="path to the input image")

    ap.add_argument("-t", "--type", required=True,
        help="train/test/datagen choose one to choose what to run")

    ap.add_argument("-model", "--model", type=str, default="vgg16",
	    help="name of pre-trained network to use: choices (vgg16custom, vgg16pretrained, vgg16startingweights)")


    ap.add_argument("-d", "--development", required=True,
        help="y/n value. if in development mode only the first 5000 images will be used to speed up training/testing")

    ap.add_argument("-r", "--redodata", required=True,
        help="y/n value. if individual numbers should be generated again from the raw data and saved as new pickle")
    
    args = vars(ap.parse_args())

    if args['development'] is 'y':
        dev_mode = True
    else: 
        dev_mode = False


    run_type = args['type']

    #batch size for training
    #2000 seems to work
    # batch_size = 2000
    batch_size=200

    MODELS = {
        "vgg16": VGG16,
        "custom": VGG16Custom,
        "pretrained": VGG16Pretrained
    }

    model = args["model"]

    f= open("guru99.txt","w+")
    for i in range(10):
        f.write("This is line %d\r\n" % (i+1))
    f.close() 



    # esnure a valid model name was supplied via command line argument
    if args["model"] not in MODELS.keys():
        raise AssertionError("The --model command line argument should "
            "be a key in the `MODELS` dictionary")



    if model == "vgg16":
        running_model = "transfer"
        model_name = "vgg16Transfer.h5"
    elif model == "custom":
        running_model = "custom"
        model_name = "vgg16Custom3.h5"

    elif model == "pretrained":
        running_model = "pretrained"
        model_name = "vgg16Pretrained.h5"


    if run_type == 'datagen':
        print('redoing data....')

        #USE THIS ONE FOR FLOYDHUB
        # training_data_reader = reader(folder='/input/train', file='digitStruct.mat', pickle_name='training', development_mode=dev_mode, batch_size=batch_size)

        #USE THIS ONE LOCALLY
        training_data_reader = reader(folder='input/train', file='digitStruct.mat', pickle_name='training', development_mode=dev_mode, batch_size=batch_size)

        images_left = True
        model = None
        index = 0
        run_counter = 0


        while images_left:
            # print(index)
            training_labels, raw_training_images, images_left, final_image_index = training_data_reader.get_number_images_starting_at_index(start_index=index)
            # raw_training_images = training_data_reader.number_images
            # raw_training_images = np.array(raw_training_images)
            # training_images = preprocess_input(raw_training_images)
            # training_labels = training_data_reader.labels

            # pickle.dump(raw_training_images, open( 'pickles/training_true_data.p', "wb" ))
            # pickle.dump(training_labels, open( 'pickles/training_true_labels.p', "wb" ))

            # print('final image index:')
            # print(final_image_index)
            print(set(training_labels))
            # training_images = []
            print('preprocessing...')

            # for image in raw_training_images:

            training_images = np.array(raw_training_images, dtype=np.float)

            # print(training_images[0])

            """
            training_images = training_images / 255
            training_images -= training_images.mean()
            training_images /= training_images.std()
            """
            #DO I NEED TO NORMALIZE HERE??
            # training_images = cv2.normalize(training_images, 0, 255, cv2.NORM_MINMAX)  
            # for image in training_images[0:25]:
            #     cv2.imshow('image', image)
            #     cv2.waitKey()

            training_images = preprocess_input(training_images)
            # training_images[:] = training_images[:] / 255

            # training_images = cv2.normalize(training_images, 0, 255, cv2.NORM_MINMAX)  


            # exit(0)
            # print('done preprocessing...')
            training_images = np.array(training_images)

            # print(training_images[0])
            # exit(0)

            print('starting transfer learner training...')
            
            if running_model =="transfer":
                model = trainTransferLearner(data_reader=training_data_reader, training_labels=training_labels, training_images=training_images, model=model, num_classes=len(set(training_labels)))
            elif running_model == "custom":
                model = trainCustomLearner(data_reader=training_data_reader, training_labels=training_labels, training_images=training_images, model=model, num_classes=len(set(training_labels)))
            else:
                model = trainPretrainedLearner(data_reader=training_data_reader, training_labels=training_labels, training_images=training_images, model=model, num_classes=len(set(training_labels)))

            # index = index + len(training_labels)
            index = index + batch_size


            run_counter = run_counter + 1
            if dev_mode is True and run_counter > 10:
                break
            if final_image_index > 6000:
                break

        
        model.save(model_name)

        exit(0)
    

    







    # if run_type == 'train':
    #     print("beginning model training")
    #     training_images_picklename = os.path.join('pickles', 'training_true' + "_data.p")
    #     training_labels_picklename = os.path.join('pickles', 'training_true_labels' + ".p")

    #     training_images = pickle.load( open( training_images_picklename, "rb" ))
    #     training_labels = pickle.load(open(training_labels_picklename, "rb"))
    #     training_images = preprocess_input(training_images)

    #     #START TESTING AREA
    #     trainTransferLearner(training_labels=training_labels, training_images=training_images)
    #     # exit(0)
    #     trainPretrainedLearner(training_labels=training_labels, training_images=training_images)
    #     # exit(0)
    #     trainCustomLearner(training_labels=training_labels, training_images=training_images)

    #     exit(1)


    if run_type == 'test':
        print('testing models.....')
        print('nothing here yet....')
        exit(2)
        pass


    #END TESTING AREA

    if run_type == 'segment':
        print('segmenting image.....')
        # segment_images()
        # segment_video()


        exit(2)
        pass


    pass


def trainTransferLearner(data_reader, training_labels, training_images, model=None, num_classes=11):

    train_y = training_labels

    le = LabelEncoder()

    train_y = le.fit_transform(train_y)
    # print(train_y[0])
    # print(train_y[1])


    train_y = to_categorical(train_y)

    # print(train_y[0])
    # print(train_y[1])

    # image_temp = cv2.normalize(training_images[0], 0, 255, cv2.NORM_MINMAX)
    # cv2.imshow('test',image_temp)
    # cv2.waitKey(0)

    # image_temp = cv2.normalize(training_images[1], 0, 255, cv2.NORM_MINMAX)
    # cv2.imshow('test',image_temp)
    # cv2.waitKey(0)
    # exit(0)

    # training_images = cv2.normalize(training_images, 0, 1, cv2.NORM_MINMAX)
    train_y = np.array(train_y)
 
    X_train, X_valid, Y_train, Y_valid = train_test_split(training_images, train_y, test_size=0.01, random_state=None)

    y_integers = np.argmax(Y_train, axis=1)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
    print('class_weights')
    print(class_weights)
    class_weights = dict(enumerate(class_weights))

    # Example to fine-tune on 3000 samples from Cifar10
    # test_i = 265

    # print(Y_train[test_i])
    # print(X_train[0])

    # image_temp = cv2.normalize(X_train[test_i], 0, 255, cv2.NORM_MINMAX)

    # cv2.imshow('test', image_temp)
    # cv2.waitKey()


    # exit(0)



    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = num_classes
    batch_size = 100
    nb_epoch = 5

    # Load our model
    # model = vgg16_model(img_rows, img_cols, channel, num_classes)

    if model is None:
        print('redoing model from scratch')
        vgg16_transfer = VGG16Transfer(num_classes=num_classes)
        model = vgg16_transfer.model

        model.summary()
    # Start Fine-tuning


    # class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
    # print(class_weights)
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, shuffle=True, verbose=1, validation_data=(X_valid, Y_valid), class_weight=class_weights)

    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    print(predictions_valid[0])
    print(Y_valid[0])
    print('**********')
    print(predictions_valid[1])
    print(Y_valid[1])
    # cv2.imshow('test', X_valid[0])
    # cv2.waitKey()
    # Cross-entropy loss score
    score = log_loss(Y_valid, predictions_valid)

    print(score)
    return model

def trainCustomLearner(data_reader, training_labels, training_images, model=None, num_classes=11):

    train_y = training_labels

    le = LabelEncoder()

    train_y = le.fit_transform(train_y)

    train_y = to_categorical(train_y)

    train_y = np.array(train_y)
 
    X_train, X_valid, Y_train, Y_valid = train_test_split(training_images, train_y, test_size=0.01)


    y_integers = np.argmax(Y_train, axis=1)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
    print('class_weights')
    print(class_weights)
    class_weights = dict(enumerate(class_weights))

    # Example to fine-tune on 3000 samples from Cifar10
    # test_i = 265

    # print(Y_train[test_i])
    # print(X_train[0])

    # image_temp = cv2.normalize(X_train[test_i], 0, 255, cv2.NORM_MINMAX)

    # cv2.imshow('test', image_temp)
    # cv2.waitKey()


    # exit(0)

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = num_classes
    batch_size = 100
    nb_epoch = 8

    # Load our model
    # model = vgg16_model(img_rows, img_cols, channel, num_classes)

    if model is None:
        print('redoing model from scratch')
        vgg16_custom = VGG16Custom(num_classes=num_classes)
        model = vgg16_custom.model

        model.summary()
    # Start Fine-tuning


    # class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
    # print(class_weights)
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, shuffle=True, verbose=1, validation_data=(X_valid, Y_valid), class_weight=class_weights)

    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    print(predictions_valid[0])
    print(Y_valid[0])
    print('**********')
    print(predictions_valid[1])
    print(Y_valid[1])
    # cv2.imshow('test', X_valid[0])
    # cv2.waitKey()
    # Cross-entropy loss score
    score = log_loss(Y_valid, predictions_valid)

    print(score)
    return model

def trainPretrainedLearner(data_reader, training_labels, training_images, model=None, num_classes=11):

    train_y = training_labels

    le = LabelEncoder()

    train_y = le.fit_transform(train_y)

    train_y = to_categorical(train_y)



    

    train_y = np.array(train_y)
 
    X_train, X_valid, Y_train, Y_valid = train_test_split(training_images, train_y, test_size=0.01)


    y_integers = np.argmax(Y_train, axis=1)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
    print('class_weights')
    print(class_weights)
    class_weights = dict(enumerate(class_weights))

    # Example to fine-tune on 3000 samples from Cifar10
    # test_i = 265

    # print(Y_train[test_i])
    # print(X_train[0])

    # image_temp = cv2.normalize(X_train[test_i], 0, 255, cv2.NORM_MINMAX)

    # cv2.imshow('test', image_temp)
    # cv2.waitKey()


    # exit(0)

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = num_classes
    batch_size = 100
    nb_epoch = 5

    # Load our model
    # model = vgg16_model(img_rows, img_cols, channel, num_classes)

    if model is None:
        print('redoing model from scratch')
        vgg16_pretrained = VGG16Pretrained(num_classes=num_classes)
        model = vgg16_pretrained.model

        model.summary()
    # Start Fine-tuning


    

    # exit(0)
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, 
        shuffle=True, verbose=1, validation_data=(X_valid, Y_valid), class_weight=class_weights)

    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    print(predictions_valid[0])
    print(Y_valid[0])
    print('**********')
    print(predictions_valid[1])
    print(Y_valid[1])
    # cv2.imshow('test', X_valid[0])
    # cv2.waitKey()
    # Cross-entropy loss score
    score = log_loss(Y_valid, predictions_valid)

    print(score)
    return model

if __name__ == "__main__":
    main()