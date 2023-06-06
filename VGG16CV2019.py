import numpy as np
from cv2 import cv2
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Reshape, Activation, Conv2D, MaxPool2D
from sklearn.utils import class_weight
import networkx as nx
import math
# from skimage.feature import hog
# from sklearn.externals import joblib


class VGG16Custom():
    def __init__(self, num_classes=11):
        self.model = self.create_model(num_classes)
        pass

    def fit(self):
        pass

    def test(self):
        pass

    def create_model(self, num_classes):
        model = Sequential()
        model.add(Conv2D(input_shape=(224,224,3), filters=64, kernel_size=(3,3), padding="same", activation="relu"))

        model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=num_classes, activation="softmax"))

        # adam = Adam(lr=0.001)
        sgd = SGD(lr=1e-2, decay=1e-7, momentum=0.9, nesterov=True)

        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        return model


class VGG16Pretrained():
    def __init__(self, num_classes=11):
        self.model = self.create_model(num_classes)
        pass

    def train(self):
        pass

    def test(self):
        pass

    def create_model(self, num_classes):
        vgg_model = VGG16(weights='imagenet', include_top=True)

        model = Sequential()

        model.add(vgg_model)
        model.add(Dense(num_classes, activation='softmax'))
        model.layers[0].trainable = False

        #To set the first 8 layers to non-trainable (weights will not be updated)

        # for layer in model.layers[:8]:
        #     layer.trainable = False

        # Learning rate is changed to 0.001
        sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        # adam = Adam(lr=0.001)

        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        return model


class VGG16Transfer():
    def __init__(self, num_classes=11):
        self.model = self.create_model(num_classes)
        pass

    def train(self):
        pass

    def test(self):
        pass

    def create_model(self, num_classes):
        vgg_model = VGG16(weights='imagenet', include_top=True)
        # vgg_model.layers.pop()

        for index in range(len(vgg_model.layers)):
            if index is 3:
                print('only training until layer:')
                print(index)
                break
            vgg_model.layers[index].trainable = False
            


        model = Sequential()

        model.add(vgg_model)
        model.add(Dense(num_classes, activation='softmax'))
        # model.layers[0].trainable = False

        #To set the first 8 layers to non-trainable (weights will not be updated)

        # for layer in model.layers[:8]:
        #     layer.trainable = False

        # Learning rate is changed to 0.001
        sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        return model




def segment_images(images=None):
    # takes in numpy array as input
    # returns same numpy array of images but with each segmented to find numbers
    # image = images[0]
    # image = cv2.imread('input/test/9.png')
    image=images
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gray = gray / 255
    gray -= gray.mean()
    gray /= gray.std()
    gray = cv2.normalize(gray, None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)

    #UNCOMMENT BELOW
    # cv2.imshow('test segmentation 0', image)
    # cv2.waitKey()


    # cv2.imshow('test segmentation 0', gray)
    # cv2.waitKey()


    #UNCOMMENT ABOVE

    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # mser = cv2.MSER_create()

    # vis = image.copy()

    # regions = mser.detectRegions(gray)
    # hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
    # cv2.polylines(image, hulls, 1, (0,255,0)) 

    regions, boxes = mser(gray)
    # print(regions[0])

    #filter based on aspect ratio
    # width =
    # height =    
    print('not filtered regions')    
    print(len(regions))
    
    rectangle_picks = []
    for index, p in enumerate(regions):
        p = regions[index]
        xmax, ymax = np.amax(p, axis=0)
        xmin, ymin = np.amin(p, axis=0)

        width = xmax - xmin
        height = ymax - ymin

        aspect_ratio = height / width

        center_x = width // 2
        center_y = height // 2

        # intersection_area = (xmax - xmin + 1) * (ymin - ymax + 1)


        if aspect_ratio > 1.4 and aspect_ratio < 4 and height < 120:
            rectangle_picks.append(index)
            # cv2.rectangle(image, (xmin,ymax), (xmax,ymin), (0, 255, 0), 1)
            # cv2.rectangle(image, (xmin + center_x, ymax - center_y), (xmax - center_x, ymin + center_y), (0, 0, 255), 1)


    




    chosen_regions = np.array(regions)
    chosen_boxes = np.array(boxes)

    chosen_regions = chosen_regions[rectangle_picks]
    chosen_boxes = chosen_boxes[rectangle_picks]
    # print('filter 1')
    # print(len(chosen_regions))
    # print('chosen regions')
    # print(chosen_regions)
    final_regions = []
    for index, coord in enumerate(chosen_regions):
        bbox = cv2.boundingRect(coord)
        x,y,w,h = bbox
        # print('box')
        # print(bbox)
        # print('coords')
        index_to_append = None


        center_coords = (x + w //2, y + h //2)
        # print(center_coords)
        added_box = False

        # if w< 10 or h < 10 or w/h > 5 or h/w > 5:
        #     continue

        for index2, coord2 in enumerate(chosen_regions):
            if index2 == index:
                continue

            bbox2 = cv2.boundingRect(coord2)
            x2,y2,w2,h2 = bbox2

            center_coords2 = (x2 + w2 // 2, y2 + h2 //2)

            # print(center_coords2)

            distance = calculate_distance(center_coords, center_coords2)
            # print(distance)

            if distance < 10:
                if w2*h2 > w * h:
                    # final_regions.append(index2)
                    # index_to_append = index2
                    added_box = True

                # else:
                    # final_regions.append(index)
                    # index_to_append = index
                    # added_box = False

            else:
                continue
                # final_regions.append(index)
        if added_box is False:
            final_regions.append(index)
        if added_box is True:
            # final_regions.append(index_to_append)
            pass


            

        # final_regions.append(index)
    final_regions = np.array(final_regions, dtype=np.uint8)
    final_regions = np.unique(final_regions)
    # print(final_regions)
    chosen_regions = chosen_regions[final_regions]

    print(len(chosen_regions))


    #     p = chosen_regions[index]
        
    #     xmax, ymax = np.amax(p, axis=0)
    #     xmin, ymin = np.amin(p, axis=0)

    #     width = xmax - xmin
    #     height = ymax - ymin

    #     center_x = width // 2
    #     center_y = height // 2




    #     cv2.rectangle(image, (xmin,ymax), (xmax,ymin), (0, 255, 0), 1)
    #     cv2.rectangle(image, (xmin + center_x, ymax - center_y), (xmax - center_x, ymin + center_y), (0, 0, 255), 1)


        # other_regions = chosen_regions
        # for other_region in chosen_regions[range(len(chosen_regions))]
    # cv2.rectangle(image, (xmin,ymax), (xmax,ymin), (0, 255, 0), 1)


    # cv2.imshow('test segmentation 1', image)
    # cv2.waitKey()

    return image, chosen_regions

    # #remove noise
    # kernel = np.ones((3,3), np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # #background area
    # s_bg = cv2.dilate(opening, kernel, iterations=3)

    # #find foreground
    # dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # ret, sure_fg = cv2.threshold(dist_transform, 0.7* dist_transform.max(), 255, 0)

    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(s_bg, sure_fg)

    # cv2.imshow('test segmentation 0.5', unknown)
    # cv2.waitKey()

    # ret, markers = cv2.connectedComponents(sure_fg)

    # markers = markers + 1

    # markers[unknown==255] = 0

    # markers = cv2.watershed(image, markers)
    # image[markers == -1] = [255, 0, 0]

    # # gray = cv2.watershed()

    # cv2.imshow('test segmentation', sure_fg)
    # cv2.waitKey()

    # cv2.imshow('test segmentation 2', unknown)
    # cv2.waitKey()

    # cv2.imshow('test segmentation 3', image)
    # cv2.waitKey()

def mser(cv_image):
    vis = cv_image.copy()
    mser = cv2.MSER_create(_delta=3, _min_area=50, _max_area=4000, _max_variation=0.06)
    # mser = cv2.MSER_create()
    regions, other = mser.detectRegions(cv_image)
    # print(other)
    

    # hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    # cv2.polylines(vis, hulls, 1, (0, 255, 0))

    # cv2.imshow('img', vis)
    # mask = np.zeros((vis.shape[0], vis.shape[1], 1), dtype=np.uint8)
    # for contour in hulls:
    #     cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    # text_only = cv2.bitwise_and(vis, vis, mask=mask)

    

    
    # cv2.imshow('img', vis)

    return regions, other


def combine_boxes(boxes):
    new_array = []
    for boxa, boxb in zip(boxes, boxes[1:]):
        if intersection(boxa, boxb):
            new_array.append(union(boxa, boxb))
        else:
            new_array.append(boxa)
    return np.array(new_array).astype('int')

def union(a,b):
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  w = max(a[0]+a[2], b[0]+b[2]) - x
  h = max(a[1]+a[3], b[1]+b[3]) - y
  return (x, y, w, h)

def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return () # or (0,0,0,0) ?
  return (x, y, w, h)


def overlapping_area(detection_1, detection_2):
    
    x1_tl = detection_1[0]
    x2_tl = detection_2[0]
    x1_br = detection_1[0] + detection_1[3]
    x2_br = detection_2[0] + detection_2[3]
    y1_tl = detection_1[1]
    y2_tl = detection_2[1]
    y1_br = detection_1[1] + detection_1[4]
    y2_br = detection_2[1] + detection_2[4]
    # Calculate the overlapping Area
    x_overlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    area_1 = detection_1[3] * detection_2[4]
    area_2 = detection_2[3] * detection_2[4]
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)

def calculate_distance(point1, point2):
    p1 = point1
    p2 = point2

    # print(p1[0])
    # print(p1[1])
    # print(p2[0])
    # print(p2[1])


    distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )

    return distance

def main():
    print('main')

if __name__ == "__main__":
    main()