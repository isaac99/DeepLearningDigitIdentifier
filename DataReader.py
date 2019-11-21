from __future__ import division
import numpy as np
import h5py
import tqdm
import os
from cv2 import cv2
import pickle

class DataReader():
    def __init__(self, folder="./input/train", file='digitStruct', pickle_name=None, pickle_exists=False, development_mode=False):
        self.folder = folder
        self.pickle_name = pickle_name
        self.file = file
        self.development_mode = development_mode

        data_filename = os.path.join('pickles', pickle_name + "_data.p")
        image_names_filename = os.path.join('pickles', pickle_name + "_image_names.p")        

        

        if not os.path.isfile(data_filename) or not os.path.isfile(image_names_filename):
            print('Existing pickle not found!')
            self.do_data_setup(data_filename, image_names_filename)
        else:
            print('Running from existing pickle!')
            loaded_pickle_data = pickle.load( open( data_filename, "rb" ))
            loaded_pickle_image_names = pickle.load(open(image_names_filename, "rb"))
            self.data = loaded_pickle_data
            self.original_image_names = loaded_pickle_image_names
        
    
        self.labels, self.number_images = self.get_individual_number_images()
    

    def do_data_setup(self, data_filename, image_names_filename):

        mat_data = h5py.File(os.path.join(self.folder, self.file + '.mat'))
        print(mat_data)
        size = mat_data['/' + self.file + '/name'].size
        data = []
        images = []

        for _i in tqdm.tqdm(range(size)):
            pic = self.get_name(_i, mat_data)
            box = self.get_box_data(_i, mat_data)

            # print(pic)
            # print(box)

            for index, number in enumerate(box['label']):
                height = int(box['height'][index])
                width = int(box['width'][index])
                left = int(box['left'][index])
                top = int(box['top'][index])

                # print(index)
                # print(number)
                # print(height)
                # print(width)
                # print(left)
                # print(top)

                # data = np.append(data, [number, height, width, left, top], axis=0)
                data.append((number, height, width, top, left, pic))
                images.append(pic)
            
        self.data = np.array(data)
        self.original_image_names = np.array(images)

        pickle.dump(self.data, open( data_filename, "wb" ))
        pickle.dump(self.original_image_names, open( image_names_filename, "wb" ))



    def get_box_data(self, index, hdf5_data):
        """
        get `left, top, width, height` of each picture
        :param index:
        :param hdf5_data:
        :return:
        """
        meta_data = dict()
        meta_data['height'] = []
        meta_data['label'] = []
        meta_data['left'] = []
        meta_data['top'] = []
        meta_data['width'] = []

        def print_attrs(name, obj):
            vals = []
            if obj.shape[0] == 1:
                vals.append(obj[0][0])
            else:
                for k in range(obj.shape[0]):
                    vals.append(int(hdf5_data[obj[k][0]][0][0]))
            meta_data[name] = vals

        box = hdf5_data['/digitStruct/bbox'][index]
        hdf5_data[box[0]].visititems(print_attrs)
        return meta_data

    def get_name(self, index, hdf5_data):
        name = hdf5_data['/digitStruct/name']
        return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])

    def get_individual_number_images(self):
        image_labels = []
        number_image_cutouts = []

        for index, image_name in enumerate(self.original_image_names):
            #if in development mode only return the first 5 images so testing model is faster
            if self.development_mode is True and index > 300:
                break

            print('finished image: ' + image_name)
            path = os.path.join(self.folder, image_name)

            #read image in grayscale
            image = cv2.imread(path, 0)
            digits_in_image = self.data[self.data[:, 5] == image_name]

            for i, digit in enumerate(digits_in_image):
                label = int(float(digit[0]))
                height = int(float(digit[1]))
                width = int(float(digit[2]))
                top = int(float(digit[3]))
                left = int(float(digit[4]))

                digit_image = image[top: top+height, left:left+width]

                digit_image = resize_if_necessary(digit_image)

                black_canvas = np.zeros((224, 224), dtype=np.uint8)
                black_canvas[0:digit_image.shape[0], 0:digit_image.shape[1]] = digit_image

                final_digit_image = black_canvas

                number_image_cutouts.append(final_digit_image)
                image_labels.append(label)

        return np.array(image_labels), np.array(number_image_cutouts)



# def resize_height(digit_image):
#     aspectRatio = digit_image.shape[1] / digit_image.shape[0]


#     return digit_image


# def resize_width(digit_image):

#     return digit_image

# def resize_image()

def resize_if_necessary(image):
    height = image.shape[0]
    width = image.shape[1]

    # aspect_ratio = height / width


    # print(image.shape)

    # print(height)
    # print(width)

    if width > 224 or height > 224:
        if width >= height:
            aspect_ratio = height / width
            # print(aspect_ratio)
            new_width = 224
            new_height = aspect_ratio * new_width

            new_size = (int(new_width), int(new_height))
            # print(new_size)


            resized_image = cv2.resize(image, new_size, interpolation = cv2.INTER_AREA)
            return resized_image
            # width_ratio = 224/width


        elif height > width:
            aspect_ratio = width / height
            new_height= 224
            new_width = aspect_ratio * new_height



            new_size = (int(new_width), int(new_height))


            # print(new_size)


            resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            return resized_image

    else:
        return image