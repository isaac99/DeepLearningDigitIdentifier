from __future__ import division
import numpy as np
import h5py
# from tqdm import tqdm
import os
from cv2 import cv2
import pickle
from tensorflow.keras.utils import Sequence

class DataReader():
    def __init__(self, folder="/input/train", file='digitStruct', pickle_name=None, pickle_exists=False, development_mode=False, batch_size=100):
        self.folder = folder
        self.pickle_name = pickle_name
        self.file = file
        self.development_mode = development_mode
        self.batch_size = batch_size

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


        self.number_of_images = len(self.original_image_names)

        
        # print('getting individual number images')
        # self.labels, self.number_images, new_start_index = self.get_individual_number_images()

        # print('done getting number images')
        # self.labels = np.array(self.labels)
        # self.number_images = np.fromiter(iter(self.number_images), count=len(self.number_images), dtype=np.uint8)
        # self.number_images = np.asarray(self.number_images, dtype=np.uint8)
    
    def get_number_images_starting_at_index(self, start_index):
        print('getting individual number images')
        labels, number_images, should_continue_getting_images, final_image_name = self.get_individual_number_images(start_index)

        image_number = int(final_image_name.split('.')[0])

        return labels, number_images, should_continue_getting_images, image_number

    def do_data_setup(self, data_filename, image_names_filename):

        # print(self.folder)
        # print(self.file)
        mat_data = h5py.File(os.path.join(self.folder, self.file))
        # print(mat_data)
        # print(self.file)
        filename = '/' + self.file.split('.')[0] + '/name'
        # print(filename)
        size = mat_data[filename].size
        data = []
        images = []

        # for _i in tqdm.tqdm(range(size)):
        for _i in range(size):
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

    def get_individual_number_images(self, start_index=0):
        image_labels = []
        number_image_cutouts = []
        images_to_use = self.original_image_names[start_index:start_index+self.batch_size]
        print(len(images_to_use))
        should_continue_getting_images = True
        final_image = None
        image_name = None

        if len(images_to_use) < self.batch_size:
            should_continue_getting_images = False

        for index, image_name in enumerate(images_to_use):
            #if in development mode only return the first 5 images so testing model is faster
            # if self.development_mode is True and index > 2000:
            #     break

            # print(image_name)
            final_image = image_name

            print('finished image: ' + str(image_name))
            path = os.path.join(self.folder, image_name)

            #read image in grayscale
            # image = cv2.imread(path, 0)

            #read in images in 3 channels
            image = cv2.imread(path)

            digits_in_image = self.data[self.data[:, 5] == image_name]

            for i, digit in enumerate(digits_in_image):
                label = int(float(digit[0]))
                height = int(float(digit[1]))
                width = int(float(digit[2]))
                top = int(float(digit[3]))
                left = int(float(digit[4]))

                # one_hot = [0,0,0,0,0,0,0,0,0,0,0]
                

                # print(digit)
                # if i is 300:
                #     exit(0)

                digit_image = image[top: top+height, left:left+width]


                

                digit_image = resize_if_necessary(digit_image)


                temp_height = digit_image.shape[0]
                temp_width = digit_image.shape[1]
                print(temp_height)
                print(temp_width)

                if(temp_height > temp_width and temp_height != 0 and temp_width != 0):
                    temp_aspect_ratio = temp_width / temp_height
                    new_height= 224
                    new_width = temp_aspect_ratio * new_height


                    new_size = (int(new_width), int(new_height))

                    digit_image = cv2.resize(digit_image, new_size, interpolation=cv2.INTER_AREA)

                digit_image = digit_image.astype(np.uint8)

                # cv2.imshow('test', digit_image)
                # cv2.waitKey()

                black_canvas = np.zeros((224, 224, 3), dtype=np.uint8)
                black_canvas[0:digit_image.shape[0], 0:digit_image.shape[1]] = digit_image

                final_digit_image = black_canvas

                number_image_cutouts.append(final_digit_image)
                image_labels.append(label)

                if label == 0 or label == 1:
                    false_image_label, false_image_black_bg = get_false_image_if_necessary(list_size=len(digits_in_image), index=i, image=image,
                        starting_top=top+height, starting_left=left+height)
                    if(false_image_black_bg is not None):
                        print('adding in false image')

                        # cv2.imshow('test', false_image_black_bg)
                        # cv2.waitKey()
                        # exit(0)
                        number_image_cutouts.append(false_image_black_bg)
                        image_labels.append(false_image_label)


        # number_image_cutouts = np.asarray(number_image_cutouts, dtype=np.uint8)
        # result = np.array((224, 224, 3), dtype=np.uint8)
        # for element in number_image_cutouts:
        #     # np.append(result, element, axis=1)
        #     # np.dstack((result, element)).shape
        #     print(element.shape)
        #     np.concatenate((result, element), axis=0)
        # print('done getting images returning labels and images...')
        return np.array(image_labels), number_image_cutouts, should_continue_getting_images, image_name



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


def get_false_image_if_necessary(list_size, index, image, starting_top, starting_left):
    if index == list_size - 1:
        black_canvas_new = np.zeros((224, 224, 3), dtype=np.uint8)

        false_image_cutout = image[starting_top:, starting_left:]

        if(false_image_cutout.shape[0] == 0 or false_image_cutout.shape[1] == 0):
            #if no size return None, None
            return None, None

        false_image_cutout = resize_if_necessary(false_image_cutout)
        # cv2.imshow('test', false_image_cutout)
        # cv2.waitKey()
        # exit(0)
        black_canvas_new[0:false_image_cutout.shape[0], 0:false_image_cutout.shape[1]] = false_image_cutout
        return 11, black_canvas_new

    return None, None