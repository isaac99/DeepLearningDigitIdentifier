import numpy as np
import h5py
import tqdm
import os

class DataReader():
    def __init__(self, folder="./input/test"):
        mat_data = h5py.File(os.path.join(folder, 'digitStruct.mat'))
        print(mat_data)
        size = mat_data['/digitStruct/name'].size

        for _i in tqdm.tqdm(range(size)):
            pic = self.get_name(_i, mat_data)
            box = self.get_box_data(_i, mat_data)

            print(pic)
            print(box)

    
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