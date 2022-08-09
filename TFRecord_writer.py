from asyncore import read
import tensorflow as tf 
from glob import glob
import numpy as np
from os.path import isfile, join
import os
from check_if_ds_files_in_good_format import check_if_files_in_good_format
import tqdm
import shutil
from random import shuffle
from math import floor


def create_label_files(image_dir, label_dir):
    _, filenames = check_if_files_in_good_format(image_dir)
    for filename in filenames:
        label_filename = filename[:-3]+'txt'
        label_path = join(label_dir, label_filename)
        if 'nonaesthetic' in label_path:
            label = 0
        else:
            label = 1
        with open(label_path, 'w') as f:
            f.write(str(label))


def copy_files(curr_dir, dest_dir):
    _, filenames = check_if_files_in_good_format(curr_dir)
    for filename in filenames:
        curr_filename = join(curr_dir, filename)
        dest_filename = join(dest_dir, filename)
        shutil.copyfile(curr_filename, dest_filename)
    

def shuffle_files(curr_dir, train_dir, val_dir, train_coef=0.8):
    file_num, _ = check_if_files_in_good_format(curr_dir)
    filenames = []
    train_ds_size = floor(train_coef*file_num)
    val_ds_size = file_num-train_ds_size
    if os.path.isdir(curr_dir):
        for filename in os.listdir(curr_dir):
            # print(filename)
            filenames.append(filename)
        shuffled_filenames = filenames
        shuffle(shuffled_filenames)
        it = 0
        for filename in shuffled_filenames:
            if it < train_ds_size:
                curr_filename = join(curr_dir, filename)
                dest_filename = join(train_dir, filename)
            else:
                curr_filename = join(curr_dir, filename)
                dest_filename = join(val_dir, filename)
            shutil.copyfile(curr_filename, dest_filename)
            it += 1
    
def associate_image_with_label(img_dir, label_dir):
    img_label_dict = {}
    if os.path.isdir(img_dir) and os.path.isdir(label_dir):
        for filename in os.listdir(img_dir):
            label_filename = filename[:-3]+'txt'
            label_file_path = os.path.join(label_dir, label_filename)
            with open(label_file_path) as f:
                label = str(f.readline())
                img_label_dict[filename] = label
    return img_label_dict

def read_img_dir(path):
    num_files, filenames = check_if_files_in_good_format(path)
    print(num_files)
    # labels = np.zeros(num_files)
    # return labels, filenames


def _int64_feature(value):
    ''' Returns an int64_list from a bool / enum / int / uint '''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    '''Returns a bytes_list from a string / byte'''
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    '''Returns a float_list from a float / double'''
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def write_tfrec(tfrec_filename, image_dir, label_dir):
    img_label_dict = associate_image_with_label(image_dir, label_dir)
    with tf.io.TFRecordWriter(tfrec_filename) as writer:
        img_list = list(img_label_dict.keys())
        label_list = list(img_label_dict.values())
        for i in range(len(img_list)):
            filePath = join(image_dir, img_list[i])
            # read the JPEG source file into a tf string
            image = tf.io.read_file(filePath)
            # get the shape of the image from the JPEG file header
            image_shape = tf.io.extract_jpeg_shape(image, output_type=tf.dtypes.int32, name=None)

            # features dictionary
            feature_dict = {
            'label' : _int64_feature(int(label_list[i])),
            'height': _int64_feature(image_shape[0]),
            'width' : _int64_feature(image_shape[1]),
            'chans' : _int64_feature(image_shape[2]),
            'image' : _bytes_feature(image)
            }

            # Create Features object
            features = tf.train.Features(feature = feature_dict)

            # create Example object
            tf_example = tf.train.Example(features=features)

            # serialize Example object into TfRecord file
            writer.write(tf_example.SerializeToString())
    return



def make_tfrec(image_dir, img_shard, tfrec_base, label_file, tfrec_dir, num_images):

    # make destination directory
    os.makedirs(tfrec_dir, exist_ok=True)
    print('Directory',tfrec_dir,'created',flush=True)

    # make lists of images and their labels
    all_labels, all_images = _create_images_labels(label_file)
    print('Found',len(all_labels),'images and labels in',label_file)

    if (num_images != 0 and num_images < len(all_images)):
        all_images = all_images[:num_images]
        all_labels = all_labels[:num_images]
        print('Using',num_images,'images..')
    else:
        print('Using',len(all_labels),'images..')

    # calculate how many shards we will generate and number of images in last shard
    last_shard, num_shards = _calc_num_shards(all_images, img_shard)
    print (num_shards,'TFRecord files will be created.')
    if (last_shard>0):
        print ('Last TFRecord file will have',last_shard,'images.')

    # create TFRecord files (shards)
    start = 0

    for i in tqdm(range(num_shards)):

        tfrec_filename = tfrec_base+'_'+str(i)+'.tfrecord'
        write_path = os.path.join(tfrec_dir, tfrec_filename)

        if (i == num_shards-1):
            write_tfrec(write_path, image_dir, all_images[start:], all_labels[start:])
        else:
            end = start + img_shard
            write_tfrec(write_path, image_dir, all_images[start:end], all_labels[start:end])
            start = end
    return


aesthetic_label_path = '/home/bdroix/bdroix/aesthetics_analysis/dane do analizy/dane_scaled_300_432_label/aesthetic/labels/'
aesthetic_img_path = '/home/bdroix/bdroix/aesthetics_analysis/dane do analizy/dane_scaled_300_432_label/aesthetic/images/'
aesthetic_tf_path = '/home/bdroix/bdroix/aesthetics_analysis/dane do analizy/dane_scaled_300_432_label/aesthetic/tfrecords/'
nonaesthetic_label_path = '/home/bdroix/bdroix/aesthetics_analysis/dane do analizy/dane_scaled_300_432_label/nonaesthetic/labels/'
nonaesthetic_img_path = '/home/bdroix/bdroix/aesthetics_analysis/dane do analizy/dane_scaled_300_432_label/nonaesthetic/images/'
nonaesthetic_tf_path = '/home/bdroix/bdroix/aesthetics_analysis/dane do analizy/dane_scaled_300_432_label/nonesthetic/tfrecords/'
label_path = '/home/bdroix/bdroix/aesthetics_analysis/dane do analizy/dane_scaled_300_432_label/labels'
img_path = '/home/bdroix/bdroix/aesthetics_analysis/dane do analizy/dane_scaled_300_432_label/images'
tf_train_path = '/home/bdroix/bdroix/aesthetics_analysis/dane do analizy/dane_scaled_300_432_label/tfrecords/train_ds.tfrecord'
tf_val_path = '/home/bdroix/bdroix/aesthetics_analysis/dane do analizy/dane_scaled_300_432_label/tfrecords/val_ds.tfrecord'
train_img_path = '/home/bdroix/bdroix/aesthetics_analysis/dane do analizy/dane_scaled_300_432_label/train_images'
val_img_path = '/home/bdroix/bdroix/aesthetics_analysis/dane do analizy/dane_scaled_300_432_label/val_images'

if __name__ == '__main__':
    write_tfrec(tf_train_path, train_img_path, label_path)
    write_tfrec(tf_val_path, val_img_path, label_path)
    
