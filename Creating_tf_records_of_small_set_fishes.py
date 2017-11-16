#Creating labels for our images


import os
import numpy as np
import cv2
import tensorflow as tf

IMAGE_DIR =r'C:\Users\Moondra\Desktop\TF_FISH_PROJECT\FINAL_FISHES'
tfrecords_filename = 'small_fish.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)

categories = os.listdir(IMAGE_DIR)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



labels =[i for i in range(0, len(os.listdir(IMAGE_DIR)))]

for num, category in  enumerate(categories):
    images_and_label = [(os.path.join(IMAGE_DIR, category, image),
                         num) for image in os.listdir(
                             os.path.join(IMAGE_DIR, category))]


    for image_label in images_and_label:
        image, label = image_label
        print(image)
        try:
            img = cv2.imread(image)
            img = cv2.resize(img, (299, 299), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)
        except  cv2.error as e:
            
                 print(e)
                 continue


        feature = {'train/label': _int64_feature(label),
                   'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))
                   }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
writer.close()


            


   

    
    

