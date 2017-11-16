




#more testing tfrecords


import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread, imresize
import os
import tensorflow as tf
import numpy
import cv2


#Gather image paths

DIR = r'C:\Users\Moondra\Desktop\DATA\IMAGE_2'

images = [os.path.join(DIR, image) for image in os.listdir(DIR)]




def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

tfrecords_filename = 'testing_2.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

for image in images:
    img = cv2.imread(image)
    img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)


    feature ={'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    

    
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

writer.close()




tfrecords_filename = r'C:\Users\Moondra\Desktop\Transfer Learning Tutorials\testing_2.tfrecords'


with tf.Session() as sess:
    feature = {'train/image': tf.FixedLenFeature([],tf.string)
               }

    filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=1)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features=feature)
    image = tf.decode_raw(features['train/image'], tf.float32)
    image = tf.reshape(image, [299, 299, 3])
    print(image.shape)
    images = tf.train.shuffle_batch([image], batch_size=3, capacity=30, num_threads=1, min_after_dequeue=10)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for batch_index in range(2):
        img  = sess.run([images])
        img = img.astype(np.uint8)
        for j in range(6):
            plt.subplot(2, 3, j+1)
            plt.imshow(img[j, ...])
            plt.show()
    coord.request_stop()
    coord.join(threads)
    sess.close()





    
