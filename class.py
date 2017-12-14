import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt


import urllib.request as urllib

from datasets import imagenet
from nets import vgg
from nets import inception
from nets import inception_resnet_v2
from nets import resnet_v2
from nets.nasnet import nasnet
from nets import mobilenet_v1
from preprocessing import vgg_preprocessing
from preprocessing import inception_preprocessing

from tensorflow.contrib import slim
model='nasnet_large'


with tf.Graph().as_default():

    # url = 'https://upload.wikimedia.org/wikipedia/commons/d/d9/First_Student_IC_school_bus_202076.jpg'
    # image_string = urllib.urlopen(url).read()
    image_string=open('1.png','rb').read()
    image = tf.image.decode_jpeg(image_string, channels=3)

    if model == 'vgg':
        checkpoints_dir = '/tmp/checkpoints'
        image_size = vgg.vgg_16.default_image_size
        processed_image = vgg_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)

        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(vgg.vgg_arg_scope()):
            # 1000 classes instead of 1001.
            logits, _ = vgg.vgg_16(processed_images, num_classes=1000, is_training=False)
        probabilities = tf.nn.softmax(logits)

        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
            slim.get_model_variables('vgg_16'))
    elif model=='inception_v1':
        checkpoints_dir = '/tmp/checkpoints'
        image_size = inception.inception_v1.default_image_size
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)

        # Create the model, use the default arg scope to configure the batch norm parameters.
        # with slim.arg_scope(inception.inception_v3_arg_scope()):
        #     logits, _ = inception.inception_v3(processed_images, num_classes=1001, is_training=False)
        # probabilities = tf.nn.softmax(logits)
        # with slim.arg_scope(inception.inception_v1_arg_scope()):
        #     logits, _ = inception.inception_v1(processed_images, num_classes=1001, is_training=False)
        # probabilities = tf.nn.softmax(logits)
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            logits, _ = inception.inception_v1(processed_images, num_classes=1001, is_training=False)
        probabilities = tf.nn.softmax(logits)

        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'inception_v1.ckpt'),
            slim.get_model_variables('InceptionV1'))
    elif model=='inception_v3':
        checkpoints_dir = 'C:/Users/cdcd/Downloads/imageClass/inception_v3'
        image_size = inception.inception_v3.default_image_size
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)

        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, _ = inception.inception_v3(processed_images, num_classes=1001, is_training=False)
        probabilities = tf.nn.softmax(logits)

        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'inception_v3.ckpt'),
            slim.get_model_variables('InceptionV3'))
    elif model=='inception_v4':
        checkpoints_dir = 'C:/Users/cdcd/Downloads/imageClass/inception_v4'
        image_size = inception.inception_v4.default_image_size
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)

        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception.inception_v4_arg_scope()):
            logits, _ = inception.inception_v4(processed_images, num_classes=1001, is_training=False)
        probabilities = tf.nn.softmax(logits)

        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'inception_v4.ckpt'),
            slim.get_model_variables('InceptionV4'))
    elif model=='inception_resnet':
        checkpoints_dir = 'C:/Users/cdcd/Downloads/imageClass/inception_resnet_v2'
        image_size = inception_resnet_v2.inception_resnet_v2.default_image_size
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)

        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, _ = inception_resnet_v2.inception_resnet_v2(processed_images, num_classes=1001, is_training=False)
        probabilities = tf.nn.softmax(logits)

        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'inception_resnet_v2.ckpt'),
            slim.get_model_variables('InceptionResnetV2'))

    elif model=='nasnet_large':
        checkpoints_dir = 'C:/Users/cdcd/Downloads/imageClass/nasnet-a_large'
        image_size = nasnet.build_nasnet_large.default_image_size
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)

        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(nasnet.nasnet_large_arg_scope()):
            logits, _ = nasnet.build_nasnet_large(processed_images, num_classes=1001, is_training=False)
        probabilities = tf.nn.softmax(logits)

        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'model.ckpt'),
            slim.get_model_variables(''))
    elif model=='resnet_v2_50':
        checkpoints_dir = 'C:/Users/cdcd/Downloads/imageClass/resnet_v2_50'
        image_size = resnet_v2.resnet_v2_50.default_image_size
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)

        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, _ = resnet_v2.resnet_v2_50(processed_images, num_classes=1001, is_training=False)
        probabilities = tf.nn.softmax(logits)

        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'resnet_v2_50.ckpt'),
            slim.get_model_variables('resnet_v2_50'))
    elif model=='resnet_v2_152':
        checkpoints_dir = 'C:/Users/cdcd/Downloads/imageClass/resnet_v2_152'
        image_size = resnet_v2.resnet_v2_152.default_image_size
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)

        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, _ = resnet_v2.resnet_v2_152(processed_images, num_classes=1001, is_training=False)
        probabilities = tf.nn.softmax(logits)

        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'resnet_v2_152.ckpt'),
            slim.get_model_variables('resnet_v2_152'))
    elif model=='mobilenet_v1':
        checkpoints_dir = 'C:/Users/cdcd/Downloads/imageClass/mobilenet_v1'
        image_size = mobilenet_v1.mobilenet_v1.default_image_size
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)

        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            logits, _ = mobilenet_v1.mobilenet_v1(processed_images, num_classes=1001, is_training=False)
        probabilities = tf.nn.softmax(logits)

        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'mobilenet_v1_1.0_224.ckpt'),
            slim.get_model_variables('MobilenetV1'))

    with tf.Session() as sess:
        init_fn(sess)
        np_image, probabilities = sess.run([image, probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

    plt.figure()
    plt.imshow(np_image.astype(np.uint8))
    plt.axis('off')
    plt.show()

    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        # Shift the index of a class name by one.
        print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index + 1]))