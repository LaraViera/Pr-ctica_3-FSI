# -*- coding: utf-8 -*-

# Sample code to use string producer.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code"""

    """ ponemos todo el vector a cero y metemos un 1 en la posicion x"""
    o_h = np.zeros(n)
    o_h[x] = 1
    return o_h


num_classes = 3
batch_size = 10  ### <---


### batch_size = 4


# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    ### nº que van a dar
    example_batch_list = []
    ### etiquetas que hacemos con el one_hot
    label_batch_list = []

    ### paths son las rutas de ficheros
    ### enumerate, enumera las listas del fichero
    for i, p in enumerate(paths):
        ###-- apertura de ficheros
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        ###-- decodificacion de imagenes
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), one_hot(i, num_classes)  # [float(i)]
        image = tf.image.rgb_to_grayscale(image, name=None)
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        ###-- le quitamos dos dimensiones de la matriz a las imagenes con el conv2d
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        ###-- el max_pooling2d de cada 4 elementos coge el mas grande
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)
        o5 = tf.layers.conv2d(inputs=o4, filters=128, kernel_size=3, activation=tf.nn.relu)  ### <---
        o6 = tf.layers.max_pooling2d(inputs=o5, pool_size=2, strides=2)  ### <---

        ### modificamos la siguiente línea para añadirle el nº de clases
        # h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * num_classes, 18 * 33 * 64]), units=5,
        h = tf.layers.dense(inputs=tf.reshape(o6, [batch_size * num_classes, 12 * 20 * 64]), units=5,  ### <---
                            activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=num_classes, activation=tf.nn.softmax)
    return y


example_batch_train, label_batch_train = dataSource(
    ["Dataset/train/0/*.jpg", "Dataset/train/1/*.jpg", "Dataset/train/2/*.jpg"], batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(
    ["Dataset/valid/0/*.jpg", "Dataset/valid/1/*.jpg", "Dataset/valid/2/*.jpg"], batch_size=batch_size)
example_batch_test, label_batch_test = dataSource(
    ["Dataset/test/0/*.jpg", "Dataset/test/1/*.jpg", "Dataset/test/2/*.jpg"], batch_size=batch_size)

###-- con el reuse a true no mezclamos el mismo modelo
example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, reuse=True)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - tf.cast(label_batch_train, tf.float32)))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - tf.cast(label_batch_valid, tf.float32)))
cost_test = tf.reduce_sum(tf.square(example_batch_test_predicted - tf.cast(label_batch_test, dtype=tf.float32)))
###-- Descenso del gradiente --###
### cuanto más pequeño el learning_rate, más lento aprende pero más seguro
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

y_true = tf.placeholder(tf.float32, [None, 3])  # etiquetas
y = tf.placeholder(tf.float32, [None, 3])  # etiquetas
# a partir del elemento mayor de cada fila
y_sup = tf.argmax(y, 1)
y_true_max = tf.argmax(y_true, 1)
# comprobamos si las etiquetas sup y la true son iguales en un vector
correct_prediction = tf.equal(y_sup, y_true_max)
# para comprobar, pasamos booleamos a fraccion y calculamos la media
precision = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    control = True
    epoch = 0
    error_prev = 1000
    errors_valid = []
    error_valid = 0
    while control and epoch < 200:
        epoch += 1
        sess.run(optimizer)
        error_valid = sess.run(cost_valid)
        errors_valid.append(error_valid)
        if epoch % 20 == 0:
            print("Iter:", epoch, "---------------------------------------------")
            print(sess.run(label_batch_train))
            print(sess.run(example_batch_train_predicted))
            print("Error:", error_valid)
            print("Error:", sess.run(cost))

        if error_prev - error_valid < 0.005:
            control = False
        else:
            previousError = errors_valid[len(errors_valid) - 2]

    # --------------------------------------------------
    #
    #       TESTING
    #
    # --------------------------------------------------
    print("-----------------------")
    print("   Empieza el test...  ")
    print("-----------------------")

    print("Error de test: ", sess.run(cost_test))
    percent = sess.run(precision, feed_dict={y: example_batch_test_predicted.eval(), y_true: label_batch_test.eval()}) * 100
    print("Precisión de test (%): ", percent)

    plt.plot(errors_valid)
    plt.show()

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)
