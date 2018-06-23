
# coding: utf-8

import numpy as np
import tensorflow as tf
import os.path as ops
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def LeNet(x):    
    
    # Layer 1: Convolutional. Input = 24x32x1. Output = 20x28x32. ReLU activation
    conv1 = tf.layers.conv2d(x, kernel_size=5, filters=32, padding='valid', activation=tf.nn.relu)

    # Poling. input = 20x28x32, output = 10x14x32
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

    # layer 2: convolutional input = 10x14x24, output = 8x12x32
    conv2 = tf.layers.conv2d(pool1, kernel_size=3, filters=64, padding='valid', activation=tf.nn.relu)

    # Poling. input = 8x12x32, output = 4x6x32
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

    fc1 = tf.contrib.layers.flatten(pool2)

    fc1 = tf.layers.dense(fc1, 512)

    logits = tf.layers.dense(fc1, 5, name='prediction')

    return logits


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def toInt(data):
    print('Converting to int')
    data_float = np.zeros(len(data), dtype=np.float16)
    for i, d in enumerate(data):
        data_float[i] = float(d)
    
    return data


def saver_graphs(sess, i, file_name, acc):
    output_node_names = 'x_input,prediction'
    constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names.split(','))
    with tf.gfile.GFile('trained_model/{}-{:.3f}-{}.pb'.format(file_name, acc, i), "wb") as f:
        f.write(constant_graph.SerializeToString())
    print("SAVED MODEL: {}-{:.3f}-{}".format(file_name, acc, i))
    print()


if __name__ == '__main__':

    EPOCHS = 7
    BATCH_SIZE = 40

    print('Loading data ...')

    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')

    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')

    # y_train = toInt(y_train)
    # y_test = toInt(y_test)
    # y_test = toInt(y_test)

    print('Init trainging model ...')
    x = tf.placeholder(tf.float32, (None, 24, 32, 1), name='x_input')
    y = tf.placeholder(tf.int32, (None), 'y_input')
    prob = tf.placeholder_with_default(1.0, shape=())
    # isTraining = tf.placeholder(tf.bool, name='isTraining')
    one_hot_y = tf.one_hot(y, 5)

    rate = 0.001
    logits = LeNet(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # saver = tf.train.Saver()

    #reshap data for tf logits
    X_train = X_train.reshape(len(X_train),24,32,1)
    X_test = X_test.reshape(len(X_test),24,32,1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)

        # tensorboard
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('model/', sess.graph)
        
        print("Training...")
        print()
        
        for i in range(EPOCHS):
            batch_counter = 0
            print("EPOCH {} ...".format(i+1))
            loss_avr = []
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                _, loss = sess.run([training_operation, loss_operation], 
                                feed_dict={x: batch_x, y: batch_y, prob: 0.7})
                # _, loss = sess.run([training_operation, loss_operation], feed_dict={x: batch_x, y: batch_y, isTraining: True})
                batch_counter += 1
                if batch_counter % 4 == 0:
                    print("Loss {}: {}".format((batch_counter//4), loss))
                    loss_avr.append(loss)
            validation_accuracy = evaluate(X_test, y_test)

            loss_avr2 = np.asarray(loss_avr)
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print("LOSS AVR:",np.average(loss_avr2))
            print()
        saver_graphs(sess=sess, i=(i+1), file_name = 'charnet', acc = validation_accuracy) 


        sess.close()
