import tensorflow as tf
import numpy as np
import tqdm


class neural_network:
    def _inference(self):
        raise NotImplementedError("Must create a deriving class with own architecture implementation")
    #     inp = self.input
    #     with tf.name_scope('classificator'):
    #         with tf.name_scope('convolution'):
    #             layer = tf.layers.conv2d(inp, 32, 3, activation=tf.nn.relu, name='layer1')
    #             layer = tf.layers.max_pooling2d(layer, pool_size=3, strides=2)
    #             layer = tf.layers.conv2d(layer, 64, 3, activation=tf.nn.relu)
    #             layer = tf.layers.max_pooling2d(layer, pool_size=3, strides=2)
    #             layer = tf.nn.dropout(layer, self.dropout_prob)
    #             layer = tf.layers.conv2d(layer, 64, 3, activation=tf.nn.relu)
    #             layer = tf.layers.max_pooling2d(layer, pool_size=3, strides=2)
    #             layer = tf.layers.conv2d(layer, 64, 3, activation=tf.nn.relu, padding='same')
    #             layer = tf.layers.max_pooling2d(layer, pool_size=3, strides=2)
    #             layer = tf.layers.conv2d(layer, 32, 5, activation=tf.nn.relu, padding='same', name='layer-1')
    #             layer = tf.nn.dropout(layer, .5)
    #             layer = tf.layers.max_pooling2d(layer, pool_size=3, strides=2)
    # #             layer = tf.layers.conv2d(layer, 16, 5, activation=tf.nn.relu)
    # #             layer = tf.layers.max_pooling2d(layer, pool_size=3, strides=2)
    #             layer = tf.layers.flatten(layer)

    #         with tf.name_scope('dense'):
    #             layer = tf.layers.dense(layer, 1024, activation=tf.nn.relu)
    #             layer = tf.nn.dropout(layer, self.dropout_prob)
    #             layer = tf.layers.dense(layer, 512, activation=tf.nn.relu)
    #             layer = tf.layers.batch_normalization(layer)
    #             layer = tf.layers.dense(layer, 128, activation=tf.nn.relu)
    #             layer = tf.layers.batch_normalization(layer)
    #             layer = tf.layers.dense(layer, 2, activation=tf.nn.softmax)

        # return layer

    def __init__(self, input_shape, X_train=None, y_train=None, X_test=None, y_test=None):
        self.input = None
        self.inference = None
        self.sess = None
        self.dropout_prob = None

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.saver = tf.train.Saver()
        self.train_writer = tf.summary.FileWriter('logs/', self.sess.graph)

        self.input_shape = input_shape

    def reset(self):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.input = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name='features')
        self.dropout_prob = tf.placeholder_with_default(1.0, shape=())

    def training(self, X_train=None, y_train=None, X_test=None, y_test=None, epochs=100, batch_size=64):
        self.reset()

        # If arguments are empty, take fields
        X_train = self.X_train if X_train is None else X_train
        y_train = self.y_train if y_train is None else y_train
        X_test = self.X_test if X_test is None else X_test
        y_test = self.y_test if y_test is None else y_test

        if any([arg is None for arg in [X_train, y_train]]):
            raise TypeError("Training data cannot be empty!")

        y_input = tf.placeholder(tf.float32, shape=(None, 2))

        with tf.variable_scope('cnn'):
            self.inference = self._inference()

        logits = tf.nn.softmax(self.inference, name='output')

        with tf.name_scope('training'):
            loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y_input, logits=self.inference))
            optimize = tf.train.AdamOptimizer().minimize(loss)

        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        iterations = int(len(X_train) / batch_size)
        for i in tqdm.trange(epochs):
            ls = 0
            for j in range(iterations):
                point = j * batch_size
                ls_temp, _, answ = self.sess.run([loss, optimize, logits],
                                                 feed_dict={self.input: X_train[point:point + batch_size],
                                                            y_input: y_train[point:point + batch_size],
                                                            self.dropout_prob: 0.5})
                ls += ls_temp / iterations

            if i % 10 == 0:
                acc_train = 1 - np.sum(np.logical_xor(np.asarray(y_train)[point:point + batch_size, 0], np.round(answ)[:, 0])) / batch_size

                if X_test is not None and y_test is not None:
                    acc = self._get_accuracy(X_test, y_test)
                    print(f'Test accuracy: {int(acc*100)}% \tTraining loss: {ls} \t Training accuracy: {int(acc_train*100)}%')
                else:
                    print(f'Training loss: {ls} \t Training accuracy: {int(acc_train*100)}%')

    def _get_accuracy(self, X_test, y_test, batch_size=128):
        res = []
        for j in range(int(len(X_test) / batch_size) + 1):
            point = j * batch_size
            res.extend(self.predict(X_test[point:point + batch_size]))

        return 1 - np.sum(np.logical_xor(np.asarray(y_test)[:, 0], np.round(res))) / len(X_test)

    def predict(self, img):
        logits = tf.nn.softmax(self.inference, name='output')

        return self.sess.run(logits, feed_dict={self.input: img})[:, 0]
