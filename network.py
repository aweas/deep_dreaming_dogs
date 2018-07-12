import tensorflow as tf
import numpy as np
import tqdm


class abstract_network:
    def _inference(self):
        raise NotImplementedError("Must create a deriving class with own architecture implementation")

    def __init__(self, input_shape, X_train=None, y_train=None, X_test=None, y_test=None, mode='files'):
        self.input = None
        self.inference = None
        self.sess = None
        self.dropout_prob = None
        self.logits = None

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.saver = None
        self.train_writer = None

        self.mode = mode

        self.input_shape = input_shape

    def reset(self):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.input = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name='features')
        self.dropout_prob = tf.placeholder_with_default(1.0, shape=())

    def training(self, epochs=100, batch_size=64, iter_before_validation=10, log_training=False):
        """ Trains network with data provided in constructor """
        y_input, loss, optimize = self._prepare_for_training(log_training)

        iterations = int(len(self.X_train) / batch_size)
        for i in tqdm.trange(epochs):
            ls = 0
            for j in range(iterations):
                point = j * batch_size
                ls_temp, _, answ = self.sess.run([loss, optimize, self.logits],
                                                 feed_dict={self.input: self.X_train[point:point + batch_size],
                                                            y_input: self.y_train[point:point + batch_size],
                                                            self.dropout_prob: 0.5})
                ls += ls_temp / iterations

            if i % iter_before_validation == 0:
                acc_train = 1 - np.sum(np.logical_xor(self.y_train[point:point + batch_size, 0], np.round(answ)[:, 0])) / batch_size

                if self.X_test is not None and self.y_test is not None:
                    acc = self._get_accuracy(self.X_test, self.y_test)
                    print(f'Test accuracy: {int(acc*100)}% \tTraining loss: {ls} \t Training accuracy: {int(acc_train*100)}%')
                else:
                    print(f'Training loss: {ls} \t Training accuracy: {int(acc_train*100)}%')

    def _prepare_for_training(self, log_training):
        """ Prepares all necessary variables and fields """
        self.reset()

        if any([arg is None for arg in [self.X_train, self.y_train]]):
            raise TypeError("Training data cannot be empty!")

        y_input = tf.placeholder(tf.float32, shape=(None, 2))

        with tf.variable_scope('cnn'):
            self.inference = self._inference()

        self.logits = tf.nn.softmax(self.inference, name='output')

        with tf.name_scope('training'):
            loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y_input, logits=self.inference))
            optimize = tf.train.AdamOptimizer().minimize(loss)

        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        self.saver = tf.train.Saver()

        if log_training:
            self.train_writer = tf.summary.FileWriter('logs/', self.sess.graph)

        return y_input, loss, optimize

    def _get_accuracy(self, X_test, y_test, batch_size=128):
        res = []
        for j in range(int(len(X_test) / batch_size) + 1):
            point = j * batch_size
            res.extend(self.predict(X_test[point:point + batch_size]))

        return 1 - np.sum(np.logical_xor(np.asarray(y_test)[:, 0], np.round(res))) / len(X_test)

    def predict(self, img):
        return self.sess.run(self.logits, feed_dict={self.input: img})[:, 0]

    def save_model(self, location):
        self.saver.save(self.sess, location)
