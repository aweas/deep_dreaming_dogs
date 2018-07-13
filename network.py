import tensorflow as tf
import numpy as np
import tqdm
import cv2
import skimage.transform
import matplotlib.pyplot as plt


class abstract_network:
    def _inference(self):
        raise NotImplementedError("Must create a deriving class with own architecture implementation")

    def __init__(self, input_shape):
        self.input = None
        self.inference = None
        self.sess = None
        self.dropout_prob = None
        self.logits = None

        self.saver = None
        self.train_writer = None

        self.mode = None

        self.input_shape = input_shape

    def _input_parser(self, img_path, label):
        # convert the label to one-hot encoding
        one_hot = tf.one_hot(label, 2, dtype=np.uint8)

        # read the img from file
        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_image(img_file, channels=3)

        img_decoded = tf.image.convert_image_dtype(
            img_decoded,
            tf.float32,
            saturate=False,
            name=None
        )

        return img_decoded, one_hot

    def set_training_data(self, *args):
        if len(args) == 4:
            self.X_train = args[0]
            self.y_train = args[1]
            self.X_test = args[2]
            self.y_test = args[3]

            self.mode = 'files'

        elif len(args) == 2:
            self.train_location = np.asarray(args[0])
            self.test_location = np.asarray(args[1])

            self.mode = 'strings'

        elif len(args) == 1:
            self.train_location = tf.constant(args[0])
        else:
            raise AttributeError("Invalid number of parameters passed")

    def reset(self):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.input = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name='features')
        self.dropout_prob = tf.placeholder_with_default(1.0, shape=())

    def training(self, epochs=100, batch_size=64, iter_before_validation=10, log_training=False):
        """ Trains network with provided data """

        y_input, loss, optimize = self._prepare_for_training(batch_size, log_training)

        print(f'{int(len(self.train_labels[self.train_labels==0])/len(self.train_labels)*100)}% of training is class 0')
        print(f'{int(len(self.test_labels[self.test_labels==0])/len(self.test_labels)*100)}% of test is class 0')

        for i in range(epochs):
            ls = 0
            training_answers = []

            if self.mode == 'files':
                iterations = int(np.ceil(len(self.X_train) / batch_size))

                for j in tqdm.trange(iterations):
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

            else:
                real_answers = []
                self.sess.run(self.training_init_op)
                self.sess.run(self.test_init_op)

                iterations = int(np.ceil(len(self.train_location) / batch_size))

                for j in tqdm.trange(iterations):
                    point = j * batch_size
                    X_train, y_train = self.sess.run(self.next_training_batch)

                    ls_temp, _, answ = self.sess.run([loss, optimize, self.logits],
                                                     feed_dict={self.input: X_train,
                                                                y_input: y_train,
                                                                self.dropout_prob: 0.5})
                    ls += ls_temp / iterations
                    training_answers.extend(answ)
                    real_answers.extend(y_train)

                if i % iter_before_validation == 0:
                    training_answers = np.asarray(training_answers)
                    real_answers = np.asarray(real_answers)

                    acc_train = 1 - np.sum(np.logical_xor(real_answers[:, 0], np.round(training_answers)[:, 0])) / len(training_answers)

                    if self.test_location is not None:

                        acc = self._get_accuracy(batch_size)
                        print(f'Test accuracy: {int(acc*100)}% \tTraining loss: {ls} \t Training accuracy: {int(acc_train*100)}%')

                    else:
                        print(f'Training loss: {ls} \t Training accuracy: {int(acc_train*100)}%')

    def _read_images(self, files_to_read):
        X = np.empty((len(files_to_read), *self.input_shape), dtype=np.float32)
        y = np.empty((len(files_to_read), 2), dtype=np.uint8)

        for num, i in enumerate(files_to_read):
            X[num] = cv2.imread(i)[:, :, ::-1]

        y['cat' in files_to_read] = [1, 0]
        y['cat' not in files_to_read] = [0, 1]

        return X, y

    def _preprocess_img(self, img):
        return skimage.transform.resize(img, self.input_shape)

    def _prepare_for_training(self, batch_size, log_training):
        """ Prepares all necessary variables and fields """
        self.reset()

        if self.mode is None:
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

        # I MIGHT clean this up someday, but this day has yet to come
        # Prepare everything for training data
        self.train_labels = np.zeros(len(self.train_location), dtype=np.uint8)
        self.train_labels[['dog' in x for x in self.train_location]] = 1
        self.train_labels_tensor = tf.constant(self.train_labels)
        self.train_tensor = tf.constant(self.train_location)

        self.tr_data = tf.data.Dataset.from_tensor_slices((self.train_tensor, self.train_labels_tensor))
        self.tr_data = self.tr_data.map(self._input_parser, num_parallel_calls=8)
        self.tr_data = self.tr_data.batch(batch_size)
        # self.tr_data = self.tr_data.shuffle(10 * batch_size)
        self.tr_data = self.tr_data.prefetch(buffer_size=100 * batch_size)

        self.training_iterator = tf.data.Iterator.from_structure(self.tr_data.output_types,
                                                                 self.tr_data.output_shapes)
        self.next_training_batch = self.training_iterator.get_next()
        self.training_init_op = self.training_iterator.make_initializer(self.tr_data)

        # Prepare everything for test data
        self.test_labels = np.zeros(len(self.test_location), dtype=np.uint8)
        self.test_labels[['dog' in x for x in self.test_location]] = 1
        self.test_labels_tensor = tf.constant(self.test_labels)
        self.test_tensor = tf.constant(self.test_location)

        self.val_data = tf.data.Dataset.from_tensor_slices((self.test_tensor, self.test_labels_tensor))
        self.val_data = self.val_data.map(self._input_parser, num_parallel_calls=8)
        self.val_data = self.val_data.batch(batch_size)
        # self.val_data = self.val_data.shuffle(10 * batch_size)
        self.val_data = self.val_data.prefetch(buffer_size=100 * batch_size)

        self.test_iterator = tf.data.Iterator.from_structure(self.val_data.output_types,
                                                             self.val_data.output_shapes)
        self.next_test_batch = self.test_iterator.get_next()
        self.test_init_op = self.test_iterator.make_initializer(self.val_data)

        return y_input, loss, optimize

    def _get_accuracy(self, batch_size):
        res = []
        real_res = []
        for j in range(int(len(self.test_location) / batch_size) + 1):
            X_test, y_test = self.sess.run(self.next_test_batch)

            res.extend(self.predict(X_test))
            real_res.extend(y_test[:, 0])

        return 1 - np.sum(np.logical_xor(real_res, np.round(res))) / len(res)

    def predict(self, img):
        return self.sess.run(self.logits, feed_dict={self.input: img})[:, 0]

    def freeze_model(self, location):
        inference_nodes = tf.graph_util.remove_training_nodes(tf.get_default_graph().as_graph_def())

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            self.sess,
            inference_nodes,
            ['output']
        )

        with tf.gfile.GFile(location, "wb") as f:
            f.write(output_graph_def.SerializeToString())
