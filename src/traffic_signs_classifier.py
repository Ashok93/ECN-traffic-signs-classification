## imports #################################################

import numpy as np # scientific computations library (http://www.numpy.org/)
import tensorflow as tf # deep learning library (https://www.tensorflow.org/)
import cv2 # OpenCV computer vision library (https://opencv.org/)
import matplotlib.pyplot as plt # Plotting library (https://matplotlib.org/)
import pickle # library for saving and retrieving trained model (https://docs.python.org/2/library/pickle.html)
import random

import os # file operations
import math
import csv

from sklearn.model_selection import train_test_split
#############################################################

NUM_CLASSES = 43
curr_dirname = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(os.path.abspath(curr_dirname))
MODEL_EXPORT_DIR = os.path.join(project_root_dir, 'models/new')

class TrafficSignsClassifier:
	def __init__(self, img_path):
		self.img_path = img_path
		self.x_train = None
		self.y_train = None
		self.x_test = None
		self.y_test = None

	def get_images(self):

		images = []
		labels = []

		for c in range(0,43):
			prefix = self.img_path + '/' + format(c, '05d') + '/' # subdirectory for class
			gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
			gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
			next(gtReader) # skip header

			for row in gtReader:
				images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
				labels.append(row[7]) # the 8th column is the label
			gtFile.close()

		return images, labels


	def preprocess_images(self, images, to_color = 'GRAY', size=(30,30)):
		processed_imgs = []
		for image in images:
			image = cv2.resize(image, (size[0], size[1]), interpolation=cv2.INTER_LINEAR)
			#image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
			processed_imgs.append(image)

		return processed_imgs

	def train_test_split_images(self, images, labels, test_size):
		#labels = np_utils.to_categorical(labels, NUM_CLASSES)
		labels = np.array(labels, dtype=np.int8)
		labels = self.convert_to_one_hot(labels, NUM_CLASSES)
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(np.array(images), labels, test_size=test_size)
		self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], self.x_train.shape[2], 3)
		self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], self.x_test.shape[2], 3)

	def create_placeholders(self, nH, nW, nC, nY):

		X = tf.placeholder(tf.float32, shape=[None, nH, nW, nC], name="X")
		Y = tf.placeholder(tf.float32, shape=[None, nY], name="Y")

		return X, Y

	def initialize_parameters(self):
		W1 = tf.get_variable("W1", shape = [5, 5, 3, 6], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
		W2 = tf.get_variable("W2", shape = [5, 5, 6, 16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))

		return { "W1": W1, "W2":W2 }

	def forward_propagation(self, X, parameters):
		W1 = parameters["W1"]
		W2 = parameters["W2"]

		# Conv1 layer with stride 1 and same padding
		Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding="VALID")

		# Relu
		A1 = tf.nn.relu(Z1)

		# max-pool Kernel[2X2] stride 2
		P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides = [1,2,2,1], padding="VALID")

		# Conv2 with stride 1 and same padding
		Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding="VALID")

		# Relu
		A2 = tf.nn.relu(Z2)

		# max-pool kernel[2X2] stride 2
		P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides = [1,2,2,1], padding="VALID")

		# Flatten
		P2 = tf.contrib.layers.flatten(P2)

		#fully connected
		Z3 = tf.contrib.layers.fully_connected(P2, 43, activation_fn=None)

		return Z3

	def compute_cost(self, Z3, Y):
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = Z3, labels = Y))
		return cost


	def random_mini_batches(self, X, Y, mini_batch_size = 64, seed = 0):
		m = X.shape[0] # number of training examples
		mini_batches = []
		np.random.seed(seed)
		permutation = list(np.random.permutation(m))
		shuffled_X = X[permutation,:,:,:]
		shuffled_Y = Y[permutation,:]

		num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning

		for k in range(0, num_complete_minibatches):
			mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
			mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
			mini_batch = (mini_batch_X, mini_batch_Y)
			mini_batches.append(mini_batch)

		return mini_batches

	def convert_to_one_hot(self, Y, C):
		Y = np.eye(C)[Y.reshape(-1)]
		return Y

	def build_model(self, restore = False, learning_rate = 0.001, num_epochs = 100, minibatch_size = 64, print_cost = True):
		#self.predict_data()
		costs = []
		tf.set_random_seed(1)
		seed = 3

		(m, nH, nW, nC) = self.x_train.shape
		nY = self.y_train.shape[1]

		X, Y = self.create_placeholders(nH, nW, nC, nY)
		
		parameters = self.initialize_parameters()

		Z3 = self.forward_propagation(X, parameters)

		cost = self.compute_cost(Z3, Y)

		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

		prediction = tf.argmax(tf.nn.softmax(Z3), 1)

		truth = tf.argmax(Y, 1)
		
		equality = tf.equal(prediction, truth)
		
		accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
		
		init = tf.global_variables_initializer()

		saver = tf.train.Saver()
		
		with tf.Session() as sess:
			sess.run(init)

			if restore == True:
				saver.restore(sess,tf.train.latest_checkpoint(MODEL_EXPORT_DIR))
				traffic_sign_classifier.visualize_dataset(self.x_test[1:5], 2, 2, (8,8))
				pred, tru, acc= sess.run([prediction, truth, accuracy], feed_dict = {X: self.x_test, Y: self.y_test})
				print(acc)
			else:
				for epoch in range(num_epochs):
					minibatch_cost = 0.
					num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
					minibatches = self.random_mini_batches(self.x_train, self.y_train, minibatch_size, seed)
					for minibatch in minibatches:
						(minibatch_X, minibatch_Y) = minibatch
						_ , temp_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})
						minibatch_cost += temp_cost / num_minibatches

					if print_cost == True:
						accuracy= sess.run(accuracy, feed_dict = {X: minibatch_X, Y: minibatch_Y})
						print(accuracy, '%')
						costs.append(minibatch_cost)

					if print_cost == True and epoch % 5 == 0:
						self.save_model(sess, epoch)
						print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))


	def save_model(self, sess, epoch):
		saver = tf.train.Saver()
		saver.save(sess, MODEL_EXPORT_DIR + '/my-model', global_step = epoch)

	def visualize_dataset(self, images, rows = 1, columns = 1, fsize=(8,8)):
		fig = plt.figure(figsize=fsize)

		for i in range(0, rows*columns):
			fig.add_subplot(rows, columns, i + 1)
			plt.imshow(images[i])

		plt.show()


if __name__ == "__main__":
	img_path = os.path.join(project_root_dir, 'GTSRB/Final_Training/Images')
	traffic_sign_classifier = TrafficSignsClassifier(img_path)
	images, labels = traffic_sign_classifier.get_images()
	preprocessed_images = traffic_sign_classifier.preprocess_images(images)
	#sample_imgs = random.sample(preprocessed_images, 6)
	#traffic_sign_classifier.visualize_dataset(sample_imgs, 3, 2, (8,8))
	traffic_sign_classifier.train_test_split_images(preprocessed_images, labels, 0.3)
	traffic_sign_classifier.build_model(True)