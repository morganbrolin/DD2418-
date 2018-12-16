import tensorflow as tf
import os
import argparse
import time
import string
import numpy as np
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, LSTM,Activation
import matplotlib.pyplot as plt
import random
import sys
import io

class Rnn(object):
	def __init__(self, filenames,epochs=30,batch_size=128,time_steps=5,maxlen = 20,generated_text_size = 400):
		self.__vocab = set()
		self.__sources = filenames
		self.batch_size = batch_size
		self.time_steps = time_steps
		self.maxlen = maxlen
		self.epochs = epochs
		self.generated_text_size = generated_text_size
	def text_gen(self):
		for fname in self.__sources:
			with open(fname, encoding='utf8', errors='ignore') as f:
				for line in f:
					yield self.clean_line(line)
	def clean_line(self, line):
		return line
	def build_vocabulary(self):
		"""
		Build vocabulary of words from the provided text files
		"""
		print("Building vocab..")
		for line in self.text_gen():
			for word in line:
				for char in word:
					self.__vocab.add(char), 
		self.__vocab.add(" ")

		self.write_vocabulary()
		self.i2w = list(self.__vocab)
		i = 0
		w2i = {}
		for x in (self.i2w):
			w2i[x] = i
			i = i + 1
		self.w2i = w2i
		print("Finished building vocab.")
	@property
	def vocabulary_size(self):
		return len(self.__vocab)
	def vocab_exists(self):
		return os.path.exists('vocab.txt')
	def read_vocabulary(self):
		vocab_exists = self.vocab_exists()
		if vocab_exists:
			with open('vocab.txt') as f:
				for line in f:
					for char in line:
						self.__vocab.add(char)
			self.__vocab.add(" ")
		self.i2w = list(self.__vocab)
		i = 0
		w2i = {}
		for x in (self.i2w):
			w2i[x] = i
			i = i + 1
		self.w2i = w2i
		return vocab_exists


	def write_vocabulary(self):
		with open('vocab.txt', 'w') as f:
			for w in self.__vocab:
				f.write('{}\n'.format(w))
	def load_data(self):
		if self.vocab_exists():
			self.read_vocabulary()
		else:
			self.build_vocabulary()

		train_data = []
		train_datastring = ""
		for line in self.text_gen():
			for word in line:
				for char in word:
					train_data.append(self.w2i[char])
					train_datastring= train_datastring + char
		self.train_data = train_data
		self.train_datastring = train_datastring
				
	def one_hot(self,sentences):
		x = np.zeros((len(sentences), self.maxlen, len(self.w2i)), dtype=np.bool)
		y = np.zeros((len(sentences), len(self.w2i)), dtype=np.bool)
		for i, sentence in enumerate(sentences):
			for t, char in enumerate(sentence):
				x[i, t, self.w2i[char]] = 1
			y[i, self.w2i[self.next_chars[i]]] = 1
		return(x,y)
	def sentenctifier(self):
		sentences = []
		next_chars = []
		for i in range(0, len(self.train_datastring) - self.maxlen, self.time_steps):
			sentences.append(self.train_datastring[i: i + self.maxlen])
			next_chars.append(self.train_datastring[i + self.maxlen])
		print('Number of sequences:', len(sentences), "\n")
		return(sentences,next_chars)


	def train(self):
		self.sentences,self.next_chars =self.sentenctifier()
		self.test_sentences,self.test_next_chars =self.sentenctifier()
		x,y = self.one_hot(self.sentences)
		model = Sequential()
		self.model = model
		model.add(LSTM(self.batch_size, input_shape=(self.maxlen, (len(self.w2i)))))
		model.add(Dense(len(self.w2i)))
		model.add(Activation('softmax'))
		model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
		fitting = model.fit(x, y, epochs=self.epochs,validation_split=0.33, batch_size=self.batch_size, verbose=2)
		model.save('model')
		plt.plot(fitting.history['loss'])
		plt.plot(fitting.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'])
		plt.show()
		while True:
			sentence = input("Start text to be generated: ")
			if len(sentence)<self.maxlen:
				print("you need a bigger text",self.maxlen)
			else:
				sentence = sentence[0:self.maxlen]
				gentext = ''
				gentext += sentence
				for i in range(self.generated_text_size):
					x_pred = np.zeros((1, self.maxlen, len(self.w2i)))
					for t, char in enumerate(sentence):
						x_pred[0, t, self.w2i[char]] = 1.
					output = model.predict(x_pred, verbose=0)[0]
					sums = 0
					rint = random.random()
					counter= 0
					next_index = 0
					for i in ((output.tolist())):
						sums = sums + i
						counter += 1
						if rint < sums:
							next_index = counter-1

							break

					#next_index = (output.tolist()).index(max((output.tolist())))
					next_char = self.i2w[next_index]
					gentext += next_char
					sentence = sentence[1:] + next_char

				print("generatet text: ",gentext)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Random Indexing word embeddings')
	parser.add_argument('-fv', '--force-vocabulary', action='store_true', help='regenerate vocabulary')
	parser.add_argument('-c', '--cleaning', action='store_true', default=False)
	parser.add_argument('-co', '--cleaned_output', default='cleaned_example.txt', help='Output file name for the cleaned text')
	args = parser.parse_args()

	if args.force_vocabulary:
		os.remove('vocab.txt')
	if args.cleaning:
		#ri = RandomIndexing([os.path.join('data', 'example.txt')])
		ri = RandomIndexing(['example.txt'])
		with open(args.cleaned_output, 'w') as f:
			for part in ri.text_gen():
				f.write("{}\n".format(" ".join(part)))
	else:
		dir_name = "data"
		filenames = [os.path.join(dir_name, fn) for fn in os.listdir(dir_name)]

		rnn = Rnn(filenames)
		rnn.load_data()
		rnn.train()