#https://www.kaggle.com/mrisdal/intro-to-lstms-w-keras-gpu-for-text-generation
import tensorflow as tf
import os
import argparse
import time
import string
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional,Activation,GRU,SimpleRNN
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint
import random
import sys
import io

class Rnn(object):
	def __init__(self, filenames,learning_rate = 0.01,batch_size=1000,time_steps=25,num_steps=20,data_distribution=[1,1,1],lstm_size=100,maxlen = 10):
		self.__vocab = set()
		self.__sources = filenames
		self.batch_size = batch_size
		self.data_distrubution = data_distribution
		self.time_steps = time_steps
		self.lstm_size = lstm_size
		self.num_steps = num_steps
		self.maxlen = maxlen
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
					self.__vocab.add(char)
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

		train_size,test_size,valid_size = self.data_distrubution
		train_size_count = 0
		test_size_count = 0
		valid_size_count = 0
		train_data = []
		test_data = []
		valid_data = []
		train_datastring = ""
		test_datastring = ""
		valid_datastring = ""
		for line in self.text_gen():
			if train_size != train_size_count:
				for word in line:
					for char in word:
						train_data.append(self.w2i[char])
						train_datastring= train_datastring + char
				train_size_count += 1
			elif test_size != test_size_count:
				for word in line:
					for char in word:
						test_data.append(self.w2i[char])
						test_datastring=test_datastring + char
				test_size_count += 1
				test_data.append(len(self.w2i)-1)       
			elif valid_size != valid_size_count:
				for word in line:
					for char in word:
						valid_data.append(self.w2i[char])
						valid_datastring= valid_datastring + char
				valid_size_count += 1   
			else:
				train_size_count = 0
				test_size_count = 0
				valid_size_count = 0
				for word in line:
					for char in word:
						train_data.append(self.w2i[char])
						train_datastring=train_datastring +char
				train_size_count += 1
		self.train_data = train_data
		self.test_data = test_data
		self.valid_data = valid_data
		self.train_datastring = train_datastring
		self.test_datastring = test_datastring
		self.valid_datastring = valid_datastring
				
	def one_hot(self,sentences):
		x = np.zeros((len(sentences), self.maxlen, len(self.chars)), dtype=np.bool)
		y = np.zeros((len(sentences), len(self.chars)), dtype=np.bool)
		for i, sentence in enumerate(sentences):
			for t, char in enumerate(sentence):
				x[i, t, self.char_indices[char]] = 1
			y[i, self.char_indices[self.next_chars[i]]] = 1
		return(x,y)
	def one_hot_back(self,data_list):
		output = []
		for word in data_list:
			output.append(word.index(max(word)))
		return output
	def sentenctifier(self):
		maxlen = 40
		step = 3
		sentences = []
		next_chars = []
		for i in range(0, len(self.train_datastring) - self.maxlen, step):
			sentences.append(self.train_datastring[i: i + self.maxlen])
			next_chars.append(self.train_datastring[i + self.maxlen])
		print('Number of sequences:', len(sentences), "\n")
		self.sentences = sentences
		self.next_chars = next_chars
		print(sentences[:10], "\n")
		print(next_chars[:10])


	def sample(self,preds, temperature=1.0):
		# helper function to sample an index from a probability array
		preds = np.asarray(preds).astype('float64')
		preds = np.log(preds) / temperature
		exp_preds = np.exp(preds)
		preds = exp_preds / np.sum(exp_preds)
		probas = np.random.multinomial(1, preds, 1)
		return np.argmax(probas)

	def on_epoch_end(self,epoch, logs):
		# Function invoked for specified epochs. Prints generated text.
		# Using epoch+1 to be consistent with the training epochs printed by Keras
		if epoch+1 == 5 or epoch+1 == 15 or epoch+1 == 20:
			print()
			print('----- Generating text after Epoch: %d' % epoch)

			start_index = random.randint(0, len(self.train_datastring) - self.maxlen - 1)
			for diversity in [0.2, 0.5, 1.0, 1.2]:
				print('----- diversity:', diversity)

				generated = ''
				sentence = self.train_datastring[start_index: start_index + self.maxlen]
				generated += sentence
				print('----- Generating with seed: "' + sentence + '"')
				sys.stdout.write(generated)

				for i in range(20):
					x_pred = np.zeros((1, self.maxlen, len(self.chars)))
					for t, char in enumerate(sentence):
						x_pred[0, t, self.char_indices[char]] = 1.

					preds = self.model.predict(x_pred, verbose=0)[0]
					next_index = self.sample(preds, diversity)
					next_char = self.indices_char[next_index]

					generated += next_char
					sentence = sentence[1:] + next_char

					sys.stdout.write(next_char)
					sys.stdout.flush()
				print()
		else:
			print()
			print('----- Not generating text after Epoch: %d' % epoch)



	def train(self):
		self.char_indices = dict((c, i) for i, c in enumerate(self.__vocab))
		self.indices_char = dict((i, c) for i, c in enumerate(self.__vocab))
		self.chars = self.w2i
		self.sentenctifier()
		x,y = self.one_hot(self.sentences)
		generate_text = LambdaCallback(on_epoch_end=self.on_epoch_end)
		#callback genereate_text
		model = Sequential()
		self.model = model
		model.add(LSTM(128, input_shape=(self.maxlen, (len(self.w2i)))))
		model.add(Dense(len(self.w2i)))
		model.add(Activation('softmax'))
		model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
		model.fit(x, y, epochs=100, batch_size=self.batch_size, verbose=2)
		gentext = ''
		start_index = 100
		sentence = self.train_datastring[start_index: start_index + self.maxlen]
		gentext += sentence
		for i in range(20):
			x_pred = np.zeros((1, self.maxlen, len(self.chars)))
			for t, char in enumerate(sentence):
				x_pred[0, t, self.char_indices[char]] = 1.
			output = model.predict(x_pred, verbose=0)[0]
			next_index = (output.tolist()).index(max((output.tolist())))
			print(next_index)
			next_char = self.indices_char[next_index]
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

