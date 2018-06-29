# -*- coding: utf-8 -*-

import os
import cv2
import random
from os import listdir 
from os.path import isfile, join 

class Generator:
	def __init__(self, data_info, Xfolder, Yfolder, test_floder, batch_size, train_pers, input_size):
		self.input_size = input_size
		self.Xfolder = Xfolder
		self.Yfolder = Yfolder
		self.batch_num = 0
		self.train_batches = [[]]
		self.train_batch_index = 0
		self.valid_batches = [[]]
		self.valid_batch_index = 0
		self.test_batches = [[]]
		self.test_batch_num = 0
		#содержит в себе список пар картинок [Y, X] -> [оригинал, корраптед]
		files = self.get_image_pair(data_info)	
		#Количество тренировочных файлов кратных номеру батча
		total_train_files = int(len(files) * train_pers - (len(files) * train_pers) % batch_size)
		#Общее количество файлов
		total_files = int(len(files) - (len(files) % batch_size))
		
		self.split_files_batch(0, total_train_files, self.train_batches, 
			self.train_batch_index, batch_size, files)
		self.train_batch_index = 0		
		self.split_files_batch(total_train_files, total_files, self.valid_batches,
			self.valid_batch_index, batch_size, files)		
		files = []
		for f in listdir(test_floder):
			f2 = join(test_floder, f)
			if isfile(f2):
				files.append(f2)
		self.split_files_batch(0, len(files), self.test_batches,
			self.test_batch_num, batch_size, files)

	def split_files_batch(self, start, finish, batches, batch_num, batch_size, files):
		for i in range(start, finish):
			if len(batches[batch_num]) >= batch_size:
				batch_num += 1
				batches.append([]) 
			batches[batch_num].append(files[i])

	def get_image_pair(self, data_info):
		data = open(data_info, 'r')

		#получаем список пар файлов [оригинал, корраптед]
		files = []
		for line in data.readlines():
			file = line.split()
			if(len(file) == 0):
				continue
			for i in range(1, len(file)):
				files.append([self.Yfolder + file[0], self.Xfolder + file[i]])	
		random.seed(1337)
		random.shuffle(files)	
		return files

	def get_train_batch(self):
		x_batch = []
		y_batch = []
		for f in self.train_batches[self.train_batch_index]:
			x = cv2.imread(f[0])
			y = cv2.imread(f[1])
			x = cv2.resize(x, (self.input_size, self.input_size))
			y = cv2.resize(y, (self.input_size, self.input_size))
			x_batch.append(x)
			y_batch.append(y)
		self.train_batch_index = (self.train_batch_index + 1) % (len(self.train_batches))
		return x_batch, y_batch

	def get_valid_batch(self):
		x_batch = []
		y_batch = []
		for f in self.valid_batches[self.valid_batch_index]:
			x = cv2.imread(f[0])
			y = cv2.imread(f[1])
			x = cv2.resize(x, (self.input_size, self.input_size))
			y = cv2.resize(y, (self.input_size, self.input_size))
			x_batch.append(x)
			y_batch.append(y)
		self.valid_batch_index = (self.valid_batch_index + 1) % (len(self.valid_batches))
		return x_batch, y_batch

	def get_test_batch(self):
		x_batch = []
		for f in self.test_batches[self.test_batch_num]:
			x = cv2.imread(f)
			x = cv2.resize(x, (self.input_size, self.input_size))
			x_batch.append(x)
		self.test_batch_num = (self.test_batch_num + 1) % (len(self.test_batches))
		return x_batch


'''
img_folder = 'images/'
gen = Gen(img_folder + 'kak_ygodno.txt', img_folder + 'X/', img_folder + 'new_Y/', 'porch/', 5, 0.8, 512)


print('train')
print(len(gen.get_train_batch()[0]))
print(len(gen.get_train_batch()[0]))
print(len(gen.get_train_batch()[0]))
print(len(gen.get_train_batch()[0]))
print(len(gen.get_train_batch()[0]))
print('valid')
print(len(gen.get_valid_batch()[0]))
print(len(gen.get_valid_batch()[0]))
print(len(gen.get_valid_batch()[0]))
print(len(gen.get_valid_batch()[0]))
print(len(gen.get_valid_batch()[0]))
print(len(gen.get_valid_batch()[0]))
print('test')
print(len(gen.get_test_batch()))
print(len(gen.get_test_batch()))
print(len(gen.get_test_batch()))
print(len(gen.get_test_batch()))
print(len(gen.get_test_batch()))
print(len(gen.get_test_batch()))
'''