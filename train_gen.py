# -*- coding: utf-8 -*-

import os
import cv2
import random
from os import listdir 
from os.path import isfile, join 

class Gen:
	def __init__(self, data_info, Xfolder, Yfolder, Testfloder,train_pers, input_size):
		self.train_pers = train_pers
		self.input_size = input_size
		self.x_folder = Xfolder
		self.y_folder = Yfolder
		self.test_floder = Testfloder

		self.files = []
		data = open(data_info, 'r')
		for line in data.readlines():
			file = line.split()
			if(len(file) == 0):
				continue
			for i in range(1, len(file)):
				self.files.append([file[0], file[i]])
		random.seed(1337)
		random.shuffle(self.files)		
		self.j = 0	

	def get_epoch(self, batch_size, is_training = True):
		x_batch = []
		y_batch = []

		if is_training:
			for i in range(batch_size):
				X = cv2.imread(self.x_folder + self.files[self.j * batch_size + i][1])
				Y = cv2.imread(self.y_folder + self.files[self.j * batch_size + i][0])
				x_batch.append(X)
				y_batch.append(Y)
			self.j +=1
			# print(self.j)
			# print(x_batch)
			# print(y_batch)
			if self.j >= len(self.files) // batch_size:
				self.j=0
		else:
			for j in range(	int(len(self.files) * 0.8) // batch_size , len(self.files) // batch_size ):	
				x_batch.append([])
				y_batch.append([])

				for i in range(0, batch_size):
					X = cv2.imread(self.x_folder + self.files[j * batch_size + i][1])
					Y = cv2.imread(self.y_folder + self.files[j * batch_size + i][0])
					
					x_batch[len(x_batch) - 1].append(X)
					y_batch[len(y_batch) - 1].append(Y)
		
		return x_batch, y_batch	
	def get_test(self):

		files = []
		for f in listdir(self.test_floder):
			f2 = join(self.test_floder, f)
			if isfile(f2):
				files.append(f2)

		# print(files)
		images = [None] * len(files)
		for i in range(len(files)):
			images[i] = cv2.resize(cv2.imread(files[i]), (512, 512))

		return images
#gen = Gen('kak_ygodno.txt', 'images/X/', 'images/Y/', 80, 256)

#X, Y = gen.get_batch(5)