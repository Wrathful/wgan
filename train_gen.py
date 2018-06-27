import os
import cv2
import random

class Gen:
	def __init__(self, data_info, Xfolder, Yfolder, train_pers, input_size):
		self.train_pers = train_pers
		self.input_size = input_size
		self.x_folder = Xfolder
		self.y_folder = Yfolder

		self.x_files = []
		self.y_files = []
		data = open(data_info, 'r')
		for line in data.readlines():
			files = line.split()
			if(len(files) == 0):
				continue
			self.y_files.append(files[0])
			x_files = []
			for i in range(1, len(files)):
				x_files.append(files[i])
			self.x_files.append(x_files)

	def get_batch(self, batch_size, is_training = True):
		x_batch = []
		y_batch = []
		for i in range(0, batch_size):
			if(is_training):
				number = random.randint(0, int(len(self.x_files) * self.train_pers))
			else:
				number = random.randint(int(len(self.x_files) * self.train_pers), len(self.x_files) - 1)
			number2 =  random.randint(0, len(self.x_files[number]) - 1)

			X = cv2.imread(self.x_folder + self.x_files[number][number2])
			X = cv2.resize(X, (self.input_size,self.input_size))

			Y = cv2.imread(self.y_folder + self.y_files[number])
			Y = cv2.resize(Y, (self.input_size,self.input_size))

			x_batch.append(X)
			y_batch.append(Y)
		return x_batch, y_batch	


#gen = Gen('kak_ygodno.txt', 'images/X/', 'images/Y/', 80, 256)

#X, Y = gen.get_batch(5)