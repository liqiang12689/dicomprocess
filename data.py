from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2                  #2017.7.17
from lib.LoadDicomSerices import LoadDicomSerices, LoadDicomData
from lib.LoadLabelData import LoadLabelSerices, LoadLabelData
# import tensorflow as tf
# import keras.backend as K
#from libtiff import TIFF    #2017.7.17

class myAugmentation(object):
	
	"""
	A class used to augmentate image
	Firstly, read train image and label seperately, and then merge them together for the next process
	Secondly, use keras preprocessing to augmentate image
	Finally, seperate augmentated image apart into train image and label
	"""

	def __init__(self, train_path="./data1/train/volume", label_path="./data1/train/label", merge_path="./data1/merge", aug_merge_path="./data1/augmentation_merge", aug_train_path="./data1/augmentation_train", aug_label_path="./data1/augmentation_label", img_type="DCM"):
	# def __init__(self, train_path="./data/train/volume", label_path="./data/train/label", merge_path="./data/merge",
	# 			 aug_merge_path="./data/augmentation_merge", aug_train_path="./data/augmentation_train",
	# 			 aug_label_path="./data/augmentation_label", img_type="tif"):

		"""
		Using glob to get all .img_type form path
		"""
		self.train_imgs = LoadDicomSerices(train_path)#训练数据列表
		# self.train_imgs = LoadDicomData(self.traindatalist)#训练数据标签列表
		self.label_imgs = LoadLabelSerices(label_path)
		# self.train_imgs = glob.glob(train_path+"/*."+img_type)   #train_path/*.tif
		# self.label_imgs = glob.glob(label_path+"/*."+img_type)
		self.train_path = train_path
		self.label_path = label_path
		self.merge_path = merge_path
		self.img_type = img_type     #tif
		self.aug_merge_path = aug_merge_path
		self.aug_train_path = aug_train_path
		self.aug_label_path = aug_label_path
		self.slices = len(self.train_imgs)   #the number of tif file
		self.datagen = ImageDataGenerator(
							        rotation_range=0.2,
							        width_shift_range=0.05,
							        height_shift_range=0.05,
							        shear_range=0.05,
							        zoom_range=0.05,
							        horizontal_flip=True,
							        fill_mode='nearest')

	def Augmentation(self):

		"""
		Start augmentation.....
		"""
		trains = self.train_imgs
		labels = self.label_imgs
		path_train = self.train_path
		path_label = self.label_path
		path_merge = self.merge_path
		imgtype = self.img_type
		path_aug_merge = self.aug_merge_path
		if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
			print ("trains can't match labels")
			return 0
		for i in range(len(trains)):
			img_t = load_img(path_train + "/" +str(i)+ "."+imgtype) # bak


			img_l = load_img(path_label + "/" +str(i)+"."+imgtype)  # bak


			x_t = img_to_array(img_t)
			x_l = img_to_array(img_l)
			print (x_l.shape ,x_t.shape)
			x_t[:,:,2] = x_l[:,:,0]
			img_tmp = array_to_img(x_t)
			img_tmp.save(path_merge+"/"+str(i)+"."+imgtype)
			img = x_t
			img = img.reshape((1,) + img.shape)
			savedir = path_aug_merge + "/" + str(i)
			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			self.doAugmentate(img, savedir, str(i))


	def doAugmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='tif', imgnum=30):

		"""
		augmentate one image
		"""
		datagen = self.datagen
		i = 0
		for batch in datagen.flow(img,
                          batch_size=batch_size,
                          save_to_dir=save_to_dir,
                          save_prefix=save_prefix,
                          save_format=save_format):
		    i += 1
		    if i > imgnum:
		        break

	def splitMerge(self):

		"""
		split merged image apart
		"""
		path_merge = self.aug_merge_path
		path_train = self.aug_train_path
		path_label = self.aug_label_path
		for i in range(self.slices):
			path = path_merge + "/" + str(i)
			train_imgs = glob.glob(path+"/*."+self.img_type)
			savedir = path_train + "/" + str(i)
			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			savedir = path_label + "/" + str(i)
			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			for imgname in train_imgs:
				midname = imgname[imgname.rindex("/")+1:imgname.rindex("."+self.img_type)]
				img = cv2.imread(imgname)
				img_train = img[:,:,2]#cv2 read image rgb->bgr
				img_label = img[:,:,0]
				cv2.imwrite(path_train+"/"+str(i)+"/"+midname+"_train"+"."+self.img_type,img_train)
				cv2.imwrite(path_label+"/"+str(i)+"/"+midname+"_label"+"."+self.img_type,img_label)

	def splitTransform(self):

		"""
		split perspective transform images
		"""
		path_merge = "transform"                                         #2017.7.17
		path_train = "transform/data/"                                   #2017.7.17
		path_label = "transform/label/"                                  #2017.7.17

		train_imgs = glob.glob(path_merge+"/*."+self.img_type)
		for imgname in train_imgs:
			midname = imgname[imgname.rindex("/")+1:imgname.rindex("."+self.img_type)]
			img = cv2.imread(imgname)
			img_train = img[:,:,2]#cv2 read image rgb->bgr
			img_label = img[:,:,0]
			cv2.imwrite(path_train+midname+"."+self.img_type,img_train)
			cv2.imwrite(path_label+midname+"."+self.img_type,img_label)



class dataProcess(object):

	def __init__(self, out_rows, out_cols, data_path = "./data1/train/volume", label_path = "./data1/train/label", test_path = "./data1/test", npy_path = "./data1/npydata", img_type = "DCM"):

		"""
		
		"""

		self.out_rows = out_rows
		self.out_cols = out_cols
		self.data_path = data_path
		self.label_path = label_path
		self.img_type = img_type
		self.test_path = test_path
		self.npy_path = npy_path

	def create_train_data(self):
		i = 0
		print('-'*30)
		print('Creating training images...')
		print('-'*30)
		train_imgs = LoadDicomSerices(self.data_path)#训练数据列表
		label_imgs = LoadLabelSerices(self.label_path)
		print('Train:',len(train_imgs))
		print('Label:', len(label_imgs))
		imgdatas = LoadDicomData(train_imgs)
		imglabels = LoadLabelData(label_imgs)
		newimgdatas = np.row_stack(imgdatas)#将多个list数据拼接
		nweimglabels = np.row_stack(imglabels)
		print('loading done')
		print('imgdatas=', imgdatas)
		# np.save(self.npy_path + '/imgs_train.npy', tf.concat(imgdatas, 0))
		np.save(self.npy_path + '/imgs_train.npy', newimgdatas[:, :, :, np.newaxis])#创建思维数组
		np.save(self.npy_path + '/imgs_mask_train.npy', nweimglabels[:, :, :, np.newaxis])
		print('Saving to .npy files done.')

	def create_test_data(self):
		i = 0
		print('-'*30)
		print('Creating test images...')
		print('-'*30)
		testimglist = LoadDicomSerices(self.test_path)  # 训练数据列表
		testimgdatas = LoadDicomData(testimglist)
		newtestimgdatas = np.row_stack(testimgdatas)
		print(len(testimgdatas))
		# imgdatas = np.ndarray((len(testimgdatas),self.out_rows,self.out_cols,1), dtype=np.uint8)
		print('loading done')
		np.save(self.npy_path + '/imgs_test.npy', newtestimgdatas[:, :, :, np.newaxis])
		print('Saving to imgs_test.npy files done.')

	def load_train_data(self):
		print('-'*30)
		print('load train images...')
		print('-'*30)
		imgs_train = np.load(self.npy_path+"/imgs_train.npy")
		print('imgs_train.shape = ', imgs_train.shape)
		imgs_mask_train = np.load(self.npy_path+"/imgs_mask_train.npy")
		imgs_train = imgs_train.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')
		imgs_train /= 255
		mean = imgs_train.mean(axis = 0)
		imgs_train -= mean	
		imgs_mask_train /= 255
		imgs_mask_train[imgs_mask_train > 0.5] = 1
		imgs_mask_train[imgs_mask_train <= 0.5] = 0
		return imgs_train,imgs_mask_train

	def load_test_data(self):
		print('-'*30)
		print('load test images...')
		print('-'*30)
		imgs_test = np.load(self.npy_path+"/imgs_test.npy")
		imgs_test = imgs_test.astype('float32')
		imgs_test /= 255
		mean = imgs_test.mean(axis = 0)
		imgs_test -= mean	
		return imgs_test

if __name__ == "__main__":

	# aug = myAugmentation()
	# aug.Augmentation()
	# aug.splitMerge()
	# aug.splitTransform()
	mydata = dataProcess(512,512)
	mydata.create_train_data()
	mydata.create_test_data()
