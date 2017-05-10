#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import scipy.misc as misc
import numpy as np
import pandas as pd

from pandas import DataFrame
from caffe_util import *

"""思路:
(1) 识别12306验证码的中文字符
(2) 搜索库，获得百度图片上抓的５张图（done）
(3) 切割12306验证码，获得小图(８张)（done）
(4) 获得每张小图的向量表示(ResNet)（done）
(5) 获得库中５张图的向量表示
(6) 获得５张图分别与８张小图哪个最近且同时满足相似度大于0.75，形成可能的答案集A
(7) 对Ａ中的每张图分别与除自身以外的图(向量)进行相似度的比较，相似度小于阈值则从A中删除

出现的问题:
(1) 
"""

TEST_DIR = 'data/test_data/'
TEMP_DIR = 'data/craw_recog_data/temp'
CRAW_DB_PATH = 'data/DB2/'
LABEL_FILE = 'data/label.txt'
DB_FILE = 'data/DB2.csv'
TOP_K = 10
LAYERS = 101
THRES = 0.60
if not os.path.exists(TEMP_DIR):
	os.makedirs(TEMP_DIR)

print 'start network initialization.....'
start_ini = time.clock()
net = init_net(layers=LAYERS, using_gpu=True)
print 'done!!!, using time: %.4f s' % (time.clock()-start_ini)


def pearson_relative(vec1, vec2):
	"""pearson相关系数
	[-1,1], 绝对值越大越相关
	"""
	assert vec1.shape==vec2.shape
	cov = np.mean(vec1*vec2)-(np.mean(vec1)*np.mean(vec2))
	std = np.std(vec1)*np.std(vec2)
	return cov/std


def cos_relative(vec1, vec2):
	"""计算余弦相似度
	[-1,1],越接近1表示方向越接近,越接近-1表示方向越相反
	"""
	assert vec1.shape==vec2.shape
	dot = np.sum(vec1*vec2)
	sqt = np.sqrt(np.sum(vec1**2))*np.sqrt(np.sum(vec2**2))
	return dot/sqt


def recog_label():
	"""识别测试集下所有12306验证码的中文字符"""
	# label一样的文字也存在不同的扭曲变形
	label_file = pd.read_table('data/test_label.txt', names=['data'])
	label_list = label_file['data'].tolist()
	return label_list


def split_to_small_img(img_path, index):
	"""将12306验证码分割成小图"""
	res = []
	im = misc.imread(img_path, mode='RGB')
	im_crop = im[37:183, 2:290, :]
	for row in range(0, 2):
		for col in range(0, 4):
			# every small content image
			row_start, row_end = row*73, row*73+72
			col_start, col_end = col*72, col*72+71
			content_img = im_crop[row_start:row_end+1, col_start:col_end+1, :]
			# resize the small content image
			# content_resize_img = misc.imresize(content_img, [200, 200])
			img_save_path = os.path.join(TEMP_DIR, str(index)+'_'+str(row)+'_'+str(col)+'.jpg')
			misc.imsave(img_save_path, content_img)
			res.append(img_save_path)
	return res


def get_ResNet_vector(img_path):
	"""获得图像的ResNet的向量表达"""
	img = load_image(img_path)
	img_p = preprocess(img)
	# feed image
	net.blobs['data'].data[0] = img_p.transpose((2, 0, 1))
	assert net.blobs['data'].data[0].shape == (3, 224, 224)
	net.forward()
	# fc layer feature
	caffe_fcb = net.blobs['fc1000'].data[0]  # numpy ndarray with shape(1000,)
	caffe_fc = caffe_fcb.reshape(1, 1000)
	return caffe_fc


def get_label_list():
	"""获得所有label"""
	label_file = pd.read_table(LABEL_FILE, names=['data'])
	label_list = label_file['data'].tolist()
	return label_list

print 'get label list...'
label_list = get_label_list()
print 'label length: %d' % len(label_list)  


def get_all_DB_vec():
	"""获得所有抓取图的vector表示"""
	names = []
	for label in label_list:
		for index in range(1, TOP_K+1):
			img_path_jpg = os.path.join(CRAW_DB_PATH, label+'_'+str(index)+'.jpg')
			img_path_png = os.path.join(CRAW_DB_PATH, label+'_'+str(index)+'.png')
			if os.path.exists(img_path_jpg):
				names.append(img_path_jpg)
			if os.path.exists(img_path_png):
				names.append(img_path_png)
	nums = len(label_list) * TOP_K
	db_vecs = np.zeros((nums, 1000), dtype=np.float32)
	for x in range(0, nums):
		path_t = names[x]
		vec_t = get_ResNet_vector(path_t)
		db_vecs[x] = vec_t
	return db_vecs


print 'start get all DB vectors.....'
start_db = time.clock()
if not os.path.exists(DB_FILE):
	print 'no pre-computed data, start calc.....'
	db_vecs = get_all_DB_vec()  # 抓取的图所代表的vector
	df = DataFrame(db_vecs)
	df.to_csv(DB_FILE, header=False, index=False)
else:
	print 'using pre-computed data...'
	db_vecs = pd.read_csv(DB_FILE, header=None).as_matrix()
print 'done!!!, using time: %.4f s' % (time.clock()-start_db)


def get_spec_vec(label):
	"""获得DB中label所代表的vector"""
	label_index = label_list.index(label)
	return db_vecs[label_index*TOP_K:label_index*TOP_K+TOP_K, :]


def calc_dis(vec1, vec2):
	"""计算两个向量的距离"""
	assert vec1.shape == vec2.shape
	return np.sqrt(np.sum((vec1-vec2) ** 2))


def recog(img_path, index):
	"""识别12306验证码"""
	# split
	small_imgs_path = split_to_small_img(img_path, index)
	assert len(small_imgs_path)==8
	'''Phase A
	(1) 获得12306验证码8张小图的vector表示
	(2) 获得抓取的图的vector表示
	(3) 对每张抓取的图判断最相似的图加入到集合A(vec距离)
	(4) 校正
	'''
	# 1
	# 获得12306小图所代表的vector
	vecs = np.zeros((8, 1000), dtype=np.float32) 
	for x in range(0, 8):
		img_path_t = small_imgs_path[x]
		vec = get_ResNet_vector(img_path_t)
		vecs[x] = vec
	
	# 2
 	# 获得DB中label所代表的图的vector
	label = recog_label()[index]
	db_s_vec = get_spec_vec(label)

	# 3
	# 找出每个抓取的图最相似的小图
	min_indexes = []
	for x in range(0, TOP_K):
		vec_d = db_s_vec[x]
		dis_t = np.zeros((1, 8))
		for y in range(0, 8):
			vec_s = vecs[y]
			dis = calc_dis(vec_s, vec_d)
			dis_t[0, y] = dis
		min_dis = np.min(dis_t)
		min_index = np.argwhere(dis_t==min_dis)[0,1]  # 与抓取的图距离最小的图的index
		# 相似度要满足大于等于0.75
		re_t = cos_relative(vec_d, vecs[min_index])
		if re_t >= 0.75:
			min_indexes.append(min_index)

	min_index_dict = dict()  # 统计出现的次数
	for index in min_indexes:
		if not index in min_index_dict.keys():
			min_index_dict[index] = 1
		else:
			min_index_dict[index] += 1
	'''
	对出现次数超过TOP_K一半的小图进行二次查找
	找出与之相似度最高的一张图
	'''
	refine_list = []
	for index in min_index_dict.keys():
		if min_index_dict[index]  > (TOP_K//2):
			refine_list.append(index)
			
	min_index_set = set(min_indexes)
	for x in refine_list:
		vec_d = vecs[x]
		dis_t = np.zeros((1, 8))
		for y in range(0, 8):
			vec_s = vecs[y]
			dis = calc_dis(vec_s, vec_d)
			dis_t[0, y] = dis
		# 找出距离最小的一个
		dist_sort = np.sort(dis_t)
		s2_least = dist_sort[0, 0:2]
		for least in s2_least:
			least_index = np.argwhere(dis_t==least)[0,1]
			if least_index == x:
				continue
			min_index_set.add(least_index)

	# 4
	# calibration
	# 不满足阈值条件的采用删除的策略
	min_index_list = list(min_index_set)
	all_sat = [True] * len(min_index_list)
	for x in range(0, len(min_index_list)):
		if not all_sat[x]:
			continue
		vec1 = vecs[min_index_list[x]]
		for y in range(0, len(min_index_list)):
			if (y == x) or (not all_sat[y]) :
				continue
			vec2 = vecs[min_index_list[y]]
			simi = cos_relative(vec1, vec2)
			if simi < THRES:
				all_sat[x] = False
				break
	fi_index = []
	for x in range(0, len(min_index_list)):
		if all_sat[x]:
			fi_index.append(min_index_list[x])
	return fi_index


def main():
	"""识别12306验证码"""
	fi = open('data/test_recog.txt', 'w')
	print 'all preparation done!!!'
	print 'start recognition procedure.....'
	all_start = time.clock()
	files = os.listdir(TEST_DIR)
	print 'all test data: %d' % len(files)
	for index in range(0, len(files)):
		img_path = os.path.join(TEST_DIR, str(index)+'.jpg')
		ans = recog(img_path, index)
		# print 'recog: ' + img_path
		# print 'ans: ' + ','.join([str(x) for x in ans])
		recog_ans = ''
		for x in ans:
			recog_ans += str(x)
			recog_ans += ' '
		fi.write(str(index)+':'+recog_ans[0:-1]+'\n')
	print 'All used: %.4f s' % (time.clock()-all_start)
	print 'Avg time: %.4f s' % ((time.clock()-all_start)/len(files))


if __name__ == '__main__':
	main()
