#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import scipy.misc as misc


def split():
	"""分割小图"""
	CPATCHA_DIR = 'data/captcha'
	SAVE_DIR = 'data/small'
	images = os.listdir(CPATCHA_DIR)
	all_images = len(images)
	for x in range(0, all_images):
		im_path = os.path.join(CPATCHA_DIR, str(x)+'.jpg')
		im = misc.imread(im_path, mode='RGB')
		# 2*4 small images
		im_crop = im[37:183, 2:290, :]
		# make temp dir
		if not os.path.exists(SAVE_DIR):
			os.makedirs(SAVE_DIR)
		index = 1
		for row in range(0, 2):
			for col in range(0, 4):
				# every small content image
				row_start, row_end = row*73, row*73+72
				col_start, col_end = col*72, col*72+71
				content_image = im_crop[row_start:row_end+1, col_start:col_end+1, :]
				# 对小图进行放大
				# content_resize_img = misc.imresize(content_image, [200, 200])
				img_save_path = os.path.join(SAVE_DIR, str(x)+'_'+str(index)+'.jpg')
				index += 1
				misc.imsave(img_save_path, content_image)


def rebuild():
	"""将所有的类别下的小图凑齐到一个目录并重新命名
	"""
	import shutil
	CAT_DIR = 'data/train_data/'
	SAVE_DIR = 'data/DB2'
	if not os.path.exists(SAVE_DIR):
		os.makedirs(SAVE_DIR)
	label_f = pd.read_table('data/label.txt', names=['data'])
	label_list = label_f['data'].tolist()
	for la in label_list:
		dir_path = os.path.join(CAT_DIR, la)
		index = 1
		for name in os.listdir(dir_path):
			src = os.path.join(dir_path, name)
			des = os.path.join(SAVE_DIR, la+'_'+str(index)+'.jpg')
			shutil.copyfile(src, des)
			index += 1


def rename(start=0):
	"""就地rename"""
	suffix = '.jpg'
	import subprocess
	DIR = 'data/test_data/'
	for file in os.listdir(DIR):
		cmd = 'mv %s %s' % (os.path.join(DIR, file), os.path.join(DIR, str(start)+suffix))
		start += 1
		out = subprocess.call(cmd, shell=True)


def get_success_rate():
	"""获得最终的成功率
	识别的答案和标准答案格式为:
		0:3 5
	0表示识别图片的index
	3 5表示需要点击的位置索引,从上到下从左到右进行编号(从0开始)
	"""
	def judge_equal(recog, ans):
		recog_split = recog.split(' ')
		ans_split = ans.split(' ')
		if (len(recog_split) == 0) or (len(recog_split) != len(ans_split)):
			return False
		recog_list = []
		ans_list = []
		for x in range(0, len(recog_split)):
			if len(recog_split[x]) > 0 and recog_split[x] != None:
				recog_list.append(float(recog_split[x]))
			if len(ans_split[x]) > 0 and ans_split[x] != None:
				ans_list.append(float(ans_split[x]))
		sorted_recog_list = sorted(recog_list)
		sorted_ans_list = sorted(ans_list)
		for y in range(0, len(sorted_ans_list)):
			if sorted_recog_list[y] != sorted_ans_list[y]:
				return False
		return True

	right_count = 0
	recog_table = pd.read_table('data/test_recog_K10_v5.txt', sep=':', names=['index', 'data'])
	ans_table = pd.read_table('data/test_ans.txt', sep=':', names=['index', 'data'])
	assert recog_table['index'].shape[0] == ans_table['index'].shape[0]
	all_count = ans_table['index'].shape[0]
	for x in range(0, all_count):
		recog = str(recog_table['data'][x])
		ans = str(ans_table['data'][x])
		if judge_equal(recog, ans):
			right_count += 1
	return float(right_count) / all_count


if __name__ == '__main__':
	print(get_success_rate())
