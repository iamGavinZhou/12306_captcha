#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, '/home/common/caffe/python')
import caffe
import skimage.io
import numpy as  np
import tensorflow as tf
from caffe.proto import caffe_pb2
from synset import *


def load_mean_bgr():
	""" bgr mean pixel value image, [0, 255]. [height, width, 3] """
	with open("data/Res_model/ResNet_mean.binaryproto", mode='rb') as f:
		data = f.read()
	blob = caffe_pb2.BlobProto()
	blob.ParseFromString(data)

	mean_bgr = caffe.io.blobproto_to_array(blob)[0]
	assert mean_bgr.shape == (3, 224, 224)

	return mean_bgr.transpose((1, 2, 0))


def preprocess(img):
	"""Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
	mean_bgr = load_mean_bgr()
	# print 'mean blue', np.mean(mean_bgr[:, :, 0])
	# print 'mean green', np.mean(mean_bgr[:, :, 1])
	# print 'mean red', np.mean(mean_bgr[:, :, 2])
	out = np.copy(img) * 255.0
	out = out[:, :, [2, 1, 0]]  # swap channel from RGB to BGR
	out -= mean_bgr
	return out


def load_image(path, size=224):
	"""read the image
	returns image of shape [224, 224, 3]
	"""
	img = skimage.io.imread(path)
	short_edge = min(img.shape[:2])
	yy = int((img.shape[0] - short_edge) / 2)
	xx = int((img.shape[1] - short_edge) / 2)
	crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
	resized_img = skimage.transform.resize(crop_img, (size, size))
	return resized_img


def init_net(layers, using_gpu=True):
	"""init ResNet"""
	if using_gpu:
		caffe.set_mode_gpu()
	else:
		caffe.set_mode_cpu()
	prototxt = "data/Res_model/ResNet-%d-deploy.prototxt" % layers
	caffemodel = "data/Res_model/ResNet-%d-model.caffemodel" % layers
	net = caffe.Net(prototxt, caffemodel, caffe.TEST)

	return net


def load_caffe(img_p, layers=101, using_gpu=True):
	"""classify the image"""
	if using_gpu:
		caffe.set_mode_gpu()
	else:
		caffe.set_mode_cpu()

	prototxt = "data/Res_model/ResNet-%d-deploy.prototxt" % layers
	caffemodel = "data/Res_model/ResNet-%d-model.caffemodel" % layers
	net = caffe.Net(prototxt, caffemodel, caffe.TEST)

	net.blobs['data'].data[0] = img_p.transpose((2, 0, 1))
	assert net.blobs['data'].data[0].shape == (3, 224, 224)
	net.forward()

	caffe_prob = net.blobs['prob'].data[0]
	print_prob(caffe_prob)

	return net


def load_caffe_fc(img_p, layers=101, using_gpu=True):
	"""ResNet fc layer representation of the image"""
	if using_gpu:
		caffe.set_mode_gpu()
	else:
		caffe.set_mode_cpu()

	prototxt = "data/Res_model/ResNet-%d-deploy.prototxt" % layers
	caffemodel = "data/Res_model/ResNet-%d-model.caffemodel" % layers
	net = caffe.Net(prototxt, caffemodel, caffe.TEST)

	net.blobs['data'].data[0] = img_p.transpose((2, 0, 1))
	assert net.blobs['data'].data[0].shape == (3, 224, 224)
	net.forward()

	caffe_fcb = net.blobs['fc1000'].data[0]  # numpy ndarray with shape(1000,)
	caffe_fc = caffe_fcb.reshape(1, 1000)

	return caffe_fc


def print_prob(prob):
	"""print the classification probablity
	return top 1 label
	"""
	#print prob
	pred = np.argsort(prob)[::-1]

	# Get top1 label
	top1 = synset[pred[0]]
	print "Top1: ", top1
	# Get top5 label
	top5 = [synset[pred[i]] for i in range(5)]
	print "Top5: ", top5
	return top1


def test_single():
	img = load_image("data/cat.jpg")
	print img.shape
	img_p = preprocess(img)
	# layers could be 50, 101, 152
	load_caffe(img_p, layers=101)


def test_ini(layers=101, using_gpu=True):
	net = init_net(layers=layers, using_gpu=using_gpu)
	files = os.listdir('data/test_data')
	for name in files:
		img_path = os.path.join('data/test_data', name)
		img = load_image(img_path)
		img_p = preprocess(img)

		# feed image
		net.blobs['data'].data[0] = img_p.transpose((2, 0, 1))
		assert net.blobs['data'].data[0].shape == (3, 224, 224)
		net.forward()

		print '-----------------------------'
		caffe_prob = net.blobs['prob'].data[0]
		print_prob(caffe_prob)


if __name__ == '__main__':
	import time
	start = time.clock()
	test_ini()
	print (time.clock() - start)
