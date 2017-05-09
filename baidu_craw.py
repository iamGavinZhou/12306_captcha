# -*- coding: utf-8 -*-
import os
import urllib.request
import requests
import re
import scipy.misc as misc
import numpy as np

CRAW_DIR = './data/craw'
LABEL_FILE = './data/label.txt'
TOP_K = 5

def get_response_html(search_cat='电饭煲'):
	"""获得搜索的html"""
	search_url = 'https://image.baidu.com/search/index?tn=baiduimage&word=%s&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=index&fr=&hs=0&xthttps=111111&sf=1&fmq=&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&rsp=-1' % search_cat
	# print(search_url)
	upload_resp = requests.get(search_url)
	res_html = upload_resp.text
	return res_html


def craw_topK_img(search_cat='电饭煲'):
	"""抓取解析的小图"""

	def parse_html(html, top_k=TOP_K):
		'''对获得的html进行解析'''
		#  "adType":"0","middleURL":"https://ss1.bdstatic.com/70cFuXSh_Q1YnxGkpoWK1HF6hhy/it/u=1795291488,415897745&fm=23&gp=0.jpg"
		pattern = re.compile(r'"adType":"0","middleURL":"(\S+)"')
		matches = pattern.findall(html)
		res = []
		if not matches:
			# 没有搜索结果
			return res
		for x in range(0, top_k):
			# 可能会出现不足top_k个小图的情况，暂时没发现
			tags_str = matches[x]
			res.append(tags_str)
		return res

	print("search for %s, get response from [image.baidu.com]...." % search_cat)
	html = get_response_html(search_cat=search_cat)
	print("start pasing urls.....")
	urls = parse_html(html)
	if len(urls) == 0 or urls is None:
		print("no search results!!!")
		return
	# save dir
	if not os.path.exists(CRAW_DIR):
		os.makedirs(CRAW_DIR)
	print("start craw.....")
	for x in range(0, len(urls)):
		url = urls[x]
		print("craw url: " + url)
		# save
		save_path = os.path.join(CRAW_DIR, search_cat+'_'+str(x+1)+'.jpg')
		urllib.request.urlretrieve(url, save_path)
	print("done!!!")


def craw_in_list(name_list):
	"""抓取name_list中的所有的图"""
	if len(name_list) == 0 or name_list is None:
		raise Exception('name_list must not be None')
	print("craw for %d category....." % len(name_list))
	for name in name_list:
		print("craw for %s" % name)
		if name == '本子':
			name = '本子文具'
		craw_topK_img(name)

def get_label_list(label_file=LABEL_FILE):
	"""获得label"""
	import pandas as pd
	label = pd.read_table(label_file, names=['data'])
	return label['data'].tolist()


if __name__ == '__main__':
	# name_list = get_label_list()
	# craw_in_list(name_list)
	# craw_topK_img('本子文具')