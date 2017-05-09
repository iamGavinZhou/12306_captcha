# -*- encooding: utf-8 -*-
# 
import os
import urllib.request
import requests
import re
import scipy.misc as misc
import numpy as np

CPATCHA_DIR = './data/test_data'
fi = open('./data/baidu_recog.txt', 'w')
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2272.89 Safari/537.36"

def baidu_image_upload(img_path):
    url = "http://image.baidu.com/pictureup/uploadshitu?fr=flash&fm=index&pos=upload"

    # im.save("./query_temp_img.png")
    raw = open(img_path, 'rb').read()

    files = {
        'fileheight'   : "0",
        'newfilesize'  : str(len(raw)),
        'compresstime' : "0",
        'Filename'     : "image.jpg",
        'filewidth'    : "0",
        'filesize'     : str(len(raw)),
        'filetype'     : 'image/jpg',
        'Upload'       : "Submit Query",
        'filedata'     : ("image.jpg", raw)
    }

    resp = requests.post(url, files=files, headers={'User-Agent':UA})

    #  resp.url
    redirect_url = "http://image.baidu.com" + resp.text
    return redirect_url


def get_upload_url_res(img_path):
	upload_url = baidu_image_upload(img_path)
	upload_resp = requests.get(upload_url)
	res_html = upload_resp.text
	
	return res_html


def get_content_res(img_path):
	# get response html
	html = get_upload_url_res(img_path)
	# pattern = re.compile(r"'multitags':\s*'(.*?)'")
	pattern = re.compile(r"<strong>(\S+)</strong>_百度百科")
	matches = pattern.findall(html)
	if not matches:
		# 需要通过相似图片查找
		return 'UNKOWN'

	tags_str = matches[0]
	result =  list(filter(None, tags_str.replace('\t', ' ').split()))
	# if result == None or len(result) == 0:
	# 	return 'UNKOWN'

	# top 3
	res = ''
	res_set = set(result)
	for t in res_set:
		if result.index(t) < 3:
			 res += t
			 res += '|'
	return res[0:-1]


def main():
	# extract small images from 12306 CAPTCHA image
	# recognize the small images by image.baidu.com
	images = os.listdir(CPATCHA_DIR)
	all_images = len(images)
	print('start recognition process, with %d images' % all_images)
	for x in range(0, all_images):
		im_path = os.path.join(CPATCHA_DIR, str(x)+'.jpg')
		im = misc.imread(im_path, mode='RGB')
		# 2*4 small images
		im_crop = im[37:183, 2:290, :]
		# make temp dir
		if not os.path.exists('./data/temp'):
			os.makedirs('./data/temp')
		index = 1
		fi.write(str(x)+':'+'\n')
		for row in range(0, 2):
			for col in range(0, 4):
				# every small content image
				row_start, row_end = row*73, row*73+72
				col_start, col_end = col*72, col*72+71
				content_image = im_crop[row_start:row_end+1, col_start:col_end+1, :]
				# 对小图进行放大
				content_resize_img = misc.imresize(content_image, [200, 200])
				img_save_path = os.path.join('./data/temp', str(x)+'_'+str(index)+'.jpg')
				index += 1
				misc.imsave(img_save_path, content_resize_img)
				# get small content image label
				content_res = get_content_res(img_save_path)
				# write result
				fi.write('('+str(row)+','+str(col)+')'+' '+content_res+'\n')
		print('index %d done!' % x)

	fi.close()
	print('end process!')


if __name__ == '__main__':
	# html = get_upload_url_res('./data/temp/0_3.jpg')
	# res = get_content_res('./data/temp/0_3.jpg')
	# print(res)
	main()
