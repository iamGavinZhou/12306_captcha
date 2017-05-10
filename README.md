# A simple way to crack 12306 CAPTCHA
The new version of the 12306 Captcha can not be cracked just by using Baidu or Google image tool. Taking into account the ability of the neural network(like ResNet or VGG), a simple way we proposed is using a powerful classification network to extract some 'useful' features from 12306 Captcha image and using that for similarity measure.
Very simple but effective!

# Requirements

> - python2
> - skimage
> - numpy
> - pandas
> - caffe

# How to use
> 1. using Baidu image for captcha break
> - **python baidu_12306.py**
> 2. using ResNet for captcha break
> - **python res_recog_v5.py**
> 3. get success rate
> - **python util.py**

# Results
On my computer, Xeon E5-1620(3.5GHZ) + 64G ram + Titan X (Maxwell)</br>**test data**: 100 images in `data/test_data`</br>**accuracy**: `41/100=0.41` (some chinese character label is not correctly recognized(15%), if with 100% accuracy we will get final result with 0.52)</br>**time**: `0.695s` / per image

# Other
Is still on build, maybe some bugs exist!</br>feel free to contact me, **gavinzhou_xd@163.com**, any question is welcomed!
