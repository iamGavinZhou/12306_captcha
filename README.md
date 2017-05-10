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
> 1. using Biadu image for captcha break
>     **python baidu_12306.py**
> 2. using ResNet for captcha break
>     **python res_recog_v5.py**
> 3. get success rate
>     **python util.py**

# Results
**test data**: 100 images in `data/test_data`
**success rate**: 41/100=0.41
**time**: 0.695 s

# Other
Is still on build, maybe some bugs exist!
feel free to contact me, **gavinzhou_xd@163.com**, any question is welcomed!
