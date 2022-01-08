"""
visualize results for test image
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable
import transforms as transforms
from skimage import io
from skimage.transform import resize
from final import my_Model
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from collections import OrderedDict
import sys
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
#from losses.model1 import my_Model
from time import time

cut_size = 44
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

raw_img = io.imread('1.jpg')
gray = rgb2gray(raw_img)

#yshen_start=time()

gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)

img = gray[:, :, np.newaxis]

img = np.concatenate((img, img, img), axis=2)
img = Image.fromarray(img)
inputs = transform_test(img)

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

net =my_Model()
net = torch.nn.DataParallel(net)

checkpoint = torch.load("PublicTest_model.t7")
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()
ncrops, c, h, w = np.shape(inputs)

inputs = inputs.view(-1, c, h, w)
inputs = inputs.cuda()
inputs = Variable(inputs)

naveed_start = time()
outputs = net(inputs)
naveed_stop=time()
print('prediction time 1 is:', (naveed_stop-naveed_start))
outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

start=time()
score = F.softmax(outputs_avg, dim=0)
#print('predict emotion cost time: %f' %(time() - start))
_, predicted = torch.max(outputs_avg.data, 0)
yshen_stop=time()


#print('test time 2 is:', (yshen_stop-yshen_start))
#print('predicted time is', (time()-start))
print("The Expression is %s" %str(class_names[int(predicted.cpu().numpy())]))


