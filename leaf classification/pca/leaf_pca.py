# -*- coding: utf-8 -*-
# @Author: wu.liao ps: a copy from Selfish Gene
# @Date:   2017-04-14 13:15:14
# @Last Modified by:   wu.liao
# @Last Modified time: 2017-04-14 17:10:11

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn import model_selection
from sklearn import decomposition
from sklearn import linear_model
from sklearn import ensemble
from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KernelDensity
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score

from skimage.transform import rescale
from scipy import ndimage as ndi

# 使用
matplotlib.style.use('fivethirtyeight')

# load the data
dataDir = 'input/'
trainData = pd.read_csv(dataDir + 'train.csv')
classEncoder = LabelEncoder()
trainLabels = classEncoder.fit_transform(trainData.ix[:,'species'])
trainIDs = np.array(trainData.ix[:,'id'])

# go over images and store them in a list
numImages = 1584

shapesMatrix = np.zeros([2,numImages])
listOfImages = []
for k in range(numImages):
    imageFilename = dataDir + 'images/' + str(k+1) + '.jpg'
    currImage = mpimg.imread(imageFilename)
    shapesMatrix[:,k] = np.shape(currImage)
    listOfImages.append(currImage)

# create a large 3d array with all images
maxShapeSize = shapesMatrix.max(axis=1)
print maxShapeSize
for k in range(len(maxShapeSize)):
    if maxShapeSize[k] % 2 == 0:
        maxShapeSize[k] += 311 # don't know why
    else:
        maxShapeSize[k] += 310

fullImageMatrix3D = np.zeros(np.hstack((maxShapeSize,np.shape(shapesMatrix[1]))).astype(int),dtype=np.dtype('u1'))
destXc = (maxShapeSize[1]+1)/2; destYc = (maxShapeSize[0]+1)/2
for k, currImage in enumerate(listOfImages):
    Yc, Xc = ndi.center_of_mass(currImage)
    Xd = destXc - Xc; Yd = destYc - Yc
    fullImageMatrix3D[round(Yd):round(Yd)+np.shape(currImage)[0],round(Xd):round(Xd)+np.shape(currImage)[1],k] = currImage

xValid = fullImageMatrix3D.mean(axis=2).sum(axis=0) > 0
yValid = fullImageMatrix3D.mean(axis=2).sum(axis=1) > 0
xlims = (np.nonzero(xValid)[0][0],np.nonzero(xValid)[0][-1])
ylims = (np.nonzero(yValid)[0][0],np.nonzero(yValid)[0][-1])
print xValid
fullImageMatrix3D = fullImageMatrix3D[ylims[0]:ylims[1],xlims[0]:xlims[1],:]