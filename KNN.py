import numpy as np
from sklearn.neighbors import NearestNeighbors
from glob import glob
import os
from skimage import io
import matplotlib.pyplot as plt

""" Use kNN as baseline algorithm for finidng nearest neighbors.  Validate results by observing classification error.  
The concept being that a model with less classification error will find nearest neighbors better as well."""

imgs = np.load(r'D:\pycharm_projects\AWSgeo\data.npy')
# imgs_labels = np.load(r'D:\pycharm_projects\AWSgeo\labels.npy')

logits = np.load(r'D:\pycharm_projects\AWSgeo\Tensorboard\model_2019-12-18-08-45-59\data_logits.npy')

############## sklearn ##############
rand_arrange = np.random.permutation(len(logits))
ind = -1
neigh = NearestNeighbors(5)
neigh.fit(logits[rand_arrange[:-1000]])
knns = neigh.kneighbors(logits[rand_arrange[ind]].reshape(1,-1), 6, return_distance=False)

plt.figure();plt.imshow(imgs[rand_arrange[ind]])
io.imshow_collection(imgs[rand_arrange[knns[0]]])




########## openCV ##################

