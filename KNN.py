import numpy as np
from sklearn.neighbors import NearestNeighbors
from glob import glob
import os
from skimage import io

""" Use kNN as baseline algorithm for finidng nearest neighbors.  Validate results by observing classification error.  
The concept being that a model with less classification error will find nearest neighbors better as well."""

files = glob(r'C:\Users\justjo\Downloads\geological_similarity\geological_similarity\**\*.jpg', recursive=True)
data = np.array([io.imread(x) for x in files])
labels = np.array([os.path.basename(os.path.dirname(x)) for x in files])
############## sklearn ##############









########## openCV ##################

