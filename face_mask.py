import cv2
import numpy as np
from PIL import Image


with_mask = np.load('face_mask\With_Mask_500.npy')


im = Image.fromarray((with_mask[4]).reshape(50,50,3))
im.show()

without_mask = np.load('face_mask\Without_Mask_500.npy')

im = Image.fromarray((without_mask[5]).reshape(50,50,3))
im.show()

with_mask = with_mask.reshape(501,1 * 50 * 50 * 3)
without_mask = without_mask.reshape(501,1 * 50 * 50 * 3)
X = np.r_[with_mask, without_mask]

labels = np.zeros(X.shape[0])

labels[501:] = 1.0
names = {0 : 'Mask', 1 : 'No Mask'}
color_dict={0:(0,255,0),1:(0,0,255)}