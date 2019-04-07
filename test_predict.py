from unet import *
from data import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


test_img = np.load('imgs_mask_test.npy')
print (test_img.shape)
test_img = test_img[4,:,:,0]
print (test_img.shape)
test_img = Image.fromarray(np.uint8(test_img * 255))
plt.imshow(test_img)
test_img.save("./test",  'JPEG')

