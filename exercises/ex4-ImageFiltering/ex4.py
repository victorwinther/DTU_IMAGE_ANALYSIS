import matplotlib.pyplot as plt
from skimage import io
from scipy.ndimage import correlate
from skimage.filters import median
from skimage.filters import gaussian
from skimage.filters import prewitt_h
from skimage.filters import prewitt_v
from skimage.filters import prewitt
from skimage.filters import threshold_otsu
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import cv2

input_img = np.arange(25).reshape(5, 5)

print(input_img)

weights = [[0, 1, 0],
		   [1, 2, 1],
		   [0, 1, 0]]

res_img = correlate(input_img, weights)

print(res_img[3,3])

print(res_img)

res_img = correlate(input_img, weights, mode="constant", cval=10)

print(res_img)

img_org = io.imread('data/car.png')
img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
io.imshow(img_org)
io.show()

size = 5
# Two dimensional filter filled with 1
weights = np.ones([size, size])
# Normalize weights
weights = weights / np.sum(weights)

meanImg = correlate(img_org, weights)

def apply_median_filter(img, size):
	footprint = np.ones([size, size])
	out_img = median(img, footprint)
	return out_img

def apply_mean_filter(img, size):
    weights = np.ones([size, size])
    weights = weights / np.sum(weights)

    out_img = correlate(img, weights, mode='reflect')
    return out_img



def printMean():
	gauss5  = apply_mean_filter(img_org, size = 5)
	gauss10 = apply_mean_filter(img_org, size = 10)
	gauss20 = apply_mean_filter(img_org, size = 20)
	gauss40 = apply_mean_filter(img_org, size = 30)

	fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (40, 6))
	ax[0].imshow(gauss5, cmap = 'gray')
	ax[1].imshow(gauss10, cmap = 'gray')
	ax[2].imshow(img_org, cmap = 'gray')
	fig.suptitle('Mean Filter', fontsize = 16)
	plt.show()

def printMedian():
	med5  = apply_median_filter(img_org, size = 5)
	med10 = apply_median_filter(img_org, size = 10)
	med20 = apply_median_filter(img_org, size = 20)
	med40 = apply_median_filter(img_org, size = 30)

	fig, ax = plt.subplots(nrows = 1, ncols = 5, figsize = (40, 6))
	ax[0].imshow(img_org, cmap = 'gray')
	ax[1].imshow(med5, cmap = 'gray')
	ax[2].imshow(med10, cmap = 'gray')
	ax[3].imshow(med20, cmap = 'gray')

	fig.suptitle('Median Filter', fontsize = 16)
	plt.show()

def printGaussian():
	fig, ax = plt.subplots(nrows = 1, ncols = 4, figsize = (40, 6))
	ax[0].imshow(img_org, cmap = 'gray')
	ax[1].imshow(gaussian(img_org, 1), cmap = 'gray')
	ax[2].imshow(gaussian(img_org, 2), cmap = 'gray')
	ax[3].imshow(gaussian(img_org, 3), cmap = 'gray')
	fig.suptitle('Gaussian Filter', fontsize = 16)
	plt.show()
	
def edgeFilter():
	img2_org = io.imread('data/donald_1.png')
	img2_grey = cv2.cvtColor(img2_org, cv2.COLOR_BGR2GRAY)
	prewitth = prewitt_h(img2_grey)
	prewittv = prewitt_v(img2_grey)
	prewittt = prewitt(img2_grey)
	fig, ax = plt.subplots(nrows = 1, ncols = 4, figsize = (40, 6))
	ax[0].imshow(img2_org)
	ax[1].imshow(prewitth, cmap = 'bwr', vmin = -1, vmax = 1)
	im = ax[2].imshow(prewittv, cmap = 'bwr', vmin = -1, vmax = 1)
	divider = make_axes_locatable(ax[2])
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)
	ax[3].imshow(prewittt, cmap = 'bwr', vmin = -1, vmax = 1)
	plt.show()
	

def edge_detection(filterType, kernel_size, thres, plot = False):
	img3_org = io.imread('data/ElbowCTSlice.png')
	img3_grey = cv2.cvtColor(img3_org, cv2.COLOR_BGR2GRAY)
	if filterType == 'gaussianFilter':
		img3_filter = gaussian(img3_grey, kernel_size)
	if filterType == 'medianFilter':
		img3_filter = apply_median_filter(img3_grey, kernel_size)
			
	gradients = prewitt(img3_filter)
	threshold = threshold_otsu(gradients)
	edges = gradients > threshold

	if plot:
		fig,ax = plt.subplots(nrows = 1, ncols = 4, figsize = (40,5))
		ax[0].imshow(img3_org, cmap = 'gray')
		ax[1].imshow(img3_filter, cmap = 'gray')
		ax[2].imshow(gradients, cmap = 'gray')
		ax[3].imshow(edges, cmap = 'gray')
		[ax_.set_axis_off() for ax_ in ax]
		plt.show()

	return edges	

gFilter = 'gaussianFilter'
mFilter = 'medianFilter'
edges = edge_detection(gFilter, 5, 0.02, True)
edges = edge_detection(mFilter, 15, 0.05, True)
	#otso thresholding
	






