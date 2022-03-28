# Maksim Makarenko
# Unsupervised segmentation of SEM images
# 20.04.2021
import skimage
# import numpy as np
# from skimage import io
# import os, re
# import matplotlib.pyplot as plt
# from sklearn.neighbors import NearestNeighbors
# from sklearn.cluster import KMeans, SpectralClustering
# from sklearn.preprocessing import OneHotEncoder
# import scipy.ndimage as ndi
# from skimage import data
# from skimage.exposure import histogram
# from skimage import color, segmentation
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.exposure import histogram
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage import data
from skimage.filters import threshold_multiotsu
from scipy.ndimage import gaussian_filter

def random_slice(image_shape, region_size=200):
    if not hasattr(region_size, '__getitem__'):
        # if 1D size, copy to sceond dimension
        region_size = [region_size]*2
    wx, wy = image_shape[:2]
    b = (wx - region_size[0]) - 1
    cx = np.random.randint(0,b)
    b = (wx - region_size[1]) - 1
    cy = np.random.randint(0,b)
    return (slice(cx,cx+region_size[0]),slice(cy,cy+region_size[1]))

fruit_dict = {
 7 : 'apple',
 6 : 'orange',
 8 : 'pepper',
 3 : 'grape',
 4 : 'green plants',
 5 : 'lemons',
 11 : 'avocado',
 9 : 'bananas',
 12 : 'starfruit',
 10 : 'unknown',
 2 : 'onion',
 1 : 'potato'
}


def merge_masks(mask, realfake):
    output = mask.copy()
    fruit_mask = mask > 0
    output[fruit_mask] = 2 * mask[fruit_mask] - (realfake[fruit_mask] + 1) // 2
    return output

def divide_masks(mask, realfake):
    return (mask + realfake) // 2

def encode_dict(fruit_dict):
    encoded_dict = {}
    for key in fruit_dict:
        encoded_dict[key * 2] = 'fake ' + fruit_dict[key]
        encoded_dict[key * 2 - 1] = 'real ' + fruit_dict[key]
    return encoded_dict

class FruitSegmentation():
    def __init__(self, rgb, mask, hyspec=None, remove_board=True):
        """
        :param sigma:
        :param convex:
        :param th:
        :param classes:
        """
        self.image = rgb
        self.mask = mask[:,:,0]
        self.hyspec = hyspec
        if remove_board:
            self.mask[self.mask == -1.0] = 0.0
        self.realfake = mask[:, :, 1]
        self.merged_labels = merge_masks(self.mask, self.realfake)
        self.encoded_dict = encode_dict(fruit_dict)

    def fit(self):
        """
        :param image:
        :return:
        """
        # create markers using grayscale threshold
        labels = skimage.measure.label(self.mask.astype(np.int64),background=0)
        # sometimes it could be better to use convex_hull of a segment
        self.segments = np.clip(self.mask, 0, 1)
        self.labels = labels
        self.regions = regionprops(labels)
        self.eval_summary()

    def fit_predict(self, *pargs, **kargs):
        self.fit()
        return self.labels

    def show_colorized(self, ax=None, mask_type='mask'):
        """
        :param ax:
        :param region:
        :return:
        """
        if ax is None:
            ax = plt.gca()
        mask = self.__getattribute__(mask_type)
        regions_colorized = label2rgb(mask, image=self.image, bg_label=0)
        ax.imshow(regions_colorized)
        ax.set_axis_off()

    def eval_centroid(self, label):
        return ndi.measurements.center_of_mass(self.segments, self.labels, label)

    def eval_color(self):
        mu_rgb = []
        std_rgb = []
        if self.hyspec is not None:
            mu_spectral = []
            std_spectral = []
        for region in self.regions:
            mu, std = self.eval_average_rgb(region)
            mu_rgb.append(mu)
            std_rgb.append(std)
            if self.hyspec is not None:
                mu, std = self.eval_average_spectra(region)
                mu_spectral.append(mu)
                std_spectral.append(std)
        self.mu_rgb = mu_rgb
        self.std_rgb = std_rgb
        if self.hyspec is not None:
            self.mu_spectral = mu_spectral
            self.std_spectral = std_spectral

    def eval_average_rgb(self, region):
        minr, minc, maxr, maxc = region.bbox
        image_crop = self.image[minr:maxr, minc:maxc]
        image_crop = image_crop[region.filled_image]
        return image_crop.mean(axis=0), image_crop.std(axis=0)

    def eval_average_spectra(self, region):
        minr, minc, maxr, maxc = region.bbox
        image_crop = self.hyspec[minr:maxr, minc:maxc]
        image_crop = image_crop[region.filled_image]
        image_crop[image_crop > 1.0] = 0
        return image_crop.mean(axis=0), image_crop.std(axis=0)

    def eval_summary(self):
        self.n_segments = len(np.unique(self.labels)) - 1
        self.seg_areas = np.array([region.area for region in self.regions])
        self.mu_area = self.seg_areas.mean()
        self.sigma_area = self.seg_areas.std()
        self.centroids = np.array([region.centroid for region in self.regions])
        self.fruit_labels = []
        for region in self.regions:
            minr, minc, maxr, maxc = region.bbox
            image_crop = self.merged_labels[minr:maxr, minc:maxc]
            self.fruit_labels.append(self.encoded_dict[image_crop.max()])

    def add_summary(self, ax=None, slices=None, text_shift=20, fontsize=30, show_centroids=True):
        if ax is None:
            ax = plt.gca()
        for (cx, cy), area, label in zip(self.centroids,self.seg_areas, self.fruit_labels):
            if slices is not None:
                xs, ys = slices[:2]
                if (cx < xs.start or cx > xs.stop) or (cy < ys.start or cy > ys.stop):
                    continue
                else:
                    cx-=xs.start
                    cy-=ys.start
            if show_centroids:
                ax.plot([cy],[cx],'ob',lw=50)
            ax.text(cy, cx+text_shift, label, fontsize=fontsize, color='w')

    def show(self, fig=None, with_summary=False, **summary_kw):
        """
        :param fig:
        :return:
        """
        if fig is None:
            fig = plt.gcf()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(self.image)
        ax1.set_axis_off()
        fig.suptitle("Number of segments = %d with average size = %.1f and std = %.1f"
                     % (self.n_segments, self.mu_area, self.sigma_area))
        self.show_colorized(ax=ax2, mask_type='mask')
        if with_summary:
            self.add_summary(ax2, **summary_kw)

    def show_region(self, label=0, with_mask=False, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.set_title('%s' % self.fruit_labels[label])
        region = self.regions[label]
        minr, minc, maxr, maxc = region.bbox
        image_crop = self.image[minr:maxr, minc:maxc]
        if with_mask:
            image_crop = label2rgb(region.image, image=image_crop, bg_label=0)
        ax.imshow(image_crop)
        ax.set_axis_off()