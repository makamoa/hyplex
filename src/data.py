import sys
import matplotlib
if sys.platform=='darwin':
   matplotlib.use("TKAgg")
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
from colors import color
from skimage import io
from skimage.color import rgb2gray
import os
from scipy.signal import lfilter
import colour
from tmm.tmm_core import coh_tmm
from sklearn.datasets import make_blobs
from collections.abc import Iterable
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import dbscan, OPTICS, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn import manifold
from sklearn import mixture, cluster
from collections import defaultdict
import dill as pickle

mpl_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

class AttributeDict(defaultdict):
    def __init__(self):
        super(AttributeDict, self).__init__()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

def smooth(wl,intensity,n=15):
    #additional function for smoothing spectra
    b = [1.0 / n] * n
    a = 1
    intensity = lfilter(b, a, intensity)
    return wl, intensity

def smooth2(wl, y, box_pts=15):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return wl, y_smooth

class Spectra():
    def __init__(self,
                 sample_file,
                 mirror_file,
                 normalized=True,
                 wl_range=[400, 750],
                 plot_color=False,
                 transform=None,
                 scaling=1.0):
        spectra = Spectra.load_data(sample_file)
        spectra_mirror = Spectra.load_data(mirror_file)
        self.transform = transform
        self.sample = Spectra.clean_data(spectra)
        self.mirror = Spectra.clean_data(spectra_mirror)
        self.wl_range = wl_range
        self.wl, self.intensity = self.get_intensity(normalized)
        self.intensity*=scaling
        self.f = self.interpolate()
        self.color = Spectra.get_color(self.wl, self.intensity,
                                       plot=plot_color)

    @staticmethod
    def load_data(fname):
        spectra = pd.read_csv(fname,
                              skiprows=33,
                              skipfooter=1,
                              engine='python',
                              delimiter=';',
                              names=['wl', 'intensity'],
                              dtype={'wl': float, 'intensity': float})
        return spectra

    @staticmethod
    def clean_data(data):
        data = data[data.intensity.notna()]
        return data

    def get_intensity(self, normalized):
        wl = self.sample.wl.values
        intensity = self.sample.intensity.values
        if self.transform:
            wl, intensity = self.transform(wl, intensity)
        mask = (wl > self.wl_range[0]) & (wl < self.wl_range[1])
        wl = wl[mask]
        intensity = intensity[mask]
        if normalized:
            mirror_int = self.mirror.intensity.values
            intensity /= mirror_int[mask]
        return wl, intensity

    @staticmethod
    def get_color(wl, intensity, plot=False):
        a = np.stack([wl, intensity], axis=-1)
        col = color(a)
        if plot:
            col.plot_rgb()
            col.plot_cie()
        return col

    def interpolate(self):
        f = interp1d(self.wl, self.intensity)
        return f

    @staticmethod
    def show_calculated_color(RGB):
        circle=plt.Circle((0.5, 0.5), radius=0.5, facecolor=tuple(RGB))
        plt.gcf().gca().add_artist(circle)
        plt.axis('off')

    def show_spectra(self):
        plt.plot(self.wl,self.intensity)
        plt.xlim(*self.wl_range)
        plt.ylim(0,1.1)
        plt.xlabel('wl, nm')
        plt.ylabel('Reflection')

    def show(self):
        plt.figure(figsize=[10,5])
        plt.subplot(121)
        self.show_calculated_color(self.color.RGB)
        plt.subplot(122)
        self.show_spectra()
        plt.show()

class SpectraTheory():
    def __init__(self, thickness,
                 wl_range=[400,750],
                 mats_dir='mats/',
                 si_file='Si.txt',
                 pmma_file='PMMA.txt',
                 plot_color=False,
                 bottom_material='si'):
        self.wl_range=wl_range
        data_PMMA = np.loadtxt(os.path.join(mats_dir, pmma_file))
        self.bottom_n, self.bottom_k = self.get_bottom_refr_index(material_type=bottom_material,filename=os.path.join(mats_dir,si_file))
        wl, n = data_PMMA[:, 0], data_PMMA[:, 1]
        self.material_n = interp1d(wl, n)
        # if non-dispersive material
        if data_PMMA.shape[1] < 3:
            self.material_k = lambda x: 0
        else:
            k = data_PMMA[:, 2]
            self.material_k = interp1d(wl, k)
        mask = (wl > self.wl_range[0]) & (wl < self.wl_range[1])
        self.wl = wl[mask]
        self.intensity = np.array([self.get_refl(wl, thickness) for wl in self.wl])
        self.f = self.interpolate()
        self.color = Spectra.get_color(self.wl, self.intensity,
                                       plot=plot_color)

    @staticmethod
    def get_bottom_refr_index(material_type, filename=None):
        if material_type == 'si':
            data_Si = np.loadtxt(filename)
            wl, n, k = data_Si[:, 0], data_Si[:, 1], data_Si[:, 2]
            bottom_n = interp1d(wl, n)
            bottom_k = interp1d(wl, k)
        elif material_type == 'mirror':
            bottom_n = (lambda x: 1e6)
            bottom_k = (lambda x: 0)
        elif material_type == 'transparent':
            bottom_n = (lambda x: 1)
            bottom_k = (lambda x: 0)
        else:
            raise ValueError("Unknown bottom material type")
        return bottom_n, bottom_k

    @staticmethod
    def get_color(wl, intensity, plot=False):
        a = np.stack([wl, intensity], axis=-1)
        col = color(a)
        if plot:
            col.plot_rgb()
            col.plot_cie()
        return col

    def interpolate(self):
        f = interp1d(self.wl, self.intensity)
        return f

    @staticmethod
    def show_calculated_color(RGB):
        circle=plt.Circle((0.5, 0.5), radius=0.5, facecolor=tuple(RGB))
        plt.gcf().gca().add_artist(circle)
        plt.axis('off')

    def show_spectra(self):
        plt.plot(self.wl,self.intensity)
        plt.xlim(*self.wl_range)
        plt.ylim(0,1.1)
        plt.xlabel('wl, nm')
        plt.ylabel('Reflection')

    def show(self):
        plt.figure(figsize=[10,5])
        plt.subplot(121)
        self.show_calculated_color(self.color.RGB)
        plt.subplot(122)
        self.show_spectra()
        plt.show()

    def get_refl(self, wl0, d):
        nk_list = [1, self.material_n(wl0) + 1j * self.material_k(wl0), self.bottom_n(wl0) + 1j * self.bottom_k(wl0), 1]
        d_list = [np.inf, d, 20000, np.inf]
        return coh_tmm('s', nk_list, d_list, 0, wl0)['R']

class SyntheticSpectra(SpectraTheory):
    def __init__(self, thickness, refr_index,
                 wl_range=[400,750],
                 mats_dir='mats/',
                 si_file='Si.txt',
                 plot_color=False,
                 camera = None,
                 bottom_material = 'si'):
        self.wl_range=wl_range
        self.thickness = thickness
        self.bottom_n, self.bottom_k = self.get_bottom_refr_index(material_type=bottom_material,
                                                                  filename=os.path.join(mats_dir, si_file))
        self.wl =np.linspace(wl_range[0],wl_range[1],100)
        self.refr_index = refr_index
        self.material_k = lambda x: 0
        self.intensity = np.array([self.get_refl(wl, thickness) for wl in self.wl])
        self.f = self.interpolate()
        self.color = self.get_color(self.wl, self.intensity,
                                       plot=plot_color, camera=camera)

    def material_n(self, x):
        return np.ones_like(x)*self.refr_index

    def get_color(self, wl, intensity, plot=False, camera=None):
        """
        get color from camera CMF if defined, or human-eye CMF from parent class
        :param wl:
        :param intensity:
        :param plot:
        :param camera:
        :return:
        """
        if camera is None:
            color = super().get_color(wl,intensity,plot)
        else:
            color = self.get_color_from_camera(wl,intensity,camera)
        return color

    @staticmethod
    def get_color_from_camera(wl, intensity, camera):
        RGB = camera.spectra_to_XYZ(wl,intensity)
        color = AttributeDict()
        color.RGB = RGB
        return color

class Image():
    def __init__(self, fname, regime='active', th=1e-1, fast=False):
        self.image = io.imread(fname)[:, :, :3]
        self.th = th
        self.fast = fast
        self.grayscale = rgb2gray(self.image)
        self.mask = self.grayscale > th
        self.estimate_rgb(regime)

    def estimate_rgb(self, regime):
        if regime == 'active':
            image = self.get_active_area()
        elif regime == 'selective':
            image = self.select_areas()
        else:
            raise ValueError('Unknown regime!')
        self.color = image.mean(axis=0)
        if np.max(self.color) > 1:
            self.color /= 255.
        self.RGB = self.color
        self.XYZ = colour.sRGB_to_XYZ(self.RGB)

    def get_active_area(self):
        if not self.fast:
            image = self.image.copy()
        else:
            image = self.image
        image[~self.mask] = [0, 0, 0]
        return image[self.mask]

    def select_areas(self, crop_frac=0.5):
        if not self.fast:
            image = self.image.copy()
        else:
            image = self.image
        wx, wy, _ = image.shape
        image = image[int(wx*crop_frac)//2:-int(wx*crop_frac)//2,int(wy*crop_frac)//2:-int(wy*crop_frac)//2,:]
        self.grayscale = rgb2gray(image)
        self.mask = self.grayscale > self.th
        return image[self.mask]

    def show(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        self.show_estimated_color()
        plt.subplot(122)
        self.show_true_image()
        plt.show()

    def show_estimated_color(self):
        circle = plt.Circle((0.5, 0.5), radius=0.5, facecolor=self.color)
        plt.gcf().gca().add_artist(circle)
        plt.axis('off')

    def show_true_image(self):
        io.imshow(self.image)
        plt.axis('off')

class Sample():
    def __init__(self,
                 sample_file,
                 mirror_file='Mirror',
                 spectra_ext='.csv',
                 image_ext='.tiff',
                 folder='data/',
                 only_spectra=False,
                 thick_file=None,
                 spectra_kargs={},
                 image_kargs={}
                 ):
        folder = os.path.abspath(folder)
        self.sample_file = os.path.join(folder,sample_file)
        self.mirror_file = os.path.join(folder, mirror_file)
        self.filename = sample_file
        self.spectra = Spectra(self.sample_file+spectra_ext,
                               self.mirror_file+spectra_ext,
                               **spectra_kargs)
        if not only_spectra:
            self.image = Image(self.sample_file+image_ext, **image_kargs)
        if thick_file:
            with open(os.path.join(folder,thick_file),'rb') as file:
                thicknesses = pickle.load(file)
                self.thickness=thicknesses[sample_file]


    def show(self):
        plt.figure(figsize=[10,5])
        plt.subplot(121)
        plt.title('Spectral RGB')
        self.spectra.show_calculated_color(self.spectra.color.RGB)
        plt.subplot(122)
        plt.title('Image RGB')
        self.image.show_estimated_color()
        plt.show()

class SampleArray():
    def __init__(self, X, Y, samples, thicknesses=None):
        self.X = X
        self.Y = Y
        self.n_materials = len(np.unique(Y))
        self.samples = np.array(samples)
        if not thicknesses:
            thicknesses = self.get_thicknesses()
        self.thicknesses = thicknesses
        self.colors = self.get_colors()
        self.material_ns = self.get_material_ns()

    def get_thicknesses(self):
        thicknesses = []
        for sample in self.samples:
            thicknesses.append(sample.thickness)
        return np.array(thicknesses)

    def get_colors(self):
        colors = []
        for sample in self.samples:
            colors.append(sample.color.RGB)
        return np.array(colors)

    def get_material_ns(self):
        material_ns = []
        for sample in self.samples:
            material_ns.append(sample.refr_index)
        return np.array(material_ns)

    def show(self):
        plt.figure(figsize=[10, 10])
        plt.subplot(221)
        self.show_classes()
        plt.subplot(222)
        self.show_thicknesses()
        plt.subplot(223)
        self.show_samples_with_colors()
        plt.subplot(224)
        self.show_refractive_indices()
        plt.show()

    def show_classes(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], marker='o', c=self.Y,
                    s=50, edgecolor='k')
        plt.axis('off')

    def show_samples_with_colors(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], marker='o', c=self.colors,
                    s=50, edgecolor='k')
        plt.axis('off')

    def show_color_space(self, labels=None):
        if labels is None:
            labels = self.Y
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.colors[:, 0], self.colors[:, 1], self.colors[:, 2], marker='o', c=labels,
                    s=50, edgecolor='k')
        plt.show()

    def show_thicknesses(self):
        for i in range(self.n_materials):
            idx = self.Y == i
            plt.hist(self.thicknesses[idx], color=mpl_colors[i], alpha=0.5, label='material %d' % i)
        plt.legend()
        plt.xlabel('thicknesses, nm')

    def show_refractive_indices(self, wl_range=[400,750]):
        wl = np.linspace(*wl_range,50)
        for i in range(self.n_materials):
            idx = self.Y == i
            plt.hist(self.material_ns[idx], color=mpl_colors[i], alpha=0.5, label='material %d' % i)
        plt.xlabel('wavelength, nm')
        plt.ylabel('n')
        plt.legend()

    def __str__(self):
        pass

    @classmethod
    def generate_random_samples(cls, refractive_indices, mus, sigmas=20, refractive_sigmas=0, n_samples=50, camera=None, **kargs):
        """
        :param refractive_indices: refractive indices of a material
        :param mus: central thicknesses of a material [nm]
        :param sigma: sigma for thickness distributions [nm]
        :param kargs: make_blobs arguments
        :return:
        """
        nclusters = len(refractive_indices)
        X1, Y1 = make_blobs(n_features=2, centers=nclusters, n_samples=n_samples, **kargs)
        nsamples = len(Y1)
        X1 = (X1 - np.min(X1, axis=0))
        X1 /= np.max(X1, axis=0)
        samples = []
        #same sigma in thickness distribution
        if not isinstance(sigmas,Iterable):
            sigmas = [sigmas]*nclusters
        if not isinstance(refractive_sigmas,Iterable):
            refractive_sigmas = [refractive_sigmas]*nclusters
        for i in range(nsamples):
            idx = Y1[i]
            mu = mus[idx]
            n = np.random.normal(refractive_indices[idx],refractive_sigmas[idx])
            sigma = sigmas[idx]
            thickness = np.random.normal(mu,sigma)
            sample = SyntheticSpectra(thickness=thickness, refr_index=n, camera=camera)
            samples.append(sample)
        return cls(X1,Y1,samples)

    @classmethod
    def generate_grid_samples(cls, refractive_indices, mus, sigmas=20, refractive_sigmas=0, n_samples_per_row=10, camera=None, bottom_material='si', **kargs):
        """
        :param refractive_indices: refractive indices of a material
        :param mus: central thicknesses of a material [nm]
        :param sigma: sigma for thickness distributions [nm]
        :param kargs: make_blobs arguments
        :return:
        """
        nclusters = len(refractive_indices)
        _, Y1 = make_blobs(n_features=2, centers=nclusters, n_samples=n_samples_per_row**2, **kargs)
        X=np.linspace(0,1,n_samples_per_row)
        X, Y = np.meshgrid(X, X)
        X1 = np.stack([X, Y], axis=-1).reshape(-1,2)
        nsamples = len(Y1)
        samples = []
        #same sigma in thickness distribution
        if not isinstance(sigmas,Iterable):
            sigmas = [sigmas]*nclusters
        if not isinstance(refractive_sigmas,Iterable):
            refractive_sigmas = [refractive_sigmas]*nclusters
        for i in range(nsamples):
            idx = Y1[i]
            mu = mus[idx]
            n = np.random.normal(refractive_indices[idx],refractive_sigmas[idx])
            sigma = sigmas[idx]
            thickness = np.random.normal(mu,sigma)
            sample = SyntheticSpectra(thickness=thickness, refr_index=n, camera=camera, bottom_material=bottom_material)
            samples.append(sample)
        return cls(X1,Y1,samples)

    @classmethod
    def load(cls,fname='samples.pkl'):
        with open(fname,'rb') as file:
            samples = pickle.load(file)
        return samples

    def save(self,fname='samples.pkl'):
        with open(fname,'wb') as file:
            pickle.dump(self,file)

class ImageSampleArray:
    def __init__(self, fname, crop=None):
        self.image = io.imread(fname)[:, :, :3]

    def show_color_space(self, labels=None):
        if labels is None:
            labels = self.Y
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.colors[:, 0], self.colors[:, 1], self.colors[:, 2], marker='o', c=labels,
                    s=50, edgecolor='k')
        plt.show()

if __name__ == '__main__':
    datadir = 'data/ocean_ccd2/'
    samplefile = 'A14'
    im = Image(datadir + samplefile  + 'ld' + '.tiff', regime='selective', fast=True)
    im.show()
    sp = Spectra(datadir + samplefile+ '_095' + '.csv', datadir + 'mirror_calib' + '.csv')
    sp.show()
    exit(0)
    ###################################
