import random
from concurrent.futures import ThreadPoolExecutor
import uuid
import cv2, os
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from PIL import Image, ImageFilter
from sklearn.decomposition import PCA
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
from sklearn.neighbors import LocalOutlierFactor

from LeafPalm.palm_grow import PalmGrow
from LeafPalm.filter import Inter_class_filter
import pickle


EPS = 1e-2


def process_xy(x: np.ndarray, y: np.ndarray):
    x = (x - 0.5) * 2.0
    y = (-y + 0.5) * 2.0
    return x, y

def inverse_xy(x: np.ndarray, y: np.ndarray):
    x = (x / 2.0) + 0.5
    y = -(y / 2.0) + 0.5
    return x, y

class PolyPalmCreator:
    def __init__(self, imsize: int, degree: int = 4, noise_weight: float = 0.2,
                 draw_thickness: int = 25, is_branch: bool = False, is_prolong: bool = False,
                 curve_1_is_cut: bool = False, curve_2_is_cut: bool = False) -> None:

        self.imsize = imsize
        self.degree = degree
        self.draw_thickness = draw_thickness  # 多项式绘制线条粗细
        self.is_branch = is_branch
        self.is_prolong = is_prolong
        self.curve_1_is_cut = curve_1_is_cut
        self.curve_2_is_cut = curve_2_is_cut
        self.noise_weight = noise_weight
    
    def fit(self, path: str, filename: str) -> None:

        with open(path, 'rb') as f:
            data_list = pickle.load(f)
        
        coeff_list, x_axis_list = [], []
        for data in data_list:
            # for each labeled images,
            # calculate the polynomial coefficients and the x axis range                
            coeff, x_axis = [], []
            for line in data:
                line = np.array(line)
                x, y = line[::2], line[1::2]
                x, y = process_xy(x, y)
                c = np.polyfit(x, y, deg=self.degree)
                coeff.append(c[None, :])
                x_axis.append([x.min(), x.max()])
            
            coeff_list.append(np.vstack(coeff)[None, :])
            x_axis_list.append(np.vstack(x_axis)[None, :])

        self.coeff = np.vstack(coeff_list)
        self.x_axis = np.vstack(x_axis_list)
        self.fdim = self.coeff.shape[-1]

        # filter the outlier out of 0.75 and 0.25
        self.filter_outliers_by_iqr()
        # calculate the convariance, mean, std of the coeff
        self.process_coeff()
        # calculate the mean and std of the start and end point
        self.process_axis()
        data = {
            "coeff0": self.coeff[0],
            "coeff1": self.coeff[1],
            "coeff2": self.coeff[2],
            "x_axis": self.x_axis,
            "fdim": self.fdim,
            "coeff_covariance": self.coeff_covariance,
            "coeff_mean": self.coeff_mean,
            "coeff_std": self.coeff_std,
            "x_axis_ms0": self.x_axis_ms[0],
            "x_axis_ms1": self.x_axis_ms[1],
            "x_axis_ms2": self.x_axis_ms[2],
        }
        np.savez(filename, **data)

    def set_seed(self, seed):
        np.random.seed(seed=seed)

    def filter_outliers_by_iqr(self):
        """
        Filter the outliers by the percentile
        """
        results = []
        for l in range(3):
            data = self.coeff[:, l]
            q1 = np.percentile(data, 25, axis=0)  
            q3 = np.percentile(data, 75, axis=0)  
            iqr = q3 - q1  
            lower_bound = q1 - 1.5 * iqr  
            upper_bound = q3 + 1.5 * iqr  
            inner = [x for x in data if all(x > lower_bound) and all(x < upper_bound)]
            results += [np.array(inner)]
        self.coeff = results

    def process_coeff(self):
        """
        Calculate the covariance, mean, and std
        """
        self.coeff_mean = np.ones(shape=(3, self.fdim))
        self.coeff_std = np.ones_like(self.coeff_mean)
        self.coeff_covariance = np.ones(shape=(3, self.fdim, self.fdim))

        for l in range(3):
            multi = self.coeff[l]
            multi = multi.T
            self.coeff_covariance[l] = np.cov(multi)
            self.coeff_mean[l] = np.mean(self.coeff[l], axis=0)
            self.coeff_std[l] = np.std(self.coeff[l], axis=0)

    def process_axis(self):
        """
        Calculate the mean and std of start and end point of three curve
        """
        self.x_axis_ms = []
        # line 1
        data = self.x_axis[:, 0]
        mean = [-1, np.mean(data[:, 1])]
        std = [np.mean(data[:, 0] + 1) * math.sqrt(math.pi / 2.0) * 0.0, np.std(data[:, 1])]
        self.x_axis_ms.append([mean, std])
        # line 2
        data = self.x_axis[:, 1]
        mean = [-1, np.mean(data[:, 1])]
        std = [np.mean(data[:, 0] + 1) * math.sqrt(math.pi / 2.0) * 0.0, np.std(data[:, 1])]
        self.x_axis_ms.append([mean, std])
        # line 3
        data = self.x_axis[:, 2]
        mean = [np.mean(data[:, 0]), 1]
        std = [np.std(data[:, 0]), np.mean(data[:, 1] - 1) * math.sqrt(math.pi / 2.0) * -1]
        self.x_axis_ms.append([mean, std])

    def draw_by_multi(self, filename_img, filename_sample, scale=1.0, bgname=None):
        feature = []
        for i in range(3):
            covariance = self.coeff_covariance[i] * scale
            mean = self.coeff_mean[i]
            res = np.random.multivariate_normal(mean, covariance, 1)
            feature.append(res.flatten().tolist())
        border = []
        for i, f in enumerate(feature):
            x = np.random.randn(2) * self.x_axis_ms[i][1] + self.x_axis_ms[i][0] 
            border.append(x)

        image, curve_set, secondary_line = self.get_LeafPalm(feature, border)

        data = {
            "feature": feature,
            "border": border,
            "curve_set": curve_set,
            "secondary_line": secondary_line
        }
        np.savez(filename_sample, **data)
        image = cv2.resize(image, dsize=(self.imsize,)*2)
        cv2.imwrite(filename_img, image)

    def get_LeafPalm(self, feature, border):
        image, curve_set, secondary_line = PalmGrow(feature, border, self.draw_thickness, self.noise_weight,
                         self.is_branch, self.is_prolong,
                         self.curve_1_is_cut, self.curve_2_is_cut)
        return image, curve_set, secondary_line


def test():
    fig = PolyPalmCreator(imsize=512)
    filepath = "/home/ra1/Project/ZZY/Diff-Palm/PolyCreases/labeled_data.pkl"
    os.makedirs('/home/ra1/Project/ZZY/Diff-Palm/PolyCreases/coeff_test', exist_ok=True)
    fig.fit(filepath, f'/home/ra1/Project/ZZY/Diff-Palm/PolyCreases/coeff_test/fit.npz')

    for i in range(0, 200):
        name_img = f'/home/ra1/Project/ZZY/Diff-Palm/PolyCreases/coeff_test/image/{i}.png'
        name_sample = f'/home/ra1/Project/ZZY/Diff-Palm/PolyCreases/coeff_test/polynominal/{i}.npz'
        fig.draw_by_multi(filename_img=name_img, filename_sample=name_sample)
        print(f'{name_img} and polynominal/{i}.npz are saved!!!')
        

if __name__ == '__main__':
    test()
