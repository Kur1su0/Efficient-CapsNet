"""
load mhd data from a path
"""
import numpy as np
import SimpleITK as sitk
import os
import cv2
import matplotlib.pylab as plt
# ct_scans = sitk.GetArrayFromImage(sitk.ReadImage("training_001_ct.mhd", sitk.sitkFloat32))

PATH_TESTING_D = "dataset/Testing/Diseased/"
PATH_TESTING_H = "dataset/Testing/Healthy/"
#path of training data
PATH_TRAINING_D = "dataset/Training/Diseased/"
PATH_TRAINING_H = "dataset/Training/Healthy/"


def load_mhd(path):
    """
    load mhd data from a path
    """
    ct_scan = sitk.GetArrayFromImage(sitk.ReadImage(path, sitk.sitkFloat32))
    return ct_scan
def plot_mhd(ct_scan):
    #subpliot 2 image in one figure
    plt.imshow(ct_scan)
def find_mhd(path):
    """
    find all mhd file in a path
    """
    ret = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".mhd"):
                ret.append(os.path.join(root, file))
    return ret


def process_img():
    all_mhd_path = find_mhd(PATH_TESTING_D)
    print(all_mhd_path)
    ret = load_mhd(all_mhd_path[-1])
    ret = cv2.resize(ret, (28,28))
    print(ret)


if __name__ == "__main__":
    process_img()
    #plot_mhd(ret)
    
plt.show()
