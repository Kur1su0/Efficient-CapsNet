"""
load mhd data from a path
"""
from matplotlib.pyplot import plot
import numpy as np
from numpy.core.records import array
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

PATH_TESTING = "dataset/Testing/"
PATH_TRAINING = "dataset/Training/"

def load_mhd(path):
    """
    load mhd data from a path
    """
    ct_scan = sitk.GetArrayFromImage(sitk.ReadImage(path, sitk.sitkFloat32))

    return ct_scan[0] if ct_scan.shape == (1, 256, 256) else ct_scan 
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
def load_all_mhd(disease_mhd_path):
    x = []
    for each_file in disease_mhd_path:
        ret = load_mhd(each_file)
        ret = cv2.resize(ret, (28,28))
        x.append(ret)
        # plot_mhd(ret)
    return np.array(x)

#_class : 0 healthy, 1 disease.

def process_mhd(path):
    disease_mhd_path = find_mhd(path + "Diseased/")
    healthy_mhd_path = find_mhd(path + "Healthy/")
    # print(len(disease_mhd_path) + len(healthy_mhd_path))
    #process X
    X_disease = load_all_mhd(disease_mhd_path)
    X_healthy = load_all_mhd(healthy_mhd_path)
    X = np.concatenate((X_disease, X_healthy), axis=0)
    # print(X.shape)
    #process label
    y_disease = np.zeros(len(disease_mhd_path))
    y_healthy = np.ones(len(healthy_mhd_path))
    # print("y_disease: ", y_disease.shape)
    # print("y_healthy: ", y_healthy.shape)
    y = np.concatenate((y_disease, y_healthy), axis=0)

    #shuffle data
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    return X, y

def process_all_mhd():
    image_path = [PATH_TESTING, PATH_TRAINING]
    X_test, y_test = process_mhd(PATH_TESTING)
    X_train, y_train = process_mhd(PATH_TRAINING)

    return (X_train, y_train),(X_test, y_test)
    
    


if __name__ == "__main__":
    process_all_mhd()
    #plot_mhd(ret)
    

