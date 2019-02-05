import matplotlib.pyplot as plt
import numpy as np
import pickle

def str_to_bool(bool_str):
    if bool_str == 'True':
        return True
    elif bool_str == 'False':
        return False
    else:
        return ValueError

def save_file(obj, fpath):
    with open(fpath, 'wb') as file:
        pickle.dump(obj, file)

def load_file(fpath):
    with open(fpath, 'rb') as file:
        return pickle.load(file)

def plot_img(img):
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)