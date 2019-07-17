import numpy as np
np.random.seed(7)
import pandas as pd
import h5py
from sklearn.utils import shuffle
from sklearn import preprocessing

def get_data(filename):
    """
        This just takes the data and puts it in the shape and format that I would like to deal with.
        The data is converted to an hdf5 file using kratsg's gML file on github: https://github.com/kratsg/gML.git
    """
    data = h5py.File(filename, 'r')
    # 28 eta rows and 32 phi columns
    gTowerEt = data['gTowerEt'][:].reshape(-1, 28, 32)
    return (gTowerEt)

def flatten_data(data, second_dim):
    """
        Flatten data from 3 Dimensions to 2 Dimensions
        data: 3D input
        second_dim: what we want to flatten the input into
    """
    flat_array = data.reshape(-1, second_dim)
    return flat_array

def add_ones_or_zeros(data_frame, one_or_zero):
    """
        data_frame: should be a pandas Data Frame

        add ones or zeros to data depending on whether it is signal or background
    """
    if one_or_zero == 1:
        label = np.ones(len(data_frame))
        label_series = pd.Series(label)


    elif one_or_zero == 0:
        label = np.zeros(len(data_frame))
        label_series = pd.Series(label)

    else:
        print("Error: Not a 1 or 0")
        return None

    data_frame["832"] = label_series
    return data_frame


def images_and_labels():
    """
        This function takes no arguments since it has been custom build for this dataset.

        However, if you want to use your own dataset I recommend that you follow a similar method that I employed in the data visualization notebook to remove null rows. You will probably also have to customize Giordon's notebook (link in the get_data function's docstring)

    """
    gTowerEt_background1 = get_data("JZ0W_hdf5/user._010556.JZ0W.hdf5")
    gTowerEt_background2 = get_data("JZ0W_hdf5/user._010590.JZ0W.hdf5")
    gTowerEt_background3 = get_data("JZ0W_hdf5/user._010592.JZ0W.hdf5")
    gTowerEt_background4 = get_data("JZ0W_hdf5/user._010594.JZ0W1.hdf5")
    gTowerEt_background = np.concatenate([gTowerEt_background1
                                    ,gTowerEt_background2
                                    ,gTowerEt_background3
                                    ,gTowerEt_background4])

    gTowerEt_signal1 = get_data("ZvvHbb_hdf5/user._000118.ZvvHbb.hdf5")
    gTowerEt_signal2 = get_data("ZvvHbb_hdf5/user._000123.ZvvHbb.hdf5")
    gTowerEt_signal3 = get_data("ZvvHbb_hdf5/user._000130.ZvvHbb.hdf5")
    gTowerEt_signal4 = get_data("ZvvHbb_hdf5/user._000139.ZvvHbb.hdf5")
    gTowerEt_signal = np.concatenate([gTowerEt_signal1
                                    ,gTowerEt_signal2
                                    ,gTowerEt_signal3
                                    ,gTowerEt_signal4])


    #flatten the datasets
    gTower_signal_flat = flatten_data(gTowerEt_signal, 28*32)
    gTower_background_flat = flatten_data(gTowerEt_background, 28*32)

    # convert to a pandas data frame
    df_signal_flat = pd.DataFrame(gTower_signal_flat)
    df_background_flat = pd.DataFrame(gTower_background_flat)

    #drop zero columns
    df_signal_flat.drop(df_signal_flat.columns[832:], axis=1, inplace=True)
    #drop zero columns
    df_background_flat.drop(df_background_flat.columns[832:], axis=1, inplace=True)

    #add labels to the data frames
    df_signal_with_label = add_ones_or_zeros(df_signal_flat, 1)
    df_background_with_label = add_ones_or_zeros(df_background_flat, 0)

    #concatenate the two datasets and shuffle them up
    df = pd.concat([df_background_with_label, df_signal_with_label])
    df = shuffle(df)

    #take just the labels and convert back to a numpy array
    labels = df['832']
    labels = np.array(labels)

    #take just the image data, convert to a numpy array, and then reshape for training
    df.drop(df.columns[832], axis=1, inplace=True)
    images = np.array(df)

    images = images[:].reshape(-1, 26, 32, 1)
     
    
    return images, labels
