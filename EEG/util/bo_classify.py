import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scipy.signal import butter, filtfilt
import numpy as np
import scipy.io as scio
import torch

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def get_abtd(dataFile):
    # print(f"\n\n{os.path.dirname(__file__)}\n")
    data = scio.loadmat(dataFile)
    #print(data.keys())
    #print(data['temp'][1:,:])
    data1 = np.sin(data['temp'])
    data2 = np.sin(data['temp'])
    data3 = np.sin(data['temp'])
    data4 = np.sin(data['temp'])

    fs = 1000
    freq = 100
    arr_set = []
    split_set = [i*freq for i in range(data1.shape[1]//freq)]
    # print(split_set)

    for i in range(1,len(split_set)):
        a = butter_bandpass_filter(data1[:,split_set[i-1]:split_set[i]], 8, 13, fs, 5)#
        a = np.mean(np.abs(a))
        b= butter_bandpass_filter(data2[:,split_set[i-1]:split_set[i]], 14, 25, fs, 5)
        b = np.mean(np.abs(b))
        t= butter_bandpass_filter(data3[:,split_set[i-1]:split_set[i]], 4, 7, fs, 1)
        t = np.mean(np.abs(t))
        d= butter_bandpass_filter(data4[:,split_set[i-1]:split_set[i]], 0.5, 4, fs, 1)
        d = np.mean(np.abs(d))
        arr = np.array([a, b, t, d])
        # print(i, arr)
        arr_set.append(arr)

    arr_set = torch.from_numpy(np.array(arr_set)) # shape = tensor(129,4)
    # import pdb; pdb.set_trace()

    return arr_set

if __name__ == "__main__":
    dataFile = '../data/dataOut4.mat'
    get_abtd(dataFile)