from scipy.signal import butter, filtfilt
import numpy as np
import scipy.io as scio


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
    data = scio.loadmat(dataFile)
    #print(data.keys())
    #print(data['temp'][1:,:])
    data1 = np.sin(data['temp'])
    data2 = np.sin(data['temp'])
    data3 = np.sin(data['temp'])
    data4 = np.sin(data['temp'])

    fs = 1000


    a = butter_bandpass_filter(data1, 8, 13, fs, 5)#
    a = np.mean(np.abs(a))
    b= butter_bandpass_filter(data2, 14, 25, fs, 5)
    b = np.mean(np.abs(b))
    t= butter_bandpass_filter(data3, 4, 7, fs, 1)
    t = np.mean(np.abs(t))
    d= butter_bandpass_filter(data4, 0.5, 4, fs, 1)
    d = np.mean(np.abs(d))
    arr = np.array([a, b, t, d])
    print(arr)

    return arr

if __name__ == "__main__":
    dataFile = '../data/dataOut4.mat'
    get_abtd(dataFile)