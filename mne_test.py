import numpy as np
import mne
from mne.decoding import CSP

def f(x):
    return  4.5/x/(2^23 - 1)
count2volts = np.vectorize(f)
data_path = 'C:\Users\innopolis\Desktop\Intership BCI\OpenBCI\OpenBCI_Python-master\data'
data_Vitaly = np.loadtxt(data_path+'Vitaly_ECG_v2,v3.txt',delimiter=',')
data_Vitaly = np.transpose(data_Vitaly)
data_Bulat = np.loadtxt(data_path+'Bulat_ECG_v2,v3.txt',delimiter=',')
data_Bulat = np.transpose(data_Bulat)
ch_names = ['V3','V2']


epochs_raw = np.asarray(np.array_split(data,10,axis=1))

sfreq = 250

info = mne.create_info(ch_names,sfreq,ch_types="eeg")
print(len(epochs_raw[0][1]))
raw = mne.io.RawArray(data,info)
Epochs = mne.EpochsArray(epochs_raw,info)


from sklearn.lda import LDA  # noqa
from sklearn.cross_validation import ShuffleSplit  # noqa

input()