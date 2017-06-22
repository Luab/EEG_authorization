import numpy as np
import mne

data = np.loadtxt('row2.csv',delimiter=',')
data = np.transpose(data)
print(data[1][1])
ch_names = ['CH1','CH2']

sfreq = 250

info = mne.create_info(ch_names,sfreq)

raw = mne.io.RawArray(data,info)
events = mne.find_events(raw, stim_channel='CH1')
baseline = (None, 0)  # means from the first instant to t = 0
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=baseline, preload=False)

raw.plot()