import signal

import numpy as np
import mne


def butter_bandpass_filter(data):
    fs_Hz = 250.0
    bp2_stop_Hz = np.array([49, 51.0])
    b2, a2 = signal.butter(2, bp2_stop_Hz / (fs_Hz / 2.0), 'bandstop')
    y = signal.lfilter(b2, a2, data)
    return y

def split_data_evenly(data,split,labels,sample_number=0):
    epochs_data = []
    i = 0
    epochs_raw = np.asarray(np.array_split(data, split, axis=1))
    labels = np.asarray(np.array_split(labels, split, axis=1))
    if np.asarray(epochs_data).size == 0:
        epochs_data = epochs_raw
    else:
        epochs_data = np.concatenate((np.asarray(epochs_data), epochs_raw))
    return epochs_data, compact_labels(labels)

def compact_labels(labels):
    for n,i in enumerate(labels):
        labels[n] = sum(i)/len(i)
    return labels

def parse_file_without_events(filename, channel_number, user_id=0):
    data_path = r'C:\Users\innopolis\Desktop\Intership BCI\OpenBCI\OpenBCI_Python-master\data\\'
    data = np.loadtxt(data_path + filename, delimiter=',')
    return data

def parse_files_without_events(filenames,channel_number,split_function,channel_names=None):
    if not channel_names:
        channel_names = [str(x) for x in range(channel_number)]
    if channel_number != len(channel_names):
        raise "Channel number should be equal to number of channel names"

    data = []
    for id,file in enumerate(filenames):
        x = parse_file_without_events(file,channel_number,id)
        info = mne.create_info(['CH1', 'CH2'], 250, 'eeg')
        info['subject_info'] = id
        # labels = np.concatenate((np.asarray(labels), z))
        x = np.transpose(x)
        x = mne.io.RawArray(x, info)
        data.append(x)

    return data



def parse_files_with_events(filenames,channel_number):

    def parse_file_with_events(filename, channel_number, user_id=0):
        data = []
        events = []
        f = open(filename, "r")
        lines = f.readlines()
        i = 1
        labels = []
        for row in lines:
            list = eval(row)
            if list[1] == 'blink':
                events = events[:-1]
                events.append(1)
            else:
                events.append(0)
                x = []
                for j in channel_number:
                    x.append(list[j + 2])
                data.append(x)
            labels.append(user_id)
            i = i + 1
        for j in range(len(data)):
            data[j] = butter_bandpass_filter(data[j])  # apply filtering
        data = np.transpose(np.asarray(data))
        return data, events, labels

    data_path = r'C:\Users\innopolis\Desktop\Intership BCI\OpenBCI\OpenBCI_Python-master\data\\'
    epochs_data = []
    labels = []
    data = []
    events = []
    labels = []
    id = 0
    for id, file in enumerate(filenames):
        x,y,z = parse_file_with_events(file,channel_number,id)
        data = np.concatenate((np.asarray(data), x))
        events = np.concatenate((np.asarray(events), y))
        labels = np.concatenate((np.asarray(labels), z))
    return data,events,labels

