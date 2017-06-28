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

def parse_files_without_events(filenames,channel_number,split_function):

    def parse_file_without_events(filename, channel_number, user_id=0):
        data = np.loadtxt(data_path + file, delimiter=',')
        return data

    data_path = r'C:\Users\innopolis\Desktop\Intership BCI\OpenBCI\OpenBCI_Python-master\data\\'
    epochs_data = []
    data = []
    events = []
    labels = []
    id = 0
    info = []
    for id,file in enumerate(filenames):
        x = parse_file_without_events(file,channel_number,id)
        if np.asarray(data).size == 0:
            data = x
        else:
            data = np.concatenate((np.asarray(data), x))
        info = mne.create_info(['CH1', 'CH2'], 250, 'eeg')
        info['subject_info']['id'] = id
        # labels = np.concatenate((np.asarray(labels), z))




    data = mne.io.RawArray(data, info)
    data.add_events(events, 'I do not understand what I am doing')
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

