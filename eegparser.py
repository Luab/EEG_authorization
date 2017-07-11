from scipy import signal

import numpy as np
import mne

def parse_from_alcoholic_dataset(directory):
    from numpy import inf,NaN
    def filter_nan_and_infinite(array):
        array = list(filter(lambda x: x != float('-inf'), array))
        array[array == inf] = 0.0
        array[array == NaN] = 0.0
        return array
    #TODO USERS 108 and 116 are bad. Drop them
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    result_data = [[] for i in range(122)]
    labels = []
    i = 0
    print(len(onlyfiles))
    print(onlyfiles[0].split(".")[0])
    subj = onlyfiles[0].split(".")[0]
    this_subj = onlyfiles[0].split(".")[0]
    for f in onlyfiles:
        subj = f.split(".")[0]
        subject_data = [[],[]]
        with open(directory+r"\\"+f) as opened:
            EC = False
            for line in opened:
                line = line.split(' ')
                if line[0] != "#":
                    if line[1] == "FP1":
                        if float(line[3]) == -np.inf or float(line[3]) == np.inf or float(line[3]) == np.NaN:
                            pass
                        else:
                            subject_data[0].append(float(line[3]))
                    if line[1] == "FP2":
                        if float(line[3]) == -inf or float(line[3]) == inf or float(line[3]) == NaN:
                            pass
                        else:
                            subject_data[1].append(float(line[3]))
                    if line[1] == "F7":
                        if float(line[3]) == -inf or float(line[3]) == inf or float(line[3]) == NaN:
                            pass
                        else:
                            pass
                           #subject_data[2].append(float(line[3]))
            for id,channel in enumerate(subject_data):
                if len(channel) == 0 or sum(channel) == 0 or np.any(np.isnan(channel)) or not np.all(np.isfinite(channel)):
                    EC = True
                else:
                    subject_data[id] = filter_nan_and_infinite(channel)
            if not EC:
                result_data[i].append(subject_data)

            if subj == this_subj:
                if not EC:
                    labels.append(i)
            else:
                this_subj = f.split(".")[0]
                i = i+1
                if not EC:
                    labels.append(i)
    return result_data,labels


def parse_openbci_data(channel_names,split,sample_start=0,sample_end=20000,
                       data_path=r'C:\Users\innopolis\Desktop\IntershipBCI\OpenBCI\OpenBCI_Python-master\data\Alpha Waves\\',filenames=None):
    '''
    :param channel_names: - array of channels (currently unused, huh)
    :param split: - number of subarrays (epochs) to split each file to
    :param sample_start: - starting sample
    :param sample_end: - ending sample
    :param data_path: - path to data (default Alpha waves)
    :param filenames: - filenames array (if None gets all files from data_path)
    :return: user matrix (n_subjects,n_files,n_epochs) matrix
            labels - useless, heh
    '''
    if not filenames:
        from os import listdir
        from os.path import isfile, join
        filenames = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    def f(x):
        return 4.5 / x / (2 ^ 23 - 1)
    count2volts = np.vectorize(f)
    def multiple(x):
        return x * pow(10, 3)
    mvolts2volts = np.vectorize(multiple)
    user_matrix = [[] for x in range(7)] #TODO Fix me pls. :c
    labels = []
    for id,file in enumerate(filenames):
        subjectid = int(file.split("_")[1][-1])
        data = np.loadtxt(data_path + file, delimiter=',')
        data =  np.transpose(data[sample_start:sample_end])
        for j in range(len(data)):
            pass
            data[j] = butter_bandpass_filter(data[j]) #apply filtering
            #data[j] = butter_bandstop_filter(data[j])
        epochs_raw = np.asarray(np.array_split(data,split,axis=1))
        user_matrix[subjectid].append(epochs_raw)
        for x in range(split):
            labels.append(subjectid)
    return user_matrix,labels




def butter_bandstop_filter(data):
    fs_Hz = 250.0
    bp2_stop_Hz = np.array([49, 51.0])
    b2, a2 = signal.butter(4, bp2_stop_Hz / (fs_Hz / 2.0), 'bandstop')
    y = signal.lfilter(b2, a2, data)
    return y

def butter_bandpass_filter(data):
    fs_Hz = 250.0
    bp2_stop_Hz = np.array([5, 30])
    b2, a2 = signal.butter(6, bp2_stop_Hz / (fs_Hz / 2.0), 'bandpass')
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

