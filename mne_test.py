import numpy as np
from numpy import inf, NaN

import mne
from mne.decoding import CSP
import matplotlib.pyplot as plt

from scipy import signal

import eegparser


from sklearn.utils import shuffle
from sklearn.lda import LDA  # noqa
from sklearn.cross_validation import ShuffleSplit  # noqa
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import cohen_kappa_score
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score

def iirnotch_filter(data):
    fs_Hz = 250.0
    b2, a2 = signal.iirnotch(50/(fs_Hz / 2.0),30)
    y = signal.lfilter(b2, a2, data)
    return y

def butter_bandpass_filter(data):
    fs_Hz = 250.0
    bp2_stop_Hz = np.array([49, 51.0])
    b2, a2 = signal.butter(2, bp2_stop_Hz / (fs_Hz / 2.0), 'bandpass')
    y = signal.lfilter(b2, a2, data)
    return y

def butter_bandstop_filter(data):
    fs_Hz = 250.0
    bp2_stop_Hz = np.array([7, 13])
    b2, a2 = signal.butter(2, bp2_stop_Hz / (fs_Hz / 2.0), 'bandpass')
    y = signal.lfilter(b2, a2, data)
    return y


def parse_data(filenames,channel_names,split,sample_start=0,sample_end=20000):
    def f(x):
        return 4.5 / x / (2 ^ 23 - 1)
    count2volts = np.vectorize(f)
    def multiple(x):
        return x * pow(10, 3)
    mvolts2volts = np.vectorize(multiple)

    data_path = r'C:\Users\innopolis\Desktop\Intership BCI\OpenBCI\OpenBCI_Python-master\data\\'
    epochs_data_train = []
    labels = []
    for id,file in enumerate(filenames):
        data = np.loadtxt(data_path + file, delimiter=',')
        data =  np.transpose(data[sample_start:sample_end])
        plt.plot(data[0])
        plt.plot(butter_bandpass_filter(data[0]))
        plt.plot(iirnotch_filter(data[0]))
        plt.show()
        for j in range(len(data)):
            data[j] = butter_bandpass_filter(data[j]) #apply filtering
            #data[j] = butter_bandstop_filter(data[j])

        epochs_raw = np.asarray(np.array_split(data,split,axis=1))
        if np.asarray(epochs_data_train).size == 0:
            epochs_data_train = epochs_raw
        else:
            epochs_data_train = np.concatenate((np.asarray(epochs_data_train),epochs_raw))
        for x in range(split):
            labels.append(id)

    return epochs_data_train,labels

def filter_nan_and_infinite(array):
    array = list(filter(lambda x: x != float('-inf'), array))
    array[array == inf] = 0.0
    array[array == NaN] = 0.0
    return array

def parse_from_alcoholic_dataset(directory):
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
                        if float(line[3]) == -inf or float(line[3]) == inf or float(line[3]) == NaN:
                            pass
                        else:
                            subject_data[0].append(float(line[3]))
                    if line[1] == "FP2":
                        if float(line[3]) == -inf or float(line[3]) == inf or float(line[3]) == NaN:
                            pass
                        else:
                            subject_data[1].append(float(line[3]))
            for id,channel in enumerate(subject_data):
                if len(channel) == 0:
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


user_matrix,labels = parse_from_alcoholic_dataset(r'C:\Users\innopolis\Desktop\IntershipBCI\OpenBCI\OpenBCI_Python-master\data\Alcoholics\result')
np.save("alcoholics_user_matrix.npy",user_matrix)
#user_matrix = np.load("alcoholics_user_matrix.npy")
user_matrix = np.delete(user_matrix,108)
user_matrix = np.delete(user_matrix,116)


#filenames = "processed_Vitaly_EEG_FP_2.txt",["processed_Bulat_EEG_FP_2.txt","processed_Subject2_Alpha_1_2_channels.txt","processed_Subject3_Alpha_1_2_channels.txt","processed_Subject4_Alpha_1_2_channels.txt"]
#filename_other = [["processed_Vitaly_EEG_FP.txt"],["processed_Bulat_EEG_FP.txt"],["processed_Subject2_Alpha_1_2_channels.txt"],["processed_Subject3_Alpha_1_2_channels.txt"],["processed_Subject1_Alpha_1_2_channels.txt"]]
#data,events,labels = parse_data(filenames)
#epochs_data_train, labels = eegparser.split_data_evenly(data, 100, labels, 25000)

#epochs_data_train,labels = parse_data(filenames,["V2","V3"],1000,sample_start=5000,sample_end=75000)
#test = ["processed_Bulat_EEG_FP.txt"]

'''
user_matrix = []
for user in filename_other:
    user_data, useless = parse_data(user,["V2","V3"],1000,sample_start=5500,sample_end=65500)
    user_matrix.append(user_data)
'''

def fit_classifier_with_csp(data, labels, classifier, classifier_name="classifier"):
    cv = ShuffleSplit(len(labels), test_size=0.8)
    csp = CSP(n_components=4, reg='ledoit_wolf', log=True, cov_est="epoch")
    clf = Pipeline([('CSP', csp), (classifier_name, classifier)])
    scores = cross_val_score(clf, data, labels, cv=cv, n_jobs=1)
    #print(scores)
    #print("Classification accuracy "+classifier_name+": %f" % np.mean(scores))
    return np.mean(scores),clf



def create_confidence_matrix(user_matix):
    score_matrix = np.full((122,122),0)
    for id,user in enumerate(user_matrix):
        for oid,other_user in enumerate(user_matrix):
            if id == oid:
                pass
            else:
                print(len(user))
                print(len(other_user))
                lda = LDA()
                labels1 = [0 for i in range(len(user))]
                labels2 = [1 for j in range(len(other_user))]
                labels = np.concatenate((np.asarray(labels1),np.asarray(labels2)))
                data = np.concatenate((np.asarray(user),np.asarray(other_user)))
                if len(data) == len(labels):
                    score_matrix[id][oid],classifier = fit_classifier_with_csp(data, labels, lda)
                else:
                    print("Smth gone wrong")
    return score_matrix

score_matrix, confusion_score = create_confidence_matrix(user_matrix)
np.save("alcoholics_score_matrix",np.asarray(score_matrix))
for row in score_matrix:
    print(np.mean(row))




'''
lda = LDA()
csp = CSP(n_components=4, reg='ledoit_wolf', log=True, cov_est="epoch")


pca = PCA(n_components=4)
svm = svm.SVC()
gnb = GaussianNB()
nn = MLPClassifier(hidden_layer_sizes=((20,15,5)))
gradien_boost = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
desfor = RandomForestClassifier(n_estimators=30)


#epochs_data_train, labels = shuffle(epochs_data_train,labels)
#epochs_data_test, labels_test = shuffle(epochs_data_test,labels_test)

#print(epochs_data_train[0])
cv = ShuffleSplit(len(labels), test_size=0.2)

scores = []




evaluate_classifier_with_csp(epochs_data_train,labels,lda,"LDA")
evaluate_classifier_with_csp(epochs_data_train,labels,svm,'SVM')
evaluate_classifier_with_csp(epochs_data_train,labels,gnb,"gnb")
evaluate_classifier_with_csp(epochs_data_train,labels,desfor,'desfor')
evaluate_classifier_with_csp(epochs_data_train,labels,nn,"neural network")
evaluate_classifier_with_csp(epochs_data_train,labels,gradien_boost,"GradientBoostingClassifier")

def draw_training(data,labels,classifier,csp):
        sfreq = 250
        w_length = int(sfreq * 0.3)   # running classifier: window length
        w_step = int(sfreq * 0.1)  # running classifier: window step size
        w_start = np.arange(0, epochs_data_train.shape[2] - 30, w_step)
        scores_windows = []
        labels =  np.asarray(labels)
        for train_idx, test_idx in cv:
            y_train, y_test = labels[ np.asarray(train_idx)], labels[ np.asarray(test_idx)]

            X_train = csp.fit_transform(epochs_data_train[ np.asarray(train_idx)], y_train)
            X_test = csp.transform(epochs_data_train[ np.asarray(test_idx)])

            # fit classifier
            classifier.fit(X_train, y_train)

            # running classifier: test classifier on sliding window
            score_this_window = []
            for n in w_start:
                X_test = csp.transform(epochs_data_train[test_idx][:, :, n:(n + w_length)])
                score_this_window.append(classifier.score(X_test, y_test))
            scores_windows.append(score_this_window)

        w_times = (w_start + w_length / 2.) / sfreq + 0

        plt.figure()
        plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
        plt.axvline(0, linestyle='--', color='k', label='Onset')
        plt.axhline(0.5, linestyle='-', color='k', label='Chance')
        plt.xlabel('time (s)')
        plt.ylabel('classification accuracy')
        plt.title('Classification score over time')
        plt.legend(loc='lower right')
        plt.show()
draw_training(epochs_data_train,labels,nn,csp)
'''