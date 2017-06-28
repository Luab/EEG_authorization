import numpy as np
import mne
from mne.decoding import CSP
import matplotlib.pyplot as plt

from scipy import signal

import eegparser

def butter_bandpass_filter(data):
    fs_Hz = 250.0
    bp2_stop_Hz = np.array([49, 51.0])
    b2, a2 = signal.butter(2, bp2_stop_Hz / (fs_Hz / 2.0), 'bandstop')
    y = signal.lfilter(b2, a2, data)
    return y

def parse_data(filenames,channel_names,split,sample_number=0):
    def f(x):
        return 4.5 / x / (2 ^ 23 - 1)
    count2volts = np.vectorize(f)
    def multiple(x):
        return x * pow(10, 3)
    mvolts2volts = np.vectorize(multiple)

    data_path = r'C:\Users\innopolis\Desktop\Intership BCI\OpenBCI\OpenBCI_Python-master\data\\'
    epochs_data_train = []
    labels = []
    i = 0
    for file in filenames:
        data = np.loadtxt(data_path + file, delimiter=',')
        data =  np.transpose(data[:sample_number])
        print(len(data[0]))
        for j in range(len(data)):
            data[j] = butter_bandpass_filter(data[j]) #apply filtering
        epochs_raw = np.asarray(np.array_split(data,split,axis=1))
        if np.asarray(epochs_data_train).size == 0:
            epochs_data_train = epochs_raw
        else:
            epochs_data_train = np.concatenate((np.asarray(epochs_data_train),epochs_raw))
        for x in range(split):
            labels.append(i)
        i = i+1

    return epochs_data_train,labels
filenames = ["processed_Bulat_EEG_FP.txt","processed_Vitaly_EEG_FP.txt"]

#data,events,labels = parse_data(filenames)
#epochs_data_train, labels = eegparser.split_data_evenly(data, 100, labels, 25000)

epochs_data_train,labels = parse_data(filenames,["V2","V3"],100,sample_number=25000)

print(len(labels))
print(len(epochs_data_train))
print(epochs_data_train.shape)

from sklearn.utils import shuffle
from sklearn.lda import LDA  # noqa
from sklearn.cross_validation import ShuffleSplit  # noqa
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import cohen_kappa_score
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

lda = LDA()
csp = CSP(n_components=4, reg='ledoit_wolf', log=True)
pca = PCA(n_components=4)
svm = svm.SVC()
gnb = GaussianNB()
nn = MLPClassifier(hidden_layer_sizes=((20,15,5)))
desfor = RandomForestClassifier(n_estimators=30)


epochs_data_train, labels = shuffle(epochs_data_train,labels)
#print(epochs_data_train[0])
cv1 = ShuffleSplit(len(labels),test_size=0.2)
cv2 = ShuffleSplit(len(labels),test_size=0.2)
cv3= ShuffleSplit(len(labels),test_size=0.2)
cv4 = ShuffleSplit(len(labels),test_size=0.2)

scores = []

from sklearn.pipeline import Pipeline  # noqa
from sklearn.cross_validation import cross_val_score  # noqa


clf = Pipeline([('CSP', csp), ('LDA', lda)])
clf2 = Pipeline([('CSP', csp), ('SVM', svm)])
predictor3 = Pipeline([('CSP', csp), ('GNB', gnb)])
predictor4 = Pipeline([('CSP', csp), ('Desicion forest', desfor)])
predictor5 = Pipeline([('CSP', csp), ('Neural Network', nn)])

scores = cross_val_score(clf, epochs_data_train, labels, cv=cv1, n_jobs=1)
scores2 = cross_val_score(clf2, epochs_data_train, labels, cv=cv2, n_jobs=1)
scores3 = cross_val_score(predictor3, epochs_data_train, labels, cv=cv3, n_jobs=1)
scores4 = cross_val_score(predictor4, epochs_data_train, labels, cv=cv4, n_jobs=1)
scores5 = cross_val_score(predictor5, epochs_data_train, labels, cv=cv4, n_jobs=1)


print(scores)
print(scores2)
print(scores3)
print(scores4)
print(scores5)

print("Classification accuracy LDA: %f" % np.mean(scores))
print("Classification accuracy SVM: %f" % np.mean(scores2))
print("Classification accuracy GNB: %f" % np.mean(scores3))
print("Classification accuracy Desicion forest: %f" % np.mean(scores4))
print("Classification accuracy Neural Network: %f" % np.mean(scores5))

sfreq = 250
w_length = int(sfreq * 0.3)   # running classifier: window length
w_step = int(sfreq * 0.1)  # running classifier: window step size
w_start = np.arange(0, epochs_data_train.shape[2] - 30, w_step)
print(w_length)
print(w_step)
print(w_start)
scores_windows = []
labels =  np.asarray(labels)
for train_idx, test_idx in cv2:
    y_train, y_test = labels[ np.asarray(train_idx)], labels[ np.asarray(test_idx)]

    X_train = csp.fit_transform(epochs_data_train[ np.asarray(train_idx)], y_train)
    X_test = csp.transform(epochs_data_train[ np.asarray(test_idx)])

    # fit classifier
    nn.fit(X_train, y_train)

    # running classifier: test classifier on sliding window
    score_this_window = []
    for n in w_start:
        X_test = csp.transform(epochs_data_train[test_idx][:, :, n:(n + w_length)])
        score_this_window.append(nn.score(X_test, y_test))
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