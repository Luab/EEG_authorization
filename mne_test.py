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
    i = 0
    for file in filenames:
        data = np.loadtxt(data_path + file, delimiter=',')
        data =  np.transpose(data[sample_start:sample_end])
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
filenames_test = ["processed_Bulat_EEG_FP_2.txt","processed_Vitaly_EEG_FP_2.txt"]
#data,events,labels = parse_data(filenames)
#epochs_data_train, labels = eegparser.split_data_evenly(data, 100, labels, 25000)

epochs_data_train,labels = parse_data(filenames,["V2","V3"],500,sample_start=5000,sample_end=75000)
epochs_data_test,labels_test = parse_data(filenames_test,["V2","V3"],500,sample_start=5000,sample_end=75000)

epochs_data_train = np.concatenate((epochs_data_test,epochs_data_train))
labels = np.append(labels,labels_test)

    
print(len(epochs_data_train))
print(len(labels))


'''
print(labels)
print(len(epochs_data_train))
print(epochs_data_train.shape)
'''
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


lda = LDA()
csp1 = CSP(n_components=4, reg='ledoit_wolf', log=True, cov_est="epoch")
csp2 = CSP(n_components=4, reg='ledoit_wolf', log=True, cov_est="epoch")
csp3 = CSP(n_components=4, reg='ledoit_wolf', log=True, cov_est="epoch")
csp4 = CSP(n_components=4, reg='ledoit_wolf', log=True, cov_est="epoch")
csp5 = CSP(n_components=4, reg='ledoit_wolf', log=True, cov_est="epoch")
csp6 = CSP(n_components=4, reg='ledoit_wolf', log=True, cov_est="epoch")

pca = PCA(n_components=4)
svm = svm.SVC()
gnb = GaussianNB()
nn = MLPClassifier(hidden_layer_sizes=((20,15,5)))
gradien_boost = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
desfor = RandomForestClassifier(n_estimators=30)


epochs_data_train, labels = shuffle(epochs_data_train,labels)
epochs_data_test, labels_test = shuffle(epochs_data_test,labels_test)

#print(epochs_data_train[0])
cv1 = ShuffleSplit(len(labels),test_size=0.2)
cv2 = ShuffleSplit(len(labels),test_size=0.2)
cv3= ShuffleSplit(len(labels),test_size=0.2)
cv4 = ShuffleSplit(len(labels),test_size=0.2)

scores = []

from sklearn.pipeline import Pipeline  # noqa
from sklearn.cross_validation import cross_val_score  # noqa


clf = Pipeline([('CSP', csp1), ('LDA', lda)])
clf.fit(epochs_data_train,labels)
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv1, n_jobs=1)
print("LDA accuracy on test data: %f" % clf.score(epochs_data_test,labels_test))
print(scores)
print("Classification accuracy LDA: %f" % np.mean(scores))

clf2 = Pipeline([('CSP', csp2), ('SVM', svm)])
scores2 = cross_val_score(clf2, epochs_data_train, labels, cv=cv2, n_jobs=1)
clf2.fit(epochs_data_train,labels)
print("SVM accuracy on test data: %f" % clf2.score(epochs_data_test,labels_test))
print(scores2)
print("Classification accuracy SVM: %f" % np.mean(scores2))

predictor3 = Pipeline([('CSP', csp3), ('GNB', gnb)])
scores3 = cross_val_score(predictor3, epochs_data_train, labels, cv=cv3, n_jobs=1)
predictor3.fit(epochs_data_train,labels)
print("GNB accuracy on test data: %f" % predictor3.score(epochs_data_test,labels_test))
print(scores3)
print("Classification accuracy GNB: %f" % np.mean(scores3))

predictor4 = Pipeline([('CSP', csp4), ('Desicion forest', desfor)])
scores4 = cross_val_score(predictor4, epochs_data_train, labels, cv=cv4, n_jobs=1)
predictor4.fit(epochs_data_train,labels)
print("Desicion forest accuracy on test data: %f" % predictor4.score(epochs_data_test,labels_test))

print(scores4)
print("Classification accuracy Desicion forest: %f" % np.mean(scores4))
'''
predictor5 = Pipeline([('CSP', csp5), ('Neural Network', nn)])
scores5 = cross_val_score(predictor5, epochs_data_train, labels, cv=cv4, n_jobs=1)
predictor5.fit(epochs_data_train,labels)
print("NN accuracy on test data: %f" % predictor5.score(epochs_data_test,labels_test))
print("NN accuracy on train data: %f" % predictor5.score(epochs_data_train,labels))
print(scores5)
print("Classification accuracy Neural Network: %f" % np.mean(scores5))
'''

predictor6 = Pipeline([('CSP', csp6), ('GradientBoostingClassifier', gradien_boost)])
scores6 = cross_val_score(predictor6, epochs_data_train, labels, cv=cv4, n_jobs=1)
predictor6.fit(epochs_data_train,labels)
print("Gradient Tree Boosting accuracy on test data: %f" % predictor6.score(epochs_data_test,labels_test))
print(scores6)
print("Classification accuracy Gradient Tree Boosting: %f" % np.mean(scores6))


sfreq = 250
w_length = int(sfreq * 0.3)   # running classifier: window length
w_step = int(sfreq * 0.1)  # running classifier: window step size
w_start = np.arange(0, epochs_data_train.shape[2] - 30, w_step)
scores_windows = []
labels =  np.asarray(labels)
for train_idx, test_idx in cv2:
    y_train, y_test = labels[ np.asarray(train_idx)], labels[ np.asarray(test_idx)]

    X_train = csp5.fit_transform(epochs_data_train[ np.asarray(train_idx)], y_train)
    X_test = csp5.transform(epochs_data_train[ np.asarray(test_idx)])

    # fit classifier
    nn.fit(X_train, y_train)

    # running classifier: test classifier on sliding window
    score_this_window = []
    for n in w_start:
        X_test = csp5.transform(epochs_data_train[test_idx][:, :, n:(n + w_length)])
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
