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






user_matrix,labels = eegparser.parse_from_alcoholic_dataset(r'C:\Users\innopolis\Desktop\IntershipBCI\OpenBCI\OpenBCI_Python-master\data\Alcoholics\result')
np.save(r"data\Alcoholics\alcoholics_user_matrix.npy",user_matrix)
#user_mauser_matrix = np.delete(user_matrix,108)
#user_matrix = np.delete(user_matrix,116)
user_matrix = np.load(r"data\Alcoholics\alcoholics_user_matrix.npy")

#filenames = "processed_Vitaly0_EEG_FP_3.txt",["processed_Bulat1_EEG_FP_2.txt","processed_Subject2_Alpha_1_2_channels_1.txt","processed_Subject3_Alpha_1_2_channels_1.txt","processed_Subject4_Alpha_1_2_channels.txt"]
#filename_other = [["processed_Vitaly0_EEG_FP_2.txt"],["processed_Bulat1_EEG_FP_1.txt"],["processed_Subject2_Alpha_1_2_channels_1.txt"],["processed_Subject3_Alpha_1_2_channels_1.txt"],["processed_Subject6_Alpha_1_2_channels_1.txt"]]
#data,events,labels = parse_openbci_data(filenames)
#epochs_data_train, labels = eegparser.split_data_evenly(data, 100, labels, 25000)

#epochs_data_train,labels = parse_openbci_data(filenames,["V2","V3"],1000,sample_start=5000,sample_end=75000)
#test = ["processed_Bulat1_EEG_FP_1.txt"]


#user_matrix, useless = eegparser.parse_openbci_data(["V2", "V3"], 1000, sample_start=5500, sample_end=65500)
#print(np.asarray(user_matrix[0]).shape)

def fit_classifier_with_csp(data, labels, classifier):
    cv = ShuffleSplit(len(labels), test_size=0.1)
    scores = cross_val_score(classifier, data, labels, cv=cv, n_jobs=1)
    return np.mean(scores)

def create_confidence_matrix(user_matix,file_number=0):
    score_matrix = np.full((len(user_matix),len(user_matix)),0)
    classifier_matrix = [[0 for x in range(len(user_matix))] for y in range(len(user_matix))]
    for id,subject in enumerate(user_matix):
        print("Checking user "+str(id))
        for oid,other_subject in enumerate(user_matix):
            if id == oid:
                pass
            else:
                lda = LDA()
                labels1 = [0 for i in range(len(subject))]
                labels2 = [1 for j in range(len(other_subject))]
                labels = np.concatenate((np.asarray(labels1),np.asarray(labels2)))
                data = np.concatenate((np.asarray(subject),np.asarray(other_subject)))
                if len(data) == len(labels):
                        csp = CSP(n_components=4)
                        clf = Pipeline([('CSP', csp), ("LDA", lda)])
                        score_matrix[id][oid] = fit_classifier_with_csp(data, labels, clf)
                        classifier_matrix[id][oid] = clf
                else:
                    print("Smth gone wrong")
    return score_matrix,classifier_matrix

def test_classifier_matrix(classifier_matrix,user_matix,file_number=0):
    from sklearn.cross_validation import permutation_test_score
    score_matrix = np.full((len(user_matix),len(user_matix)),0)
    for id,subject in enumerate(user_matix):
        print("Checking user "+str(id))
        for oid,other_subject in enumerate(user_matix):
            if id == oid:
                pass
            else:
                    try:
                        lda = LDA()
                        labels1 = [0 for i in range(len(subject[file_number]))]
                        labels2 = [1 for j in range(len(other_subject[file_number]))]
                        labels = np.concatenate((np.asarray(labels1),np.asarray(labels2)))
                        data = np.concatenate((np.asarray(subject[file_number]),np.asarray(other_subject[file_number])))
                        if len(data) == len(labels):
                            cv = ShuffleSplit(len(labels), test_size=0.1)
                            score_matrix[id][oid],permutation_scores,pvalue = permutation_test_score(classifier_matrix[id][oid],data,labels,cv)
                            print(pvalue)
                    except IndexError:
                        pass
    return score_matrix

score_matrix,classifier_matrix = create_confidence_matrix(user_matrix)
np.save(r"data\Alcoholics\alcoholics_score_matrix",np.asarray(score_matrix))
new_score = test_classifier_matrix(classifier_matrix,user_matrix)




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