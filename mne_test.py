import numpy as np

from mne.decoding import CSP
import matplotlib.pyplot as plt

from scipy import signal

import eegparser


from sklearn.lda import LDA  # noqa
from sklearn.cross_validation import ShuffleSplit  # noqa

from sklearn.cross_validation import cross_val_score


#alpha_user_matrix,labels = eegparser.parse_from_alcoholic_dataset(r'C:\Users\innopolis\Desktop\IntershipBCI\OpenBCI\OpenBCI_Python-master\data\Alcoholics\result')

#np.save(r"data\Alcoholics\alcoholics_user_matrix.npy",alpha_user_matrix)
#user_mauser_matrix = np.delete(user_matrix,108)
#user_matrix = np.delete(user_matrix,116)
#alpha_user_matrix = np.load(r"data\Alcoholics\alcoholics_user_matrix.npy")


alpha_path = r'C:\Users\innopolis\Desktop\IntershipBCI\OpenBCI\OpenBCI_Python-master\data\Alpha waves'
counting_closed_path = r'C:\Users\innopolis\Desktop\IntershipBCI\OpenBCI\OpenBCI_Python-master\data\Counting_Eyes_Closed'
counting_open_path = r'C:\Users\innopolis\Desktop\IntershipBCI\OpenBCI\OpenBCI_Python-master\data\Counting_Eyes_open'
random_task = r'C:\Users\innopolis\Desktop\IntershipBCI\OpenBCI\OpenBCI_Python-master\data\random task'

alpha_user_matrix, useless = eegparser.parse_openbci_data(["V2", "V3"], 290, sample_start=0, sample_end=70000)
#counting_closed, useless = eegparser.parse_openbci_data(["V2", "V3"], 7, sample_start=0, sample_end=1750,data_path=counting_closed_path)
#counting_open, useless = eegparser.parse_openbci_data(["V2", "V3"], 7, sample_start=0, sample_end=1750,data_path=counting_open_path)


def fit_classifier_cross_val_score(data, labels, classifier):
        cv = ShuffleSplit(len(labels), test_size=0.3)
        scores = cross_val_score(classifier, np.asarray(data), np.asarray(labels), cv=cv, n_jobs=1)
        classifier.fit(data,labels)
        return np.mean(scores)

def create_confidence_matrix(user_matix,file_number=0,limit=None):
    from sklearn.pipeline import Pipeline
    if not limit:
        limit=100
    score_matrix = np.full((len(user_matix),len(user_matix)),0)
    classifier_matrix = [[0 for x in range(len(user_matix))] for y in range(len(user_matix))]
    print("New confidence matrix")
    for id,subject in enumerate(user_matix):
        #print("Training user for "+str(id))
        for oid,other_subject in enumerate(user_matix):
            if id == oid:
                pass
            else:
                lda = LDA()
                labels1 = [0 for i in range(len(subject[file_number]))]
                labels2 = [1 for j in range(len(other_subject[file_number]))]
                labels = np.concatenate((np.asarray(labels1), np.asarray(labels2)))
                try:
                    data = np.concatenate((np.asarray(subject[file_number]), np.asarray(other_subject[file_number])))
                except:
                    print("1")
                if len(data) == len(labels):
                    csp = CSP(n_components=4)
                    clf = Pipeline([('CSP', csp), ("LDA", lda)])
                    score_matrix[id][oid] = fit_classifier_cross_val_score(data, labels, clf)
                    classifier_matrix[id][oid] = clf
                else:
                    print("Smth gone wrong")
    return score_matrix,classifier_matrix

def authorize_old(classifier_matrix,auth_data,user_auth,file_number_first=0,file_number_other=0,threshhold=0.15):
    predict_matrix = np.full((len(classifier_matrix),len(classifier_matrix)),0)
    recognition_matrix = np.full((len(classifier_matrix)),0)
    for id,clf in enumerate(classifier_matrix):
        for oid,cl in enumerate(clf):
            if id == oid:
                pass
            else:
                predict_matrix[id][oid] = np.sum(cl.predict(auth_data))
                for id,user in enumerate(predict_matrix):
                    recognition_matrix[id] = np.sum(user)
    return predict_matrix,recognition_matrix


def authorize(classifier_matrix, auth_data, user_auth, file_number_first=0, file_number_other=0, threshold=0.15):
    '''
    Main authorization function. Test auth_data against user_auth row of classifiers
    :param classifier_matrix: Classifier matrix n_users X n_users shape
    :param auth_data: portion of data to authorize. Shape n_windows,n_channels,n_samples
    :param user_auth: user by which we are trying to authorize
    :param file_number_first: deprecated
    :param file_number_other: deprecated
    :param threshold: t
    :return: predict_matrix - array of predictions for each window. result - boolean
    '''
    predict_matrix = np.full((len(classifier_matrix)), 0)
    result = True
    for id,cl in enumerate(classifier_matrix[user_auth]):
        if id == user_auth:
            pass
        else:
                temp = []
                for window in auth_data:
                    temp.append(cl.predict(np.asarray([window])))
                predict_matrix[id] = np.mean(temp)
                if predict_matrix[id] > threshold:
                    result = False
    averaged_predict = np.mean(predict_matrix)
    return predict_matrix, result, averaged_predict

def imposter_check(user_matrix, classifier, file_number=0, start=None, threshold = 0.15):
    '''
    Function which tries to authorize each user as each user. Works for n^2.
    :param user_matrix: User matrix shape (n_users,n_files,n_windows,n_channels,n_samples)
    :param classifier: Classifier matrix n_users X n_users shape
    :param file_number: number of file to use. If user doesn't have this file number, skip
    :param start: starting windows
    :param threshold: authorization threshold.
    :return: tpr - True positive rate, fpr - False positive rate, fnr - False negative rate
    '''
    if not start:
        start = 0
    fp = 0
    fn = 0
    tn = 0
    tp = 0
    for id, user in enumerate(user_matrix):
        for oid in range(len(user_matrix)):
            if len(user) > file_number:
               # print(str(id) + " as " + str(oid)+" with threshhold: "+str(threshold))
                x, res, ap1 = authorize(classifier, user[file_number], user_auth=oid, threshold=threshold) #TODO REMOVED LIMIT
                if id != oid and res == True:
                    fp = fp +1
                if id != oid and res == False:
                    tn = tn + 1
                if id == oid and res == False:
                    fn = fn + 1
                if id == oid and res == True:
                    tp = tp + 1
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    fnr = fn/(tp+fn)
    return  tpr,fpr

def incremental_training(classifier,user_matrix,file_number=0,):
    for id,subject in enumerate(user_matrix):
        #print("Retraining user for "+str(id))
        for oid,other_subject in enumerate(user_matrix):
            if id == oid:
                pass
            else:
                if len(subject)==0  or len(other_subject) ==0: #OMG PLS FIX ME
                    pass
                else:

                        labels1 = [0 for i in range(len(subject[file_number]))]
                        labels2 = [1 for j in range(len(other_subject[file_number]))]
                        labels = np.concatenate((np.asarray(labels1), np.asarray(labels2)))
                        data = np.concatenate((np.asarray(subject[file_number]), np.asarray(other_subject[file_number])))
                        if len(data) == len(labels):
                            classifier[id][oid].fit(data,labels)
                        else:
                            print("Smth gone wrong")

    return classifier

def prepare_alcoholic_dataset(user_matrix):
    for user in user_matrix:
        user[0] = [epoch for epoch in user]
        del user[1:]
    return user_matrix

def roc_curve(user_matrix,classifier,step=1):
    print("New ROC curve")
    thresh = np.arange(0.0, 1.015, step)
    tpr = [0 for x in range(len(thresh))]
    fpr = [0 for x in range(len(thresh))]
    for id, x in enumerate(thresh):
        print("thresh")
        tpr[id], fpr[id] = imposter_check(user_matrix, classifier, file_number=0, threshold=x)
    roc = [tpr, fpr]
    return np.asarray(roc)


def arraify(user_matrix):
    x = np.empty(len(user_matrix), dtype=np.ndarray)

    for uid,user in enumerate(user_matrix):
        x[uid] = user
        for fid,file in enumerate(user):
            for wid,window in enumerate(file):
                for chid,channel in enumerate(window):
                    user_matrix[uid][fid][wid][chid] = np.asarray(channel)
                    x[uid][fid][wid] = np.asarray(user_matrix[uid][fid][wid])
                    x[uid][fid] = np.asanyarray(user_matrix[uid][fid])
        #user_matrix[uid] = np.asarray(user_matrix[uid])

    return x

def epoch_test(path):
    '''
    UNWORKING SHIT
    :param path:
    :return:
    '''
    from scipy import integrate
    from blah import draw
    window_step = np.arange(280,700,10)
    res = []
    for id,x in enumerate(window_step):
        user_matrix,meh = eegparser.parse_openbci_data(["V2", "V3"], x, sample_start=0, sample_end=70000)
        score, classifier = create_confidence_matrix(np.asarray(user_matrix), file_number=0)
        roc = roc_curve(user_matrix,classifier)
        auc = integrate.trapz(roc[1],roc[0])
        res.append(auc)
    return res,window_step

def cut_small(user_matrix):
    for id,user in enumerate(user_matrix):
        if len(user)<2:
            del user_matrix[id]
    return user_matrix

def filter_users(epic_matrix):
    for aid,activity in enumerate(epic_matrix):
        for id,user in enumerate(activity):
            if len(user) == 0:
                for aid, activity in enumerate(epic_matrix):
                    del activity[id]
    return epic_matrix

def j_statistic(roc):
    return max(np.subtract(roc[0],roc[1]))

def cross_task_train(paths):
    epic_matrix = [0 for _ in range(len(paths))]
    result_matrix = [[0 for _ in range(len(paths))] for _ in range(len(paths))]
    roc_matrix = [[0 for _ in range(len(paths))] for _ in range(len(paths))]
    print("Loading")
    import pickle
    epic_matrix = np.load("epic_matrix.pick")
    for id,activity in enumerate(epic_matrix):
        print(id)
        score, classifier = create_confidence_matrix(np.asarray(activity), file_number=0)
        for oid,another_activity in enumerate(epic_matrix):
            print(oid)
            roc = roc_curve(another_activity,classifier,step=0.1)
            result_matrix[id][oid] = j_statistic(roc)
            roc_matrix[id][oid] = roc
    return result_matrix,roc_matrix

def imposter_check_parrallel(tuple):
    '''
    Function which tries to authorize each user as each user. Works for n^2. HIGHLY UNSTABLE
    :param user_matrix: User matrix shape (n_users,n_files,n_windows,n_channels,n_samples)
    :param classifier: Classifier matrix n_users X n_users shape
    :param file_number: number of file to use. If user doesn't have this file number, skip
    :param start: starting windows
    :param threshold: authorization threshold.
    :return: tpr - True positive rate, fpr - False positive rate, fnr - False negative rate
    '''
    start = 0
    file_number = 0
    user_matrix = tuple[0]
    classifier = tuple[1]
    threshold = tuple[2]
    fp = 0
    fn = 0
    tn = 0
    tp = 0
    for id, user in enumerate(user_matrix):
        for oid in range(len(user_matrix)):
            if len(user) > file_number:
               # print(str(id) + " as " + str(oid)+" with threshhold: "+str(threshold))
                x, res, ap1 = authorize(classifier, user[file_number], user_auth=oid, threshold=threshold) #TODO REMOVED LIMIT
                if id != oid and res == True:
                    fp = fp +1
                if id != oid and res == False:
                    tn = tn + 1
                if id == oid and res == False:
                    fn = fn + 1
                if id == oid and res == True:
                    tp = tp + 1
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    fnr = fn/(tp+fn)
    return tpr,fpr

def roc_curve_parallel(user_matrix,classifier,step=1.0):
    from multiprocessing import Process, Value, Array
    thresh = np.arange(0.0, 1.015, step)
    import ctypes
    tpr = Array(ctypes.c_double, range(10))
    fpr = Array(ctypes.c_double, range(10))
    from multiprocessing.dummy import Pool
    pool = Pool(processes=23)
    iterable = [(user_matrix,classifier,threshold) for threshold in thresh]
    print("MAGICK STARTS")
    roc = pool.map(imposter_check_parrallel,iterable)
    return np.asarray(roc)

def epoch_test_parallel(path):
    '''
    HIGLY UNSTABLE
    :param path:
    :return:
    '''
    from scipy import integrate
    from blah import draw
    window_step = np.arange(280,700,10)
    res = []
    for id,x in enumerate(window_step):
        user_matrix,meh = eegparser.parse_openbci_data(["V2", "V3"], x, sample_start=0, sample_end=70000)
        score, classifier = create_confidence_matrix(np.asarray(user_matrix), file_number=0)
        roc = roc_curve_parallel(user_matrix,classifier,step=0.5)
        auc = integrate.trapz(roc[1],roc[0])
        res.append(auc)
    return res,window_step

def cross_task_train_parallel(paths):
    result_matrix = [[0 for _ in range(len(paths))] for _ in range(len(paths))]
    roc_matrix = [[0 for _ in range(len(paths))] for _ in range(len(paths))]
    print("Loading")
    np.load("epic_matrix.npy",epic_matrix)
    for id,activity in enumerate(epic_matrix):
        print(id)
        score, classifier = create_confidence_matrix(np.asarray(activity), file_number=0)
        for oid,another_activity in enumerate(epic_matrix):
            print(oid)
            roc = roc_curve_parallel(another_activity,classifier,step=0.1)
            result_matrix[id][oid] = j_statistic(roc)
            roc_matrix[id][oid] = roc
    return result_matrix,roc_matrix

alpha_path = r'C:\Users\innopolis\Desktop\IntershipBCI\OpenBCI\OpenBCI_Python-master\data\Alpha waves'
counting_open_path = r'C:\Users\innopolis\Desktop\IntershipBCI\OpenBCI\OpenBCI_Python-master\data\Counting_Eyes_open'
random_task = r'C:\Users\innopolis\Desktop\IntershipBCI\OpenBCI\OpenBCI_Python-master\data\random_task'
random_task_closed_path = r'C:\Users\innopolis\Desktop\IntershipBCI\OpenBCI\OpenBCI_Python-master\data\random_task_closed'

paths = [alpha_path,counting_closed_path,counting_open_path,random_task,random_task_closed_path]
counting_closed_path = r'C:\Users\innopolis\Desktop\IntershipBCI\OpenBCI\OpenBCI_Python-master\data\Counting_Eyes_Closed'
#alpha_user_matrix = prepare_alcoholic_dataset(alpha_user_matrix)
#score,classifier = create_confidence_matrix(np.asarray(alpha_user_matrix),file_number=0)
epic_matrix = [0 for _ in range(len(paths))]
'''
for id, path in enumerate(paths):
    epic_matrix[id], labels = eegparser.parse_openbci_data(["V2", "V3"], 10, sample_start=0, data_path=path)

print("Filter")
epic_matrix = filter_users(epic_matrix)
import pickle
np.save(open("epic_matrix.pick", 'wb'),epic_matrix)
'''
print("Saving")
#x,y = epoch_test(alpha_path)
rm,rocs = cross_task_train(paths)
np.save("cross_task_train",rm)
np.save("rocs",rocs)

# np.save("ROC_curve_wc",roc_curve(alpha_user_matrix,classifier))






def authorize_onevone(classifier,auth_data,user_auth,file_number_first=None,threshhold=0.5):
    if not file_number_first:
        auth_sample = auth_data
    else:
        auth_sample = np.asarray(auth_data[file_number_first])
    score = classifier.score(auth_sample,[user_auth for x in range(len(auth_sample))])
    predict = classifier.predict(auth_sample)
    if score>threshhold:
        res = True
    else:
        res = False
    return score,predict,res

def imposter_check_onevone(user_matrix,classifier,file_number=0,limit=None):
    score_matrix = []
    fp = 0
    fn = 0
    if limit == None:
        limit = 0
    for id, user in enumerate(user_matrix):
        for oid in range(len(user_matrix)):
            if len(user) > file_number:
               # print(str(id) + " as " + str(oid))
                ap1,predict,res = authorize_onevone(classifier, np.asarray(user[file_number][limit:]), user_auth=oid)
                score_matrix.append(ap1)
                if id != oid and res == True:
                   # print("False positive")
                    #print(ap1)
                    #print(predict)
                    fp = fp +1
                if id == oid and res == False:
                   # print("False negative")
                   # print(ap1)
                    #print(predict)
                    fn = fn +1
    return score_matrix

def create_confidence_matrix_one_vs_one(user_matix,file_number=0):
    from sklearn.pipeline import Pipeline
    import OneVsOneImproved
    lda = LDA()
    csp = CSP(n_components=2,transform_into='csp_space')
    clf = Pipeline([('CSP', csp), ("LDA", lda)]) #TODO NOTE THIS CHANGE
    classifier = OneVsOneImproved.OneVsOneClassifier(clf)
    labels = []
    data = []
    for id,subject in enumerate(user_matix):
        if len(subject) > file_number:
            labels1 = [id for i in range(len(subject[file_number]))]
            if not len(labels):
                labels = labels1
                data = subject[file_number]
            else:
                labels = np.concatenate((labels,np.asarray(labels1)))
                data = np.concatenate((data,np.asarray(subject[file_number])))
    #if len(data) != len(labels):
        #print(len(data))
       # print(len(labels))
    score_matrix = fit_classifier_cross_val_score(data, labels, clf)
    classifier.fit(np.asarray(data),np.asarray(labels))
    return score_matrix,classifier

def incremental_training_ovo(classifier,user_matrix,file_number=0):
    labels = []
    data = []
    for id,subject in enumerate(user_matrix):
        if len(subject) > file_number:
                labels1 = [id for i in range(len(subject[file_number]))]
                if not len(labels):
                        labels = labels1
                        data = subject[file_number]
                else:
                        labels = np.concatenate((labels,np.asarray(labels1)))
                        data = np.concatenate((data,np.asarray(subject[file_number])))
    if data != None:
        classifier.fit(np.asarray(data),np.asarray(labels))
    return classifier
