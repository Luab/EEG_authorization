import numpy as np
from mne.decoding import CSP

import mne
matrix = np.load(r"data\Alcoholics\alcoholics_score_matrix.npy")
matrix2 = np.load(r"data\Alcoholics\alcoholics_score_matrix3.npy")
for id,row in enumerate(matrix):
    print(id)
    print(np.mean(np.concatenate((row[:id],row[id+1:])))-np.mean(np.concatenate((matrix2[id][:id],matrix2[id][id+1:]))))
    #print(min(np.concatenate((row[:id],row[id+1:120])),np.concatenate((matrix2[id][:id],matrix2[id][id+1:120]))))

