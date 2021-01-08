from keras import Input,Model
from keras.models import Sequential

from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from keras.utils import plot_model
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import pickle
import scipy.io as sio
import numpy as np
from numpy import linalg as la
import argparse
import pandas as pd
import sklearn.model_selection
import numpy as np
# import keras as K
import matplotlib.pyplot as plt
from keras import regularizers, Model, Input
import random
import tensorflow as tf
from keras import backend as K
from keras.engine import Layer
from keras.utils import np_utils
from sklearn.decomposition import PCA

from keras.layers import Conv2D, MaxPooling2D, LSTM, Flatten, Conv1D, MaxPooling1D, Dense, Activation, \
    Dropout, GlobalMaxPooling1D, AveragePooling2D, ConvLSTM2D, GlobalMaxPooling2D, Recurrent, Reshape, Bidirectional, \
    BatchNormalization, Merge, concatenate, activations, add

from keras.layers import Embedding, Permute, RNN, GRU

from Uilts.IndRNN_Layer import IndRNN


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim1 = int(inputs.shape[1])
    input_dim2 = int(inputs.shape[2])
    input_dim3 = int(inputs.shape[3])

    a = Permute((3, 1,2))(inputs)
    a = Reshape((input_dim3, input_dim2,input_dim1))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim2, activation='softmax')(a)

    a_probs = Permute((3,2,1))(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return a_probs


def attention_3d_block_zong(inputs2):
    input_dim1 = int(inputs2.shape[1])
    input_dim2 = int(inputs2.shape[2])
    input_dim3 = int(inputs2.shape[3])

    a = Permute((3, 2,1))(inputs2)
    a = Reshape((input_dim3, input_dim2,input_dim1 ))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim2, activation='softmax')(a)

    b_probs = Permute((3,2,1))(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return b_probs

class Group(Layer):
    def __init__(self,  activation=None, **kwargs):
        # self.units = units
        self.activation = activations.get(activation)
        super(Group, self).__init__(**kwargs)

    def call(self, x):
        # assert isinstance(x, list)


        group2=[]
        image_size = x.get_shape()[2].value
        n = (image_size) // 8
        # group1 = x[:,:,0:140]

        for i in range(1,9):
            x1 = x[:,:,((i-1)*n):(i*n)]
            group2.append(x1)

        x1 = group2[0]
        x2 = group2[1]
        x3 = group2[2]
        x4 = group2[3]
        x5 = group2[4]
        x6 = group2[5]
        x7 = group2[6]
        x8 = group2[7]




        return [x1,x2,x3,x4,x5,x6,x7,x8]

    def compute_output_shape(self, input_shape):
        # assert isinstance(input_shape, list)
        input_dim = (input_shape[0],input_shape[1], (input_shape[2])//8)
        input_dim1 = (input_shape[0], input_shape[1], 140)
        return [input_dim,input_dim,input_dim,input_dim,input_dim,input_dim,input_dim,input_dim]







def TransDict_from_list(groups): #给同一个类中不同的字母打数字特征
    transDict = dict()
    tar_list = ['0', '1', '2', '3', '4', '5', '6']
    result = {}
    index = 0
    for group in groups:
        g_members = sorted(group)  # Alphabetically sorted list
        for c in g_members:
            # print('c' + str(c))
            # print('g_members[0]' + str(g_members[0]))
            result[c] = str(tar_list[index])  # K:V map, use group's first letter as represent.
        index = index + 1
    return result

def get_3_protein_trids():
    nucle_com = []
    chars = ['0', '1', '2', '3', '4', '5', '6']
    base = len(chars)


    end = len(chars) ** 3

    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n / base
        ch1 = chars[int(n % base)]
        n = n / base
        ch2 = chars[int(n % base)]

        nucle_com.append(ch0 + ch1 + ch2)
    return nucle_com

def translate_sequence(seq, TranslationDict):
    '''
    Given (seq) - a string/sequence to translate,
    Translates into a reduced alphabet, using a translation dict provided
    by the TransDict_from_list() method.
    Returns the string/sequence in the new, reduced alphabet.
    Remember - in Python string are immutable..

    '''
    import string
    from_list = []
    to_list = []
    for k, v in TranslationDict.items():
        from_list.append(k)
        to_list.append(v)
    # TRANS_seq = seq.translate(str.maketrans(zip(from_list,to_list)))
    TRANS_seq = seq.translate(str.maketrans(str(from_list), str(to_list)))
    # TRANS_seq = maketrans( TranslationDict, seq)
    return TRANS_seq

def get_4_nucleotide_composition(tris, seq, pythoncount=True):
    seq_len = len(seq)
    tri_feature = [0] * len(tris)
    k = len(tris[0])

    note_feature = [[0 for cols in range(len(seq) - k + 1)] for rows in range(len(tris))]

    if pythoncount:
        for val in tris:
            num = seq.count(val)
            tri_feature.append(float(num) / seq_len)
    else:
        # tmp_fea = [0] * len(tris)
        for x in range(len(seq) + 1 - k):
            kmer = seq[x:x + k]
            if kmer in tris:
                ind = tris.index(kmer)
                # tmp_fea[ind] = tmp_fea[ind] + 1
                note_feature[ind][x] = note_feature[ind][x] + 1
        # tri_feature = [float(val)/seq_len for val in tmp_fea]    #tri_feature type:list len:256
        u, s, v = la.svd(note_feature)
        for i in range(len(s)):
            tri_feature = tri_feature + u[i] * s[i] / seq_len
        # print tri_feature
        # pdb.set_trace()

    return tri_feature
# def get_4_nucleotide_composition(tris, seq, pythoncount=True):
#     seq_len = len(seq)
#     tri_feature = []
#
#     if pythoncount:
#         for val in tris:
#             num = seq.count(val)
#             tri_feature.append(float(num) / seq_len)
#     else:
#         k = len(tris[0])
#         tmp_fea = [0] * len(tris)
#         for x in range(len(seq) + 1 - k):
#             kmer = seq[x:x + k]
#             if kmer in tris:
#                 ind = tris.index(kmer)
#                 tmp_fea[ind] = tmp_fea[ind] + 1
#         tri_feature = [float(val) / seq_len for val in tmp_fea]
#         # pdb.set_trace()
#     return tri_feature

def prepare_feature_acp740():
    label = []
    interaction_pair = {}
    RNA_seq_dict = {}
    protein_seq_dict = {}
    protein_index = 0
    with open('acp740.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                values = line[1:].strip().split('|')
                label_temp = values[1]
                proteinName = values[0]
                if label_temp == '1':
                    label.append(1)
                else:
                    label.append(0)
            else:
                seq = line[:-1]
                protein_seq_dict[protein_index] = seq
                protein_index = protein_index + 1
    # name_list = read_name_from_lncRNA_fasta('ncRNA-protein/lncRNA_RNA.fa')
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_3_protein_trids()
    # tris3 = get_3_trids()
    bpf=[]
    kmer=[]
    # get protein feature
    # pdb.set_trace()
    for i in protein_seq_dict:  # and protein_fea_dict.has_key(protein) and RNA_fea_dict.has_key(RNA):
        protein_seq = translate_sequence(protein_seq_dict[i], group_dict)
        bpf_feature = BPF(protein_seq_dict[i])
        # print('bpf:',shape(bpf_feature))
        # pdb.set_trace()
        # RNA_tri_fea = get_4_nucleotide_composition(tris, RNA_seq, pythoncount=False)
        protein_tri_fea = get_4_nucleotide_composition(protein_tris, protein_seq, pythoncount =False)

        bpf.append(bpf_feature)
        kmer.append(protein_tri_fea)
        # protein_index = protein_index + 1
        # chem_fea.append(chem_tmp_fea)
    return np.array(bpf), np.array(kmer), label

def prepare_feature_acp240():
    label = []
    interaction_pair = {}
    RNA_seq_dict = {}
    protein_seq_dict = {}
    protein_index = 1
    with open('acp240.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                values = line[1:].strip().split('|')
                label_temp = values[1]
                protein = values[0]
                if label_temp=='1':
                    label.append(1)
                else:
                    label.append(0)
            else:
                seq = line[:-1]
                protein_seq_dict[protein_index] = seq
                protein_index = protein_index + 1
    # name_list = read_name_from_lncRNA_fasta('ncRNA-protein/lncRNA_RNA.fa')
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_3_protein_trids()
    # tris3 = get_3_trids()
    bpf = []
    kmer = []
    # get protein feature
    # pdb.set_trace()
    for i in protein_seq_dict:  # and protein_fea_dict.has_key(protein) and RNA_fea_dict.has_key(RNA):

        protein_seq = translate_sequence(protein_seq_dict[i], group_dict)
        bpf_feature = BPF(protein_seq_dict[i])
        # print('bpf:',shape(bpf_feature))
        # pdb.set_trace()
        # RNA_tri_fea = get_4_nucleotide_composition(tris, RNA_seq, pythoncount=False)
        protein_tri_fea = get_4_nucleotide_composition(protein_tris, protein_seq, pythoncount =False)

        bpf.append(bpf_feature)
        kmer.append(protein_tri_fea)
        protein_index = protein_index + 1
        # chem_fea.append(chem_tmp_fea)

    return np.array(bpf), np.array(kmer), label

def BPF(seq_temp):
    seq = seq_temp
    chars = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    fea = []
    tem_vec =[]
    k = 7
    for i in range(k):
        if seq[i] =='A':
            tem_vec = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='C':
            tem_vec = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='D':
            tem_vec = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='E':
            tem_vec = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='F':
            tem_vec = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='G':
            tem_vec = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='H':
            tem_vec = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='I':
            tem_vec = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='K':
            tem_vec = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='L':
            tem_vec = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='M':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='N':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
        elif seq[i]=='P':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
        elif seq[i]=='Q':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
        elif seq[i]=='R':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
        elif seq[i]=='S':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
        elif seq[i]=='T':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
        elif seq[i]=='V':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
        elif seq[i]=='W':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        elif seq[i]=='Y':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        fea = fea + tem_vec
    return fea

def prepare_feature():
    label = []
    interaction_pair = {}
    RNA_seq_dict = {}
    protein_seq_dict = {}
    protein_index = 1
    with open('acp740.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                values = line[1:].strip().split('|')
                label_temp = values[1]
                protein = values[0]
                if label_temp == '1':
                    label.append(1)
                else:
                    label.append(0)
            else:
                seq = line[:-1]
                protein_seq_dict[protein_index] = seq
                protein_index = protein_index + 1
    # name_list = read_name_from_lncRNA_fasta('ncRNA-protein/lncRNA_RNA.fa')
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_3_protein_trids()

    # tris3 = get_3_trids()
    train = []
    # get protein feature
    # pdb.set_trace()
    for i in protein_seq_dict:  # and protein_fea_dict.has_key(protein) and RNA_fea_dict.has_key(RNA):

        protein_seq = translate_sequence(protein_seq_dict[i], group_dict)
        # bpf_feature = BPF(protein_seq_dict[i])
        # pdb.set_trace()
        # RNA_tri_fea = get_4_nucleotide_composition(tris, RNA_seq, pythoncount=False)
        protein_tri_fea = get_4_nucleotide_composition(protein_tris, protein_seq, pythoncount =False)

        train.append(protein_tri_fea)
        protein_index = protein_index + 1
        # chem_fea.append(chem_tmp_fea)

    return np.array(train), label
def calculate_performace(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / test_num
    precision = float(tp) / (tp + fp)
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return acc, precision, sensitivity, specificity, MCC

def transfer_label_from_prob(proba):
    label = [1 if val >= 0.5 else 0 for val in proba]
    return label
def plot_roc_curve(labels, probality, legend_text, auc_tag=True):
    # fpr2, tpr2, thresholds = roc_curve(labels, pred_y)
    fpr, tpr, thresholds = roc_curve(labels, probality)  # probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    if auc_tag:
        rects1 = plt.plot(fpr, tpr, label=legend_text + ' (AUC=%6.3f) ' % roc_auc)
    else:
        rects1 = plt.plot(fpr, tpr, label=legend_text)

def ACP_DL():
    # define parameters
    data_dim = 483
    timesteps = 1
    batch_size = 64   # if dataset = acp240, set batch_size = 32; if dataset = acp740, set batch_size = 64
    epochs = 30
    # get data

    bpf, kmer, label = prepare_feature_acp740()
    # bpf, kmer, label = prepare_feature_acp240()

    X = np.concatenate((bpf, kmer), axis=1)  # 1 行拼接 0 默认，列拼接

    #  expected input data shape: (batch_size, timesteps, data_dim)
    estimator = PCA(n_components=0.9999)
    estimator.fit(X)
    X = estimator.fit_transform(X)
    print(X.shape)
    X = np.reshape(X, (len(X), timesteps, 349)) #240 207;740 : 349





    # split data
    # x_train, x_test, y_train, y_test = train_test_split(X, label, test_size=0.1, random_state=1024)
    num_cross_val = 5  # 5-fold
    all_performance_lstm = []
    all_labels = []
    all_prob = {}
    num_classifier = 3
    all_prob[0] = []
    all_average = []

    for fold in range(num_cross_val):
        # train = np.array([x for i, x in enumerate(bpf_fea) if i % num_cross_val != fold])
        # test = np.array([x for i, x in enumerate(bpf_fea) if i % num_cross_val == fold])
        # train = np.array([x for i, x in enumerate(kmer_fea) if i % num_cross_val != fold])
        # test = np.array([x for i, x in enumerate(kmer_fea) if i % num_cross_val == fold])
        train = np.array([x for i, x in enumerate(X) if i % num_cross_val != fold])
        test = np.array([x for i, x in enumerate(X) if i % num_cross_val == fold])
        train_label = np.array([x for i, x in enumerate(label) if i % num_cross_val != fold])
        test_label = np.array([x for i, x in enumerate(label) if i % num_cross_val == fold])
        real_labels = []
        for val in test_label:
            if val == 1:
                real_labels.append(1)
            else:
                real_labels.append(0)

        train_label_new = []
        for val in train_label:
            if val == 1:
                train_label_new.append(1)
            else:
                train_label_new.append(0)
        all_labels = all_labels + real_labels
        # init = initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=64)


        model1_7 = Input(shape=(1,349)) #240 207;740 : 349

        # x = Bidirectional(LSTM(128, return_sequences=True))(model1_7)


        x1,x2,x3,x4,x5,x6,x7,x8= Group()(model1_7)

        x11 = Bidirectional(GRU(64, return_sequences=True))(x1)
        x22 = Bidirectional(GRU(64, return_sequences=True))(x2)
        x33 = Bidirectional(GRU(64, return_sequences=True))(x3)
        x44 = Bidirectional(GRU(64, return_sequences=True))(x4)
        x55 = Bidirectional(GRU(64, return_sequences=True))(x5)
        x66 = Bidirectional(GRU(64, return_sequences=True))(x6)
        x77 = Bidirectional(GRU(64, return_sequences=True))(x7)
        x88 = Bidirectional(GRU(64, return_sequences=True))(x8)
        x = concatenate([x11,x22,x33,x44,x55,x66,x77,x88],axis=1)
        x = Activation("relu")(x)
        # #
        # #
        xx1 = Conv1D(filters=32, kernel_size=3, strides=1,dilation_rate=3)(x)
        xx1 = BatchNormalization()(xx1)
        xx2 = Conv1D(filters=32, kernel_size=3, strides=1,dilation_rate=2)(x)
        xx2 = BatchNormalization()(xx2)
        xx12 = concatenate([xx1, xx2], axis=1)
        xx12 = Activation("relu")(xx12)
        #
        #
        #
        xx3 = Conv1D(filters=32, kernel_size=3, strides=1, dilation_rate=1)(x)
        xx3 = BatchNormalization()(xx3)
        xx4 = Conv1D(filters=32, kernel_size=5, strides=1, dilation_rate=1)(x)
        xx4 = BatchNormalization()(xx4)
        xx34 = concatenate([xx3,xx4], axis=1)
        xx34 = Activation("relu")(xx34)
        #
        xxx1 = concatenate([xx12, xx34], axis=1)

        x = concatenate([Dense(16)(xxx1),Dense(16)(x)],axis=1)

        x = Flatten()(x)

        x = Dropout(0.25)(x)
        x = Dense(1, name='full_connect')(x)
        x = Activation('sigmoid')(x)
        model = Model(inputs=[model1_7], outputs=[x])
        model.summary()

        print('Compiling the Model...')
        model.compile(loss='binary_crossentropy',  #
              optimizer='adam',
              metrics=['accuracy'])

        print("Train...")

        model.fit(train, train_label, batch_size=batch_size,epochs=epochs,verbose=2)

        lstm_proba = model.predict(test)
        all_prob[0] = all_prob[0] + [val for val in lstm_proba]
        y_pred_xgb = transfer_label_from_prob(lstm_proba)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), y_pred_xgb, real_labels)
        print(acc, precision, sensitivity, specificity, MCC)
        all_performance_lstm.append([acc, precision, sensitivity, specificity, MCC])
        print('---' * 50)

    print('mean performance of ACP_DL')
    print(np.mean(np.array(all_performance_lstm), axis=0))
    print('---' * 50)

    plot_roc_curve(all_labels, all_prob[0], 'proposed method')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    # plt.savefig(save_fig_dir + selected + '_' + class_type + '.png')
    plt.show()
ACP_DL()