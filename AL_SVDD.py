# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 20:33:05 2022

@author: CEOSpace
"""
# @title Executed code for the experiment with output.
# !/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset, Subset
# from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import logging
from scipy import stats
from pylab import rcParams
from sklearn.utils import check_random_state
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, \
    GradientBoostingClassifier
# from visualization_whole import visual
from skimage.transform import resize
from skimage.io import imread
from sklearn.metrics import roc_auc_score
import random
#===========================================================================
def download(data_path,Categories):
       
    flat_data_arr=[] #input array 
    target_arr=[] #output array
    datadir= data_path# +'/Train/' 
    #path which contains all the categories of images
    for i in Categories:
        
        print(f'loading... category : {i}')
        path=os.path.join(datadir,i)
        for img in os.listdir(path):
            img_array=imread(os.path.join(path,img))
            img_resized=resize(img_array,(32,32,3))
            flat_data_arr.append(img_resized.transpose(2,0,1))
            target_arr.append(Categories.index(i))
        print(f'loaded category:{i} successfully')
    x=np.array(flat_data_arr)
    y=np.array(target_arr)
    # df=pd.DataFrame(flat_data) #dataframe
    # df['Target']=target
    # x=np.array(df.iloc[:,:-1]) #input data 
    # y=np.array(df.iloc[:,-1]) #output data
    print('Data:', x.shape, y.shape)
    return (x, y)

#=======================================================================
class NaturalSceneClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class CIFAR10_LeNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
# ==============================================================================

class BaseModel(object):

    def __init__(self):
        pass

    def fit_predict(self):
        pass


class SvmModel(BaseModel):
    model_type = 'Support Vector Machine with linear Kernel'

    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):
        print('training svm...')
        self.classifier = SVC(C=1, kernel='linear', probability=True,
                              class_weight=c_weight)
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_test, self.val_y_predicted,
                self.test_y_predicted)
class CNN(BaseModel):
    model_type = 'CNN'

    def fit_predict(self,X_train, X_val, y_val, X_test,y_test, c_weight):             
        #load the train and test data

        batch_size = 50
        tensor_x = torch.Tensor(X_train)
        
        tensor_x_test = torch.Tensor(X_test)
        tensor_y_test = torch.Tensor(y_test)
        test_data = TensorDataset(tensor_x_test,tensor_y_test)# transform to torch tensor 
        tensor_x_val = torch.Tensor(X_val) 
        tensor_y_val = torch.Tensor(y_val)
        val_data = TensorDataset(tensor_x_val,tensor_y_val)# transform to torch tensor        

        #load the train and validation into batches.
        train_dl = DataLoader(tensor_x, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
        test_dl = DataLoader(test_data, batch_size, num_workers = 4, pin_memory = True)
        val_dl = DataLoader(val_data, batch_size, num_workers = 4, pin_memory = True)
        print('training CNN...')
        num_epochs = 20
        self.c = torch.zeros(128)+0.1
        opt_func = torch.optim.Adam
        lr = 0.001
        model = CIFAR10_LeNet() #NaturalSceneClassification()       
        optimizer = opt_func(model.parameters(),lr)
        for epoch in range(num_epochs):
            
            model.train()
            train_losses = []
            for batch in train_dl:
                # images, labels = batch 
                out = model(batch)
                # labels = labels.type(torch.LongTensor)
                dist = torch.sum((out - self.c) ** 2, dim=1)
                loss = torch.mean(dist) 
                # loss = F.cross_entropy(out, labels)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # logger.info('Loss for {}epoch:{}'.format(epoch,train_losses) )

        print('finished the training CNN...')
        list_score = []
        model.eval()
        with torch.no_grad():
            for data in train_dl:
                # images, labels = data
                out = model(data)
                scores = torch.sum((out - self.c) ** 2, dim=1)
                list_score += list(zip(scores))
        self.M = np.max(list_score)  
        logger.info('Maximum distance %s.' % self.M) 
        # logger.info('train scores\n', list_score)
        label_score = []
        model.eval()
        with torch.no_grad():
            for data in test_dl:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = model(images)
                scores = torch.sum((outputs - self.c) ** 2, dim=1)
                label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                            scores))
        # Compute AUC
        labels, scores = zip(*label_score)
        self.labels = np.array(labels)
        self.scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, self.scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))
        
        idx_label_score = []
        model.eval()
        with torch.no_grad():
            for data in val_dl:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = model(images)
                scores = torch.sum((outputs - self.c) ** 2, dim=1)
                idx_label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                            scores))
        # Compute AUC
        labels, scores = zip(*idx_label_score)
        self.val_scores = np.array(scores)
        # self.val_auc = roc_auc_score(labels, self.val_scores)
        logger.info('Maximum distance in Val %s.' %  np.max(self.val_scores) ) 

        return (X_train, X_val, X_test, self.val_scores, self.M)

class GmmModel(BaseModel):
    model_type = 'Gaussian Mixture Model'

    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):
        print('training gaussian mixture model...')
        pca = PCA(n_components=75).fit(X_train)  # ,whiten=True).fit(X_train)
        reduced_train_data = pca.transform(X_train)
        reduced_test_data = pca.transform(X_test)
        reduced_val_data = pca.transform(X_val)
        print('PCA: explained_variance_ratio_',
              np.sum(pca.explained_variance_ratio_))
        self.classifier = GaussianMixture(n_components=10, covariance_type='full')
        self.classifier.fit(reduced_train_data)
        self.test_y_predicted = \
            self.classifier.predict(reduced_test_data)
        self.val_y_predicted = self.classifier.predict(reduced_val_data)
        return (reduced_train_data, reduced_val_data,
                reduced_test_data, self.val_y_predicted,
                self.test_y_predicted)


class LogModel(BaseModel):
    model_type = 'Multinominal Logistic Regression'

    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):
        print('training multinomial logistic regression')
        train_samples = X_train.shape[0]
        self.classifier = LogisticRegression(
            C=50. / train_samples,
            multi_class='multinomial',
            penalty='l1',
            solver='saga',
            tol=0.1,
            class_weight=c_weight,
        )
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_test, self.val_y_predicted,
                self.test_y_predicted)


class GbcModel(BaseModel):
    model_type = 'Gradient Boosting Classifier'

    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):
        print('training gradient boosting...')
        parm = {
            'n_estimators': 1200,
            'max_depth': 3,
            'subsample': 0.5,
            'learning_rate': 0.01,
            'min_samples_leaf': 1,
            'random_state': 3,
        }
        self.classifier = GradientBoostingClassifier(**parm)
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_test, self.val_y_predicted,
                self.test_y_predicted)


class RfModel(BaseModel):
    model_type = 'Random Forest'

    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):
        print('training random forest...')
        self.classifier = RandomForestClassifier(n_estimators=500, class_weight=c_weight)
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_test, self.val_y_predicted, self.test_y_predicted)


#====================================================================================================

class TrainModel:

    def __init__(self, model_object):
        self.accuracies = []
        self.model_object = model_object()

    def print_model_type(self):
        print(self.model_object.model_type)

    # we train normally and get probabilities for the validation set. i.e., we use the probabilities to select the most uncertain samples

    def train(self, X_train, X_val, y_val, X_test,y_test, c_weight):
        print('Train set:', X_train.shape)
        print('Val   set:', X_val.shape)
        print('Test  set:', X_test.shape)
        t0 = time.time()
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
        (X_train, X_val, X_test, val_proba, M) = \
            self.model_object.fit_predict(X_train, X_val, y_val, X_test,y_test, c_weight)
        self.run_time = time.time() - t0
        return (X_train, X_val, X_test,val_proba, M)  # we return them in case we use PCA, with all the other algorithms, this is not needed.

    # we want accuracy only for the test set

    def get_test_accuracy(self, i, y_test):
        classif_rate = np.mean((self.test_y_predicted).ravel() == y_test.ravel()) * 100
        classif_rate = np.floor(classif_rate*100)/100
        self.accuracies.append(classif_rate)
        print('--------------------------------')
        print('Iteration:', i)
        print('--------------------------------')
        print('y-test set:', y_test.shape)
        print('Example run in %.3f s' % self.run_time, '\n')
        print("Accuracy rate for %f " % (classif_rate))
        # print("Classification report for classifier %s:\n%s\n" % (
        # self.model_object.classifier, metrics.classification_report(y_test, self.test_y_predicted)))
        # print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, self.test_y_predicted))
        logger.info('Confusion matrix:\n%s' % metrics.confusion_matrix(y_test, self.test_y_predicted) )
        print('--------------------------------')
        

#====================================================================================================

def get_k_random_samples(initial_labeled_samples, X_train_full,
                         y_train_full):
    random_state = check_random_state(0)
    np.random.seed(0)
    permutation = np.random.choice(y_train_full.shape[0],
                                   initial_labeled_samples,
                                   replace=False)
    print()
    print('initial random chosen samples', permutation.shape),
    #            permutation)
    permutation_list = permutation.tolist()
    X_train = X_train_full[permutation_list]
    y_train = y_train_full[permutation_list]
    # X_train = X_train.values
    # X_train = X_train.reshape((X_train.shape[0], -1))
    bin_count = np.bincount(y_train.astype('int64'))
    unique = np.unique(y_train.astype('int64'))
    print(
        'initial train set:',
        X_train.shape,
        y_train.shape,
        'unique(labels):',
        bin_count,
        unique,
    )
    return (permutation, X_train, y_train)


# ====================================================================================================

class BaseSelectionFunction(object):

    def __init__(self):
        pass

    def select(self):
        pass


class RandomSelection(BaseSelectionFunction):

    @staticmethod
    def select(probas_val, initial_labeled_samples,M):
        random_state = check_random_state(0)
        np.random.seed(0)
        selection = np.random.choice(probas_val.shape[0], initial_labeled_samples, replace=False)

        #     print('uniques chosen:',np.unique(selection).shape[0],'<= should be equal to:',initial_labeled_samples)

        return selection


class MinStdSelection(BaseSelectionFunction):

    # select the samples where the std is smallest - i.e., there is uncertainty regarding the relevant class
    # and then train on these "hard" to classify samples.

    @staticmethod
    def select(probas_val, initial_labeled_samples):
        std = np.std(probas_val * 100, axis=1)
        selection = std.argsort()[:initial_labeled_samples]
        selection = selection.astype('int64')

        #     print('std',std.shape,std)
        #     print()
        #     print('selection',selection, selection.shape, std[selection])

        return selection


class MarginSamplingSelection(BaseSelectionFunction):

    @staticmethod
    def select(probas_val, initial_labeled_samples):
        rev = np.sort(probas_val, axis=1)[:, ::-1]
        values = rev[:, 0] - rev[:, 1]
        selection = np.argsort(values)[:initial_labeled_samples]
        return selection

class Similar(BaseSelectionFunction):

    @staticmethod
    def select(probas_val, initial_labeled_samples,M):
        
        # probas_val[probas_val < M] = 0    
        # ibx=np.squeeze(np.nonzero(probas_val))
        # values=probas_val[np.nonzero(probas_val)]
        # if len(values) > 5:
            # k = min(len(values)-1, initial_labeled_samples)
        k = initial_labeled_samples
        selection = np.argpartition(probas_val, k)[:k] # Bottom-K
        # selection = ibx[idx]
            # selection = (np.argsort([x for x in probas_val if x < M])[::-1])[:initial_labeled_samples]
        return selection
        # else:
        #     return values
        
class BottomK(BaseSelectionFunction):

    @staticmethod
    def select(probas_val, initial_labeled_samples,M):
        
        probas_val[probas_val < M] = 0    
        ibx=np.squeeze(np.nonzero(probas_val))
        values=probas_val[np.nonzero(probas_val)]
        if len(values) > 5:
            k = min(len(values)-1, initial_labeled_samples)
            idx = np.argpartition(values, k)[:k] # Bottom-K
            selection = ibx[idx]
            # selection = (np.argsort([x for x in probas_val if x < M])[::-1])[:initial_labeled_samples]
            return selection
        else:
            return values
        
class TopK(BaseSelectionFunction):

    @staticmethod
    def select(probas_val, initial_labeled_samples,M):
        
        probas_val[probas_val > M] = 0    
        ibx=np.squeeze(np.nonzero(probas_val))
        values=probas_val[np.nonzero(probas_val)]
        if len(values) > 5:
            k = min(len(values)-1, initial_labeled_samples)
            idx = np.argpartition(values, -k)[-k:] # Top-K
            selection = ibx[idx]
            return selection
        else:
            return values
class EntropySelection(BaseSelectionFunction):

    @staticmethod
    def select(probas_val, initial_labeled_samples):
        e = (-probas_val * np.log2(probas_val)).sum(axis=1)
        selection = (np.argsort(e)[::-1])[:initial_labeled_samples]
        return selection


# ====================================================================================================

class Normalize(object):

    def normalize(self, X_train, X_val, X_test):
        self.scaler = MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        return (X_train, X_val, X_test)

    def inverse(self, X_train, X_val, X_test):
        X_train = self.scaler.inverse_transform(X_train)
        X_val = self.scaler.inverse_transform(X_val)
        X_test = self.scaler.inverse_transform(X_test)
        return (X_train, X_val, X_test)

    # ====================================================================================================


class TheAlgorithm(object):
    accuracies = []

    def __init__(self, initial_labeled_samples, model_object, selection_function):
        self.initial_labeled_samples = initial_labeled_samples
        self.model_object = model_object
        self.sample_selection_function = selection_function

    def run(self, X_train_full, y_train_full, X_test, y_test):
        # initialize process by applying base learner to labeled training data set to obtain Classifier

        (permutation, X_train, y_train) = get_k_random_samples(self.initial_labeled_samples,X_train_full, y_train_full)
        self.queried = self.initial_labeled_samples
        self.samplecount = [self.initial_labeled_samples]

        # permutation, X_train, y_train = get_equally_k_random_samples(self.initial_labeled_samples,classes)

        # assign the val set the rest of the 'unlabelled' training data

        X_val = np.array([])
        y_val = np.array([])
        X_val = np.copy(X_train_full)
        X_val = np.delete(X_val, permutation, axis=0)
        y_val = np.copy(y_train_full)
        y_val = np.delete(y_val, permutation, axis=0)
        print('val set:', X_val.shape, y_val.shape, permutation.shape)
        print()

        # normalize data

        # normalizer = Normalize()
        # X_train, X_val, X_test = normalizer.normalize(X_train, X_val, X_test)

        self.clf_model = TrainModel(self.model_object)
        (X_train, X_val, X_test,val_proba, M) = self.clf_model.train(X_train, X_val,y_val, X_test,y_test, 'balanced')
        active_iteration = 1
        # self.clf_model.get_test_accuracy(1, y_test) #later

        # fpfn = self.clf_model.test_y_predicted.ravel() != y_val.ravel()
        # print(fpfn)
        # self.fpfncount = []
        # self.fpfncount.append(fpfn.sum() / y_test.shape[0] * 100)

        while self.queried < max_queried:
            active_iteration += 1

            # get validation probabilities

            # probas_val = self.clf_model.model_object.classifier.predict_proba(X_val)
            
            # probas_val = np.array(val_proba) #later
            # print(probas_val.shape)
            # select samples using a selection function

            uncertain_samples = self.sample_selection_function.select(val_proba, self.initial_labeled_samples, M)
            # print("Now I'll see if I can break your score...")
            if len(uncertain_samples) < 6:
                logger.info('--------------------Not found enough informative samples-------------------' )
                break# normalization needs to be inversed and recalculated based on the new train and test set.

            # X_train, X_val, X_test = normalizer.inverse(X_train, X_val, X_test)

            # get the uncertain samples from the validation set

            # print('trainset before', X_train.shape, y_train.shape)
            logger.info('trainset before:{}'.format(X_train.shape) )
            X_train = np.concatenate((X_train, X_val[uncertain_samples]))
            y_train = np.concatenate((y_train, y_val[uncertain_samples]))
            # print('trainset after', X_train.shape, y_train.shape)
            logger.info('trainset after:{}'.format(X_train.shape) )
            self.samplecount.append(X_train.shape[0])

            bin_count = np.bincount(y_train.astype('int64'))
            unique = np.unique(y_train.astype('int64'))
            print(
                'updated train set:',
                X_train.shape,
                y_train.shape,
                'unique(labels):',
                bin_count,
                unique,
            )

            X_val = np.delete(X_val, uncertain_samples, axis=0)
            y_val = np.delete(y_val, uncertain_samples, axis=0)
            print('val set:', X_val.shape, y_val.shape)
            print()

            # normalize again after creating the 'new' train/test sets
            # normalizer = Normalize()
            # X_train, X_val, X_test = normalizer.normalize(X_train, X_val, X_test)

            self.queried += self.initial_labeled_samples
            (X_train, X_val, X_test,val_proba, M) = self.clf_model.train(X_train, X_val,y_val, X_test,y_test,'balanced')
            # self.clf_model.get_test_accuracy(active_iteration, y_test) #later

        # logger.info('final active learning accuracies: %s' %self.clf_model.accuracies )

#================================================================================================


def pickle_save(fname, data):
    filehandler = open(fname, "wb")
    pickle.dump(data, filehandler)
    filehandler.close()
    print('saved', fname, os.getcwd(), os.listdir())


def experiment(d, models, selection_functions, Ks, repeats, contfrom):
    # algos_temp = []
    print('stopping at:', max_queried)
    count = 0
    for model_object in models:
        if model_object.__name__ not in d:
            d[model_object.__name__] = {}

        for selection_function in selection_functions:
            if selection_function.__name__ not in d[model_object.__name__]:
                d[model_object.__name__][selection_function.__name__] = {}

            for k in Ks:
                d[model_object.__name__][selection_function.__name__][k] = []

                for i in range(0, repeats):
                    count += 1
                    if count >= contfrom:
                        print('Count = %s, using model = %s, selection_function = %s, k = %s, iteration = %s.' % (
                        count, model_object.__name__, selection_function.__name__, k, i))
                        alg = TheAlgorithm(k,
                                           model_object,
                                           selection_function
                                           )
                        alg.run(X_train_full, y_train_full, X_test, y_test)
                        d[model_object.__name__][selection_function.__name__][k].append(alg.clf_model.accuracies)
                        # fname = 'Active-learning-experiment-' + str(count) + '.pkl'
                        # pickle_save(fname, d)
                        # if count % 5 == 0:
                            # print(json.dumps(d, indent=2, sort_keys=True))
                        print()
                        print('---------------------------- FINISHED ---------------------------')
                        print()
    return d
#==============================================================================================

if __name__ == '__main__':

    # Set up logging    
    logging.basicConfig(level=logging.INFO)    
    logger = logging.getLogger()     
    logger.setLevel(logging.INFO)     
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')   
    log_file = './log_SVDD_rand_all.txt'     
    file_handler = logging.FileHandler(log_file)   
    file_handler.setLevel(logging.INFO)    
    file_handler.setFormatter(formatter)   
    logger.addHandler(file_handler)      
        
    # ==============================================================================
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    logger.info('Set seed to 0')
    max_queried = 500
    data_path = 'D:/Omid/Datasets/EuroSAT_RGB'
    # Categories=['Industrial','Residential']
    all_classes=['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial','Pasture','PermanentCrop','Residential','River','SeaLake']
    # for i in range(10):
    #     for j in range(10):
    #         if (i != j):
    #             Categories=[all_classes[i],all_classes[j]] 
    #             # Print arguments    
    #             logger.info('Classes include %s.' % Categories)
            
    #             (X, y) = download(data_path,Categories)
    #             # X=np.load('X.npy')
    #             # y=np.load('y.npy')
    #             X_train_full,X_test,y_train_full,y_test=train_test_split(X,y,test_size=0.50,random_state=0,stratify=y)
    #             # train_idx_normal = get_target_label_idx(X_train_full.target_ten.clone().data.cpu().numpy(), self.normal_classes)    
    #             X_train_full = X_train_full[np.where(y_train_full == 0)[0]]
    #             y_train_full = y_train_full[np.where(y_train_full == 0)[0]]
    #             print('train:', X_train_full.shape, y_train_full.shape)
    #             print('test :', X_test.shape, y_test.shape)
    #             classes = len(np.unique(y))
    #             print('unique classes', classes)
                
    #             repeats = 1
                
    #             models = [CNN]
                
    #             selection_functions = [BottomK]
                
    #             Ks = [100]
                
    #             d = {}
    #             stopped_at = -1
                
    #             d = experiment(d, models, selection_functions, Ks, repeats, stopped_at+1)
                
    X=np.load('X.npy')
    y=np.load('y.npy')           
    for i in range(10):

        logger.info('Classes include %s.' % all_classes[i])
        X_train_full,X_test,y_train_full,y_test=train_test_split(X,y,test_size=0.50,random_state=0,stratify=y)
        # train_idx_normal = get_target_label_idx(X_train_full.target_ten.clone().data.cpu().numpy(), self.normal_classes)    
        X_train_full = X_train_full[np.where(y_train_full == i)[0]]
        y_train_full = y_train_full[np.where(y_train_full == i)[0]]
        print('train:', X_train_full.shape, y_train_full.shape)
        X_test_in = X_test[np.where(y_test == i)[0]]
        X_test_out = X_test[np.where(y_test != i)[0]]
        X_test=np.concatenate((X_test_in,X_test_out),axis=0)
        y_test_in = y_test[np.where(y_test == i)[0]]
        y_test_in = np.zeros_like(y_test_in)
        y_test_out = np.ones(len(X_test_out))
        y_test=np.concatenate((y_test_in,y_test_out),axis=0)
        print('test :', X_test.shape, y_test.shape)
        classes = len(np.unique(y_test))
        print('unique classes', classes)
        repeats = 1
        # max_queried = len(y_train_full)
        models = [CNN]
        
        selection_functions = [ RandomSelection]
        
        Ks = [100]
        
        d = {}
        stopped_at = -1
        
        d = experiment(d, models, selection_functions, Ks, repeats, stopped_at+1)
    # results = json.loads(json.dumps(d, indent=2, sort_keys=True))
    # print(results)
    # with open('results_SVDD'+str(Categories[0])+'_vs_'+str(Categories[1])+'.json', 'w') as f:
    #     json.dump(results, f)

#===============================================================================================
# def performance_plot(fully_supervised_accuracy, dic, models, selection_functions, Ks, repeats):  
#     fig, ax = plt.subplots()
#     ax.plot([0,500],[fully_supervised_accuracy, fully_supervised_accuracy],label = 'algorithm-upper-bound')
#     for model_object in models:
#       for selection_function in selection_functions:
#         for idx, k in enumerate(Ks):
#             x = np.arange(float(Ks[idx]), 500 + float(Ks[idx]), float(Ks[idx]))            
#             Sum = np.array(dic[model_object][selection_function][k][0])
#             for i in range(1, repeats):
#                 Sum = Sum + np.array(dic[model_object][selection_function][k][i])
#             mean = Sum / repeats
#             ax.plot(x, mean ,label = model_object + '-' + selection_function + '-' + str(k))
#     ax.legend()
#     ax.set_xlim([50,500])
#     ax.set_ylim([40,100])
#     ax.grid(True)
#     plt.show()

# models_str = ['SvmModel', 'RfModel', 'LogModel']
# selection_functions_str = ['RandomSelection', 'MarginSamplingSelection', 'EntropySelection']
# Ks_str = ['250','125','50'] 
# repeats = 1
# random_forest_upper_bound = 97.
# svm_upper_bound = 94.
# log_upper_bound = 92.47
# total_experiments = len(models_str) * len(selection_functions_str) * len(Ks_str) * repeats

# print('So which is the better model? under the stopping condition and hyper parameters - random forest is the winner!')
# performance_plot(random_forest_upper_bound, d, ['RfModel'] , selection_functions_str    , Ks_str, 1)
# performance_plot(svm_upper_bound, d, ['SvmModel'] , selection_functions_str    , Ks_str, 1)
# performance_plot(log_upper_bound, d, ['LogModel'] , selection_functions_str    , Ks_str, 1)    
    
