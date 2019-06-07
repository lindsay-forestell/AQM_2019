# standard libraries
import numpy as np

# multiprocessing stuff
import multiprocessing
from multiprocessing import Pool
from itertools import product

# cross-validation
from sklearn.model_selection import KFold

# resampling
from sklearn.utils import resample

# from sklearn.linear_model import ElasticNet

# optimizer
from . import optimizers


class DataSet:

    def __init__(self, x, y, W):
        self.x = x
        self.y = y
        self.W = W
        self.x_sample = x
        self.y_sample = y
        self.W_sample = W

    def get_sample_estimates(self):
        self.x_sample, self.y_sample, self.W_sample = resample(self.x, self.y, self.W)

    def get_loss(self, lambda1, lambda2):

        pd = optimizers.CoordinateDescent(lambda1=lambda1, lambda2=lambda2, max_iter=10)
        #a = lambda1 / (2 * 90)
        #b = lambda2 / (2 * 90)
        #alpha = a + b
        #l1_ratio = a / (a + b)
        #model = ElasticNet(normalize=True, alpha=alpha, l1_ratio=l1_ratio)

        kf = KFold(n_splits = 10)
        loss_list = []

        for n_train, n_val in kf.split(self.x_sample):
            # split up data
            x_train = self.x_sample[n_train, :]
            x_val = self.x_sample[n_val, :]

            y_train = self.y_sample[n_train, :]
            y_val = self.y_sample[n_val, :]

            W_train = np.diag(self.W_sample[n_train, n_train])
            W_val = np.diag(self.W_sample[n_val, n_val])

            pd.fit(x_train, y_train, W_train, plot_flag=False)
            loss = pd.RSE(pd.scaler.transform(x_val)/np.sqrt(len(n_train)), y_val-pd.intercept, W_val)
            #model.fit(x_train, y_train)
            #y1 = y_val.reshape(-1,1)
            #y2 = model.predict(x_val).reshape(-1,1)
            #n_examples = y1.shape[0]
            #loss = (y1 - y2).T @ W_val @ (y1 - y2) / n_examples
            #loss = model.score(x_val, y_val)
            loss_list.append(loss)

        avg_loss = np.mean(loss_list)
        return lambda1, lambda2, avg_loss

    def get_bootstrap_estimates(self, lambda1, lambda2, sample_ids):
        data_idx = sample_ids[0]
        sample_id = sample_ids[1]
        x = self.x[data_idx, :]
        y = self.y[data_idx, :]
        W = np.diag(self.W)
        W = W[data_idx]
        W = np.diag(W)

        pd = optimizers.CoordinateDescent(lambda1=lambda1, lambda2=lambda2, max_iter=10)

        kf = KFold(n_splits = 10)
        loss_list = []

        for n_train, n_val in kf.split(x):
            # split up data
            x_train = x[n_train, :]
            x_val = x[n_val, :]

            y_train = y[n_train, :]
            y_val = y[n_val, :]

            W_train = np.diag(W[n_train, n_train])
            W_val = np.diag(W[n_val, n_val])

            pd.fit(x_train, y_train, W_train, plot_flag=False)
            loss = pd.RSE(pd.scaler.transform(x_val)/np.sqrt(len(n_train)), y_val-pd.intercept, W_val)
            #model.fit(x_train, y_train)
            #y1 = y_val.reshape(-1,1)
            #y2 = model.predict(x_val).reshape(-1,1)
            #n_examples = y1.shape[0]
            #loss = (y1 - y2).T @ W_val @ (y1 - y2) / n_examples
            #loss = model.score(x_val, y_val)
            loss_list.append(loss)

        avg_loss = np.mean(loss_list)
        return lambda1, lambda2, avg_loss, sample_id
