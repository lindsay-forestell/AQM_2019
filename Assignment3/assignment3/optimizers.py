# Standard Libraries
import numpy as np
# import pandas as pd

# Plotting
import matplotlib.pyplot as plt
# import matplotlib
# import seaborn as sns

# Extra
from numpy import sign
from sklearn.preprocessing import StandardScaler
import time


class SubGradientDescent:
    """
    Method to calculate the optimal WEN solution using sub-gradient descent
    Input parameters: lambda1, lambda2, max_iter, learning_rate, scale (if data needs scaling), fit_intercept
    Functions: RSE (calculates 1/n(y-yhat)^2)
               loss (calculates full loss being optimized)
               sparsity (calculates number of zero'd features)
               timer (returns how long the fit took)
               gradient (calculates the gradient for a given beta)
               fit(x_train,y_train,x_test,y_test,W_test,W, plot_flag)
                  note that x_test, y_test, W_test are optional
               predict(x_test)
    """

    def __init__(self, lambda1=1e-3, lambda2=1e-3, max_iter=100, learning_rate=1e-5, scale=True, fit_intercept=True):
        self.beta = np.empty(0)
        self.intercept = 0
        self.scaler = StandardScaler()
        self.yscaler = StandardScaler(with_std=False)
        self.scale = scale
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.timer = 0
        self.iter_best = 0
        self.fit_intercept = fit_intercept

    def RSE(self, X, y, beta):
        y = y.reshape(-1, 1)
        n_examples = y.shape[0]
        rse = (y - X @ beta).T @ (y - X @ beta) / n_examples
        return rse

    def loss(self, X, y, W, beta):
        y = y.reshape(-1, 1)
        A = 2 * (X.T @ W @ X + self.lambda2 * np.eye(beta.shape[0])) / (1 + self.lambda2)
        b = 2 * X.T @ W @ y
        loss = y.T @ W @ y + (1 / 2) * beta.T @ A @ beta - b.T @ beta + self.lambda1 * np.sum(abs(beta))
        return loss

    def sparsity(self, atol=1e-5):
        n_zero = np.sum(np.isclose(self.beta, 0, atol=atol))
        return n_zero.astype(int)

    def timer(self):
        return self.timer

    def gradient(self, X, y, W, beta):
        A = 2 * (X.T @ W @ X + self.lambda2 * np.eye(beta.shape[0])) / (1 + self.lambda2)
        b = 2 * X.T @ W @ y
        dbeta = A @ beta - b + self.lambda1 * sign(beta)
        return dbeta

    def fit(self, x_train, y_train, W_train=1, x_test=1, y_test=1, W_test=1, plot_flag=True):
        # **kwargs = (x_test,y_test,w_test)
        start = time.time()
        # Add x_test and x_train:
        try:
            if (x_test == 1):
                x_test = x_train
                y_test = y_train
        except:
            x_test = x_test
            y_test = y_test

        # Number of examples, parameters
        n_train = y_train.shape[0]
        n_test = y_test.shape[0]
        n_params = x_train.shape[1]

        # Make W a matrix:
        try:
            if W_train == 1:
                W_train = np.eye(n_train)
        except:
            W_train = W_train

        try:
            if W_test == 1:
                W_test = np.eye(n_test)
        except:
            W_test = W_test

        # Make sure y is right shape
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        # Scale the data according to training data
        # Note that x needs to have \sum x_ij^2 = 1, not sigma = 1
        if self.scale:
            x_train = self.scaler.fit_transform(x_train) / np.sqrt(n_train)
            x_test = self.scaler.transform(x_test) / np.sqrt(n_train)

        # can either choose to scale y or not: fit_intercept assumes it is not scaled
        if not self.fit_intercept:
            y_train = self.yscaler.fit_transform(y_train)
            y_test = self.yscaler.transform(y_test)

        # Add column of ones to x data
        x_initial = np.c_[np.ones(n_train), x_train]
        # x_test = np.c_[np.ones(n_test), x_test]

        # Initialize beta
        # beta = 0 * np.random.uniform(low=-0.1, high=0.1, size=(n_params + 1, 1))
        # self.beta = np.copy(beta)
        # Initialize beta (good guess)
        # self.intercept = np.mean(y_train)
        # A = 2 * (x_train.T @ W_train @ x_train + self.lambda2*np.eye(n_params)) / (1 + self.lambda2)
        # b = 2 * x_train.T @ W_train @ (y_train-self.intercept)
        # beta = np.linalg.pinv(A) @ b

        A = 2 * (x_initial.T @ W_train @ x_initial + self.lambda2 * np.eye(n_params + 1)) / (1 + self.lambda2)
        b = 2 * x_initial.T @ W_train @ (y_train)
        beta = np.linalg.pinv(A) @ b
        self.intercept = beta[0, 0] / (1 + self.lambda2)
        beta = beta[1:, 0].reshape(-1, 1)

        self.beta = np.copy(beta)
        f_best = 1e9*self.loss(x_train, y_train - self.intercept, W_train, beta)
        self.iter_best = 0

        # Initialize cost vector and iteration vector for later plotting
        iter_vec = np.array(range(self.max_iter + 1))
        rse_vec_train = np.arange(self.max_iter + 1).astype('float')
        rse_vec_test = np.arange(self.max_iter + 1).astype('float')
        loss_vec_train = np.arange(self.max_iter + 1).astype('float')
        loss_vec_test = np.arange(self.max_iter + 1).astype('float')

        # Calculate the value of the RSE and loss
        rse_train = self.RSE(x_train, y_train - self.intercept, beta)
        rse_test = self.RSE(x_test, y_test - self.intercept, beta)

        loss_train = self.loss(x_train, y_train - self.intercept, W_train, beta)
        loss_test = self.loss(x_test, y_test - self.intercept, W_test, beta)

        # Save cost for later plotting
        rse_vec_train[0] = rse_train
        rse_vec_test[0] = rse_test

        loss_vec_train[0] = loss_train / n_train
        loss_vec_test[0] = loss_test / n_test

        for ii in range(self.max_iter):

            # Calculate the gradient
            dbeta = self.gradient(x_train, y_train - self.intercept, W_train, beta)

            # Update intercept
            self.intercept = np.sum(y_train - x_train @ beta) / n_train

            # Update beta
            beta = beta - self.learning_rate * dbeta
            f_test = self.loss(x_train, y_train - self.intercept, W_train, beta)
            if f_test < f_best:
                f_best = f_test
                self.beta = np.copy(beta)
                self.iter_best = ii

            # Calculate the value of the RSE and loss
            rse_train = self.RSE(x_train, y_train - self.intercept, beta)
            rse_test = self.RSE(x_test, y_test - self.intercept, beta)

            loss_train = self.loss(x_train, y_train - self.intercept, W_train, beta)
            loss_test = self.loss(x_test, y_test - self.intercept, W_test, beta)

            # Save cost for later plotting
            rse_vec_train[ii + 1] = rse_train
            rse_vec_test[ii + 1] = rse_test

            loss_vec_train[ii + 1] = loss_train / n_train
            loss_vec_test[ii + +1] = loss_test / n_test

        end = time.time()
        self.timer = end - start

        # Plot final solution
        if plot_flag:
            fig, ax1 = plt.subplots()
            plt.plot(iter_vec, rse_vec_test, 'r', label='Test RSE')
            plt.plot(iter_vec, rse_vec_train, 'b', label='Train RSE')
            plt.xlabel('Epoch')
            plt.ylabel('RSE')
            #plt.yscale('log')
            # plt.xscale('log')
            plt.title(r'$\lambda_1$ = {0:0.1e}      $\lambda_2$ = {1:0.1e}'.format(self.lambda1, self.lambda2))
            plt.legend()

            ax2 = ax1.twinx()
            ax2.plot(iter_vec, loss_vec_test, '--r', label='Test Loss')
            ax2.plot(iter_vec, loss_vec_train, '--b', label='Train Loss')
            ax2.set_ylabel('Loss')
            ax2.legend(loc='upper center')
            #ax2.set_yscale('log')

            plt.show()

    def predict(self, X):
        # Scale new data
        if self.scale:
            X = self.scaler.transform(X) / np.sqrt(self.scaler.n_samples_seen_)
        # Add the column of ones
        # X = np.c_[np.ones(X.shape[0]), X]
        # Transform to get new y
        y_pred = X @ self.beta + self.intercept
        if not self.fit_intercept:
            y_pred = self.yscaler.inverse_transform(y_pred)
        return y_pred


class CoordinateDescent:
    """
    Method to calculate the optimal WEN solution using coordinate descent
    Input parameters: lambda1, lambda2, max_iter, scale (if data needs scaling)
    Functions: RSE (calculates 1/n(y-yhat)^2)
               loss (calculates full loss being optimized)
               sparsity (calculates number of zero'd features)
               timer (returns how long the fit took)
               gradient (calculates the gradient for a given beta)
               fit(x_train,y_train,x_test,y_test,W_test,W, plot_flag)
                  note that x_test, y_test, W_test are optional
               predict(x_test)
    """

    def __init__(self, lambda1=1e-3, lambda2=1e-3, max_iter=100, scale=True, fit_intercept=True):
        self.beta = np.empty(0)
        self.intercept = 0
        self.scaler = StandardScaler()
        self.yscaler = StandardScaler(with_std=False)
        self.scale = scale
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.timer = 0
        self.fit_intercept = fit_intercept

    def RSE(self, X, y, W):
        y = y.reshape(-1, 1)
        n_examples = y.shape[0]
        rse = (y - X @ self.beta).T @ W @ (y - X @ self.beta) / n_examples
        return rse

    def loss(self, X, y, W):
        y = y.reshape(-1, 1)
        A = 2 * (X.T @ W @ X + self.lambda2 * np.eye(self.beta.shape[0])) / (1 + self.lambda2)
        b = 2 * X.T @ W @ y
        return (y.T @ W @ y) + (1 / 2) * self.beta.T @ A @ self.beta - (b.T @ self.beta) + self.lambda1 * np.sum(
            abs(self.beta))

    def sparsity(self, atol=1e-5):
        n_zero = np.sum(np.isclose(self.beta, 0, atol=atol))
        return n_zero.astype(int)

    def timer(self):
        return self.timer

    def single_loop(self, X, y, W):
        # update betas
        n = y.shape[0]
        A = 2 * (X.T @ W @ X + self.lambda2 * np.eye(self.beta.shape[0])) / (1 + self.lambda2)
        b = 2 * X.T @ W @ y
        beta_idx = np.arange(self.beta.shape[0])
        np.random.shuffle(beta_idx)
        for jj in beta_idx:
            aj = A[jj, jj]
            A_no_j = np.delete(A, jj, axis=1)
            beta_no_j = np.delete(self.beta, jj, axis=0)
            cj = -A_no_j[jj, :] @ beta_no_j + b[jj, 0]
            if cj < -self.lambda1:
                self.beta[jj, 0] = (cj + self.lambda1) / aj
            elif cj <= self.lambda1:
                self.beta[jj, 0] = 0
            else:
                self.beta[jj, 0] = (cj - self.lambda1) / aj

    def fit(self, x_train, y_train, W_train=1, x_test=1, y_test=1, W_test=1, plot_flag=True):
        start = time.time()
        # Add x_test and x_train:
        try:
            if (x_test == 1):
                x_test = x_train
                y_test = y_train
        except:
            x_test = x_test
            y_test = y_test

        # Number of examples, parameters
        n_train = y_train.shape[0]
        n_test = y_test.shape[0]
        n_params = x_train.shape[1]

        # Make W a matrix:
        try:
            if W_train == 1:
                W_train = np.eye(n_train)
        except:
            W_train = W_train

        try:
            if W_test == 1:
                W_test = np.eye(n_test)
        except:
            W_test = W_test

        # Make sure y is right shape
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        # Scale the data according to training data
        # Note that x needs to have \sum x_ij^2 = 1, not sigma = 1
        if self.scale:
            x_train = self.scaler.fit_transform(x_train) / np.sqrt(n_train)
            x_test = self.scaler.transform(x_test) / np.sqrt(n_train)

        # can either choose to scale y or not: fit_intercept assumes it is not scaled
        if not self.fit_intercept:
            y_train = self.yscaler.fit_transform(y_train)
            y_test = self.yscaler.transform(y_test)

        # Add column of ones to x data
        x_initial = np.c_[np.ones(n_train), x_train]
        # x_test = np.c_[np.ones(n_test), x_test]

        # Initialize beta
        # self.intercept = np.mean(y_train)
        # A = 2 * (x_train.T @ W @ x_train + self.lambda2*np.eye(n_params)) / (1 + self.lambda2)
        # b = 2 * x_train.T @ W @ (y_train-self.intercept)
        # self.beta = np.linalg.pinv(A) @ b

        A = 2 * (x_initial.T @ W_train @ x_initial + self.lambda2 * np.eye(n_params + 1)) / (1 + self.lambda2)
        b = 2 * x_initial.T @ W_train @ (y_train)
        beta = np.linalg.pinv(A) @ b
        self.intercept = beta[0, 0] / (1 + self.lambda2)
        self.beta = beta[1:, 0].reshape(-1, 1)

        # Initialize cost vector and iteration vector for later plotting
        iter_vec = np.array(range(self.max_iter + 1))
        rse_vec_train = np.arange(self.max_iter + 1).astype('float')
        rse_vec_test = np.arange(self.max_iter + 1).astype('float')
        loss_vec_train = np.arange(self.max_iter + 1).astype('float')
        loss_vec_test = np.arange(self.max_iter + 1).astype('float')

        # Calculate the value of the RSE and loss
        rse_train = self.RSE(x_train, y_train - self.intercept, W_train)
        rse_test = self.RSE(x_test, y_test - self.intercept, W_test)

        loss_train = self.loss(x_train, y_train - self.intercept, W_train)
        loss_test = self.loss(x_test, y_test - self.intercept, W_test)

        # Save cost for later plotting
        rse_vec_train[0] = rse_train
        rse_vec_test[0] = rse_test

        loss_vec_train[0] = loss_train / n_train
        loss_vec_test[0] = loss_test / n_test

        for ii in range(self.max_iter):
            # Update beta
            self.single_loop(x_train, y_train - self.intercept, W_train)

            # Update intercept
            self.intercept = np.sum(y_train - x_train @ self.beta) / n_train

            # Calculate the value of the RSE and loss
            rse_train = self.RSE(x_train, y_train - self.intercept, W_train)
            rse_test = self.RSE(x_test, y_test - self.intercept, W_test)

            loss_train = self.loss(x_train, y_train - self.intercept, W_train)
            loss_test = self.loss(x_test, y_test - self.intercept, W_test)

            # Save cost for later plotting
            rse_vec_train[ii + 1] = rse_train
            rse_vec_test[ii + 1] = rse_test

            loss_vec_train[ii + 1] = loss_train / n_train
            loss_vec_test[ii + 1] = loss_test / n_test

        end = time.time()
        self.timer = end - start

        # Plot final solution
        if plot_flag:
            fig, ax1 = plt.subplots()
            plt.plot(iter_vec, rse_vec_test, 'r', label='Test RSE')
            plt.plot(iter_vec, rse_vec_train, 'b', label='Train RSE')
            plt.xlabel('Epoch')
            plt.ylabel('RSE')
            #plt.yscale('log')
            # plt.xscale('log')
            plt.title(r'$\lambda_1$ = {0:0.1e}      $\lambda_2$ = {1:0.1e}'.format(self.lambda1, self.lambda2))
            plt.legend()

            ax2 = ax1.twinx()
            ax2.plot(iter_vec, loss_vec_test, '--r', label='Test Loss')
            ax2.plot(iter_vec, loss_vec_train, '--b', label='Train Loss')
            ax2.set_ylabel('Loss')
            ax2.legend(loc='upper center')
            #ax2.set_yscale('log')

            plt.show()

    def predict(self, X):
        # Scale new data
        if self.scale:
            X = self.scaler.transform(X) / np.sqrt(self.scaler.n_samples_seen_)
        # Add the column of ones
        # X = np.c_[np.ones(X.shape[0]), X]
        # Transform to get new y
        y_pred = X @ self.beta + self.intercept
        if not self.fit_intercept:
            y_pred = self.yscaler.inverse_transform(y_pred)
        return y_pred


class ProximalDescent:
    """
    Method to calculate the optimal WEN solution using proximal descent
    Input parameters: lambda1, lambda2, max_iter, scale (if data needs scaling)
    Functions: RSE (calculates 1/n(y-yhat)^2)
               loss (calculates full loss being optimized)
               sparsity (calculates number of zero'd features)
               timer (returns how long the fit took)
               gradient (calculates the gradient for a given beta)
               fit(x_train,y_train,x_test,y_test,W_test,W, plot_flag)
                  note that x_test, y_test, W_test are optional
               predict(x_test)
    """

    def __init__(self, lambda1=1e-3, lambda2=1e-3, max_iter=200, learning_rate=1e-4, scale=True, fit_intercept=True):
        self.beta = np.empty(0)
        self.intercept = 0
        self.scaler = StandardScaler()
        self.yscaler = StandardScaler(with_std=False)
        self.scale = scale
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.timer = 0
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate

    def RSE(self, X, y, W):
        y = y.reshape(-1, 1)
        n_examples = y.shape[0]
        rse = (y - X @ self.beta).T @ W @ (y - X @ self.beta) / n_examples
        return rse

    def loss(self, X, y, W):
        y = y.reshape(-1, 1)
        A = 2 * (X.T @ W @ X + self.lambda2 * np.eye(self.beta.shape[0])) / (1 + self.lambda2)
        b = 2 * X.T @ W @ y
        return (y.T @ W @ y) + (1 / 2) * self.beta.T @ A @ self.beta - (b.T @ self.beta) + self.lambda1 * np.sum(
            abs(self.beta))

    def soft(self, a, b):
        return sign(a) * np.maximum(abs(a) - b, 0)

    def sparsity(self, atol=1e-5):
        n_zero = np.sum(np.isclose(self.beta, 0, atol=atol))
        return n_zero.astype(int)

    def timer(self):
        return self.timer

    def gradient(self, X, y, W):
        A = 2 * (X.T @ W @ X + self.lambda2 * np.eye(self.beta.shape[0])) / (1 + self.lambda2)
        b = 2 * X.T @ W @ y
        dbeta = A @ self.beta - b
        return dbeta

    def fit(self, x_train, y_train, W_train=1, x_test=1, y_test=1, W_test=1, plot_flag=True):
        start = time.time()
        # Add x_test and x_train:
        try:
            if (x_test == 1):
                x_test = x_train
                y_test = y_train
        except:
            x_test = x_test
            y_test = y_test

        # Number of examples, parameters
        n_train = y_train.shape[0]
        n_test = y_test.shape[0]
        n_params = x_train.shape[1]

        # Make W a matrix:
        try:
            if W_train == 1:
                W_train = np.eye(n_train)
        except:
            W_train = W_train

        try:
            if W_test == 1:
                W_test = np.eye(n_test)
        except:
            W_test = W_test

        # Make sure y is right shape
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        # Scale the data according to training data
        # Note that x needs to have \sum x_ij^2 = 1, not sigma = 1
        if self.scale:
            x_train = self.scaler.fit_transform(x_train) / np.sqrt(n_train)
            x_test = self.scaler.transform(x_test) / np.sqrt(n_train)

        # can either choose to scale y or not: fit_intercept assumes it is not scaled
        if not self.fit_intercept:
            y_train = self.yscaler.fit_transform(y_train)
            y_test = self.yscaler.transform(y_test)

        # Add column of ones to x data
        x_initial = np.c_[np.ones(n_train), x_train]
        # x_test = np.c_[np.ones(n_test), x_test]

        # Initialize beta
        # self.intercept = np.mean(y_train)
        # A = 2 * (x_train.T @ W @ x_train + self.lambda2*np.eye(n_params)) / (1 + self.lambda2)
        # b = 2 * x_train.T @ W @ (y_train-self.intercept)
        # self.beta = np.linalg.pinv(A) @ b

        A = 2 * (x_initial.T @ W_train @ x_initial + self.lambda2 * np.eye(n_params + 1)) / (1 + self.lambda2)
        b = 2 * x_initial.T @ W_train @ (y_train)
        beta = np.linalg.pinv(A) @ b
        self.intercept = beta[0, 0] / (1 + self.lambda2)
        self.beta = beta[1:, 0].reshape(-1, 1)

        # Initialize cost vector and iteration vector for later plotting
        iter_vec = np.array(range(self.max_iter + 1))
        rse_vec_train = np.arange(self.max_iter + 1).astype('float')
        rse_vec_test = np.arange(self.max_iter + 1).astype('float')
        loss_vec_train = np.arange(self.max_iter + 1).astype('float')
        loss_vec_test = np.arange(self.max_iter + 1).astype('float')

        # Calculate the value of the RSE and loss
        rse_train = self.RSE(x_train, y_train - self.intercept, W_train)
        rse_test = self.RSE(x_test, y_test - self.intercept, W_test)

        loss_train = self.loss(x_train, y_train - self.intercept, W_train)
        loss_test = self.loss(x_test, y_test - self.intercept, W_test)

        # Save cost for later plotting
        rse_vec_train[0] = rse_train
        rse_vec_test[0] = rse_test

        loss_vec_train[0] = loss_train / n_train
        loss_vec_test[0] = loss_test / n_test

        for ii in range(self.max_iter):
            # Calculate the gradient
            dbeta = self.gradient(x_train, y_train - self.intercept, W_train)

            # Update intercept
            self.intercept = np.sum(y_train - x_train @ self.beta) / n_train

            # Update beta
            u_k = self.beta - self.learning_rate * dbeta
            self.beta = self.soft(u_k, self.learning_rate * self.lambda1)

            # Calculate the value of the RSE and loss
            rse_train = self.RSE(x_train, y_train - self.intercept, W_train)
            rse_test = self.RSE(x_test, y_test - self.intercept, W_test)

            loss_train = self.loss(x_train, y_train - self.intercept, W_train)
            loss_test = self.loss(x_test, y_test - self.intercept, W_test)

            # Save cost for later plotting
            rse_vec_train[ii + 1] = rse_train
            rse_vec_test[ii + 1] = rse_test

            loss_vec_train[ii + 1] = loss_train / n_train
            loss_vec_test[ii + 1] = loss_test / n_test

        end = time.time()
        self.timer = end - start

        # Plot final solution
        if plot_flag:
            fig, ax1 = plt.subplots()
            plt.plot(iter_vec, rse_vec_test, 'r', label='Test RSE')
            plt.plot(iter_vec, rse_vec_train, 'b', label='Train RSE')
            plt.xlabel('Epoch')
            plt.ylabel('RSE')
            #plt.yscale('log')
            # plt.xscale('log')
            plt.title(r'$\lambda_1$ = {0:0.1e}      $\lambda_2$ = {1:0.1e}'.format(self.lambda1, self.lambda2))
            plt.legend()

            ax2 = ax1.twinx()
            ax2.plot(iter_vec, loss_vec_test, '--r', label='Test Loss')
            ax2.plot(iter_vec, loss_vec_train, '--b', label='Train Loss')
            ax2.set_ylabel('Loss')
            #ax2.set_yscale('log')
            ax2.legend(loc='upper center')

            plt.show()

    def predict(self, X):
        # Scale new data
        if self.scale:
            X = self.scaler.transform(X) / np.sqrt(self.scaler.n_samples_seen_)
        # Add the column of ones
        # X = np.c_[np.ones(X.shape[0]), X]
        # Transform to get new y
        y_pred = X @ self.beta + self.intercept
        if not self.fit_intercept:
            y_pred = self.yscaler.inverse_transform(y_pred)
        return y_pred
