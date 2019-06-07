# standard stuff
import numpy as np

# extra stuff
from scipy.stats import multivariate_normal, t, norm
from scipy.special import gamma, digamma

# plotting stuff
import matplotlib.pyplot as plt


# helper functions


# Name: get_normal
# Purpose: Return value of multivariate Normal distribution
# In: x (array), mu (array), E (2D array)
# Out: (float)
def get_normal(x, mu, E):
    norm1 = 1.0 / ((2.0 * np.pi) ** (len(x) / 2.0) * np.sqrt(np.linalg.det(E)))
    delta = (x - mu)  # (1xN)
    exponent = np.dot(np.dot(delta, np.linalg.pinv(E)), np.transpose(delta))
    return norm1 * np.exp(-0.5 * exponent)


# Name: get_t_distribution
# Purpose: Return value of multivariate t distribution
# In: x (array), mu (array), E (2D array), v (scalar)
# Out: (float)
def get_t_distribution(x, mu, E, v):
    p = len(x)
    norm1 = gamma((v + p) / 2) / ((v * np.pi) ** (p / 2) * gamma(v / 2) * np.linalg.det(E))
    delta = (x - mu)  # (1xN)
    t_squared = np.dot(np.dot(delta, np.linalg.pinv(E)), np.transpose(delta))
    return norm1 * (1 + t_squared / v) ** (-(v + p) / 2)


# Name: normalize_data
# Purpose: normalize data in a dataframe in the range [0,1]
# In: df (Pandas Dataframe)
# Out: df (Pandas Dataframe)
def normalize_data(df):
    # Number of columns
    header = df.columns.values
    # Normalize column by column
    for col in header:
        cmax = np.max(df[col])
        cmin = np.min(df[col])
        df[col] = df[col].apply(lambda r: float(r - cmin) / float(cmax - cmin))
    return df


# Name: initialize_params
# Purpose: initialize the parameters
# and the number of clusters K
# In: N (int), K (int)
# Out: x (array), mu (array), E (2D array)
def initialize_params(N, K):
    # np.random.seed(26)
    # Initialize mean vectors (one for each cluster) (KxN)
    mu = np.random.rand(K, N)
    # Initialize mixing coefficients (one for each cluster) (1xK)
    # they must be normalized
    pi = np.random.rand(K)
    pi = pi / np.sum(pi)
    # Initialize covariance matrices (one for each cluster) (NxN)
    E = list(range(K))
    for k in range(K):
        E[k] = np.identity(N)
    return mu, E, pi


# Name: initialize_params
# Purpose: initialize the parameters
# and the number of clusters K
# In: P, number of parameters (int), K, number of clusters (int), N, initial amount of data
# Out: x (array), mu (array), E (2D array)
def initialize_params_t(P, K, N):
    # np.random.seed(26)
    # Initialize mean vectors (one for each cluster) (KxP)
    mu = np.random.rand(K, P)
    # Initialize mixing coefficients (one for each cluster) (1xK)
    # they must be normalized
    pi = np.random.rand(K)
    pi = pi / np.sum(pi)
    # Initialize covariance matrices (one for each cluster) (PxP)
    E = list(range(K))
    for k in range(K):
        E[k] = np.identity(P)
    # Initialze v, degrees of freedom (one for each cluster) (1xK)
    v = (N / K) * np.ones(K) - P
    return mu, E, pi, v


# Plotting Functions
def plot_2d_contours(model, distribution='gaussian', x1label='x1', x2label='x2', display=True, save=False):
    fig, ax = plt.subplots()

    # assumes 2d data, but any number of clusters
    # uses nice blends of colors for k = 1,2,3, otherwise chooses hard classes
    if model.k == 1:
        clist = [0]
    elif model.k == 2:
        clist = [0, 2]
    elif model.k == 3:
        clist = [0, 2, 1]

    if model.k <= 3:
        color = np.zeros((model.gamma_kn.shape[1], 3))
        color[:, clist] = model.gamma_kn.T
    else:
        color = np.argmax(model.gamma_kn, axis=0)

    x1 = np.arange(0, 1.01, 0.01)
    x2 = np.arange(0, 1.01, 0.01)

    X, Y = np.meshgrid(x1, x2)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # plot PDFs first
    cmap_list = ['Reds', 'Blues', 'Greens', 'Purples', 'Greys', 'Oranges',
                 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

    for kk in range(model.k):
        mu = model.mu_k[kk]
        variance = model.sigma_k[kk]

        if distribution == 'gaussian':
            rv = multivariate_normal(mu, variance)
            Z = rv.pdf(pos)
            min_val = 5

        if distribution == 't':
            min_val = 4
            v = model.v[kk]
            Z = np.empty(X.shape)
            for ii in range(X.shape[0]):
                for jj in range(X.shape[1]):
                    x_n = [X[ii, jj], Y[ii, jj]]
                    Z[ii, jj] = get_t_distribution(x_n, mu, variance, v)

        m = np.amax(Z)

        step = m / min_val
        levels = np.arange(0.0, m + step, step) + m / min_val  # step
        plt.contourf(X, Y, Z, levels, cmap=cmap_list[kk], alpha=0.7, vmin=step)

    plt.scatter(model.x[:, 0], model.x[:, 1], c=color, s=10)
    plt.scatter(model.mu_k[:, 0], model.mu_k[:, 1], c='y', marker='x', s=500)
    plt.xlabel(x1label)
    plt.ylabel(x2label)
    plt.title('n_iterations = {:d}'.format(model.current_iter))
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    # Used to return the plot as an image array
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    if display:
        plt.show()

    if save:
        return image


def plot_1d_contours(model, distribution='gaussian', x1label='x', display=True, save=False):
    fig, ax = plt.subplots()

    # assumes 1d data, but any number of clusters
    # chooses hard classes

    class_list = np.argmax(model.gamma_kn, axis=0)

    min_val = min(model.x)
    max_val = max(model.x)
    x = np.arange(min_val, max_val, 0.01)
    n_bins = 50
    step = (max_val - min_val) / n_bins
    bins = np.arange(min_val, max_val + step, step)

    # plot PDFs first
    colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k']
    x_individuals = []
    y_total = np.zeros(x.shape)
    for kk in range(model.k):
        mu = model.mu_k[kk]
        x_k = model.x[class_list == kk]
        x_individuals.append(x_k)
        variance = model.sigma_k[kk]
        pi = model.pi_k[kk]

        if distribution == 'gaussian':
            dist = norm(loc=mu, scale=np.sqrt(variance))
            y = dist.pdf(x).ravel()
            y1 = np.max(np.histogram(x_k, bins=bins)[0]) * y / max(y)

        if distribution == 't':
            v = model.v[kk]
            dist = t(loc=mu, scale=np.sqrt(variance), df=v)
            y = dist.pdf(x).ravel()
            y1 = np.max(np.histogram(x_k, bins=bins)[0]) * y / max(y)

        y_total += pi * y

        plt.plot(x, y1, c=colors[kk])
        # plt.hist(x_k, color=colors[kk], alpha=0.5, bins=bins)

    plt.hist(x_individuals, color=colors[0:model.k], alpha=0.5, bins=bins, stacked=True)
    plt.plot(x, np.max(np.histogram(model.x, bins=bins)[0]) * y_total / max(y_total), c='k')
    plt.xlabel(x1label)
    plt.ylabel('PDF')
    plt.title('n_iterations = {:d}'.format(model.current_iter))
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])

    # Used to return the plot as an image array
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    if display:
        plt.show()

    if save:
        plt.close()
        return image


def plot_convergence(model, plot_contours, distribution='gaussian'):
    nmax = model.current_iter + 1
    iter_ticks = list(range(0, nmax, int(nmax / 4)))

    print('==========================')
    print('Final Convergence Results.')
    print('==========================')

    # plot final contours
    plot_contours(model, distribution)

    # plot pi_k convergence
    # '''
    for kk in range(model.k):
        plt.plot(model.pi_k_list[kk], label='Cluster {:d}'.format(kk))
    plt.xlabel('n_iter')
    plt.ylabel('pi_k')
    plt.ylim([0, 1])
    plt.xticks(iter_ticks)
    plt.title('Convergence of $\pi_k$')
    plt.legend()
    plt.show()
    # '''

    # plot mu convergence
    # '''
    if model.p == 2:
        for kk in range(model.k):
            x = model.mu_k_list[kk][:, 0]
            y = model.mu_k_list[kk][:, 1]
            plt.scatter(x, y, label='Mean {:d}'.format(kk))
            plt.annotate('start', (x[0] + 0.01, y[0] + 0.01))
            plt.annotate('end', (x[-1] + 0.01, y[-1] + 0.01))
            plt.arrow(x[0], y[0], x[1] - x[0], y[1] - y[0], width=0.002, head_width=0.01, color='k')
            plt.arrow(x[-2], y[-2], x[-1] - x[-2], y[-1] - y[-2], width=0.002, head_width=0.01, color='k')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Convergence of $\mu_k$')
        plt.legend()
        plt.show()
    # '''

    # plot ll convergence
    # '''
    plt.plot(model.ll_list[1:])
    plt.xlabel('n_iter')
    plt.ylabel('log likelihood')
    plt.xticks(iter_ticks)
    plt.title('Convergence of log likelihood')
    # plt.yscale('log')
    plt.show()
    # '''

    # plot mu and sigma component convergence

    linestyles = ['-', '--', ':', '-.']
    colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k']
    # '''
    for kk in range(model.k):
        color = colors[kk % 7]
        for pp in range(model.p):
            linestyle = linestyles[pp % 4]
            mu_k_p = model.mu_k_list[kk][:, pp]
            plt.plot(mu_k_p, label='Cluster {:d}, Component {:d}'.format(kk, pp), color=color, linestyle=linestyle)
    plt.xlabel('n_iter')
    plt.ylabel('$x_p$')
    plt.xticks(iter_ticks)
    plt.title('Convergence of $\mu$ Components')
    plt.legend()
    plt.show()
    # '''

    # sigma
    # '''
    for kk in range(model.k):
        color = colors[kk % 7]
        for p1 in range(model.p):
            for p2 in range(p1, model.p):
                linestyle = linestyles[(p1 * model.p + p2) % 4]
                sigma_kpp_list = []
                for nn in range(1, model.current_iter):
                    sigma_kpp_list.append(model.sigma_k_list[nn][kk][p1, p2])

                plt.plot(sigma_kpp_list, label='Cluster {:d}, Component {:d}{:d}'.format(kk, p1, p2), color=color,
                         linestyle=linestyle)
    plt.xlabel('n_iter')
    plt.ylabel('$x_p$')
    plt.xticks(iter_ticks)
    plt.title('Convergence of $\Sigma$ Components')
    plt.legend()
    plt.show()
    # '''


# EM classes

class GMM:
    """
    Produce solutions for a gaussian mixture model
    Input: x, data to cluster
           k, number of clusters to search for
    """

    def __init__(self, x, k=2, normalize=True):
        # number of clusters, examples, and features
        # x is N x P dimensional
        self.k = k
        self.n = x.shape[0]
        self.p = x.shape[1]

        # normalize data
        if normalize:
            self.x = normalize_data(x)
            self.x = self.x.values
        else:
            self.x = x

        # initialize data
        mu_init, E_init, pi_init = initialize_params(self.p, self.k)

        self.pi_k = pi_init.reshape(-1, 1)  # (K x 1)
        self.mu_k = mu_init * np.max(self.x, axis=0, keepdims=True) + (1 - mu_init) * np.min(x, axis=0,
                                                                                             keepdims=True)  # (K x P)
        self.sigma_k = E_init  # (list of K matrices, each P x P)

        self.gamma_kn = np.zeros((self.k, self.n))  # (K x N)
        self.n_k = np.sum(self.gamma_kn, axis=1, keepdims=True)  # (K x 1)
        self.update_gamma_kn()

        # extra parameters for plotting
        self.current_iter = 0
        self.pi_k_list = []
        for ii, pi_k in enumerate(self.pi_k):
            self.pi_k_list.append([pi_k[0]])

        self.mu_k_list = []
        for kk, mu_k in enumerate(self.mu_k):
            self.mu_k_list.append([mu_k])

        self.sigma_k_list = [np.copy(self.sigma_k)]
        self.ll = self.log_likelihood()
        self.ll_list = [self.ll]
        self.converged = False

    def log_likelihood(self):
        ll = 0
        for nn in range(self.n):
            net_sum = 0
            x_n = self.x[nn, :]
            for kk in range(self.k):
                pi_k = self.pi_k[kk]
                mu_k = self.mu_k[kk, :]
                E_k = self.sigma_k[kk]
                normal_kn = get_normal(x_n, mu_k, E_k)
                net_sum += pi_k * normal_kn
            ll += np.log(net_sum)
        return ll

    def update_gamma_kn(self):
        for nn in range(self.n):
            denom = 0
            x_n = self.x[nn, :]
            for kk in range(self.k):
                pi_k = self.pi_k[kk]
                mu_k = self.mu_k[kk, :]
                E_k = self.sigma_k[kk]
                normal_kn = get_normal(x_n, mu_k, E_k)
                num = pi_k * normal_kn
                denom += num
                self.gamma_kn[kk, nn] = num

            self.gamma_kn[:, nn] = self.gamma_kn[:, nn] / denom

        self.n_k = np.sum(self.gamma_kn, axis=1, keepdims=True)

    def update_mu_k(self):
        # K x P
        num = self.gamma_kn @ self.x
        denom = self.n_k
        self.mu_k = num / denom

    def update_sigma_k(self):
        # list of K matrices, each P x P
        for kk in range(self.k):
            sigma_k = np.zeros((self.p, self.p))
            for nn in range(self.n):
                x_m = self.x[nn, :].reshape(-1, 1)
                mu_k = self.mu_k[kk, :].reshape(-1, 1)
                sigma_k += self.gamma_kn[kk, nn] * ((x_m - mu_k) @ (x_m - mu_k).T)
            sigma_k = sigma_k / self.n_k[kk]
            self.sigma_k[kk] = sigma_k

    def update_pi_k(self):
        # K x 1
        self.pi_k = self.n_k / self.n

    def fit(self, n_iter=10, conv_tol=1e-5):
        if self.converged:
            print('Already converged.')
            return
        # does n_iter further iterations of the fit, unless log-likelihood stops changing (according to conv_tol)
        for ii in range(n_iter):
            self.update_gamma_kn()

            self.update_mu_k()
            for kk in range(self.k):
                self.mu_k_list[kk] = np.vstack((self.mu_k_list[kk], self.mu_k[kk]))

            self.update_sigma_k()
            self.sigma_k_list.append(np.copy(self.sigma_k))

            self.update_pi_k()
            for kk in range(self.k):
                self.pi_k_list[kk].append(self.pi_k[kk][0])

            self.ll = self.log_likelihood()
            self.ll_list.append(self.ll)

            self.current_iter += 1

            # check for convergence
            ll = self.ll
            ll_old = self.ll_list[-2]
            if abs((ll - ll_old) / ll_old) < conv_tol:
                print('====================================================')
                print('Convergence has been reached. Iterations terminated.')
                print('====================================================')
                self.converged = True
                break


class TMM:
    """
    Produce solutions for a t-distribution mixture model
    Input: x, data to cluster
           k, number of clusters to search for
    """

    def __init__(self, x, k=2, vlist=[]):
        # number of clusters, examples, and features
        # x is N x P dimensional
        self.k = k
        self.n = x.shape[0]
        self.p = x.shape[1]
        self.x = x

        # normalize data
        # self.x = normalize_data(x)
        # self.x = self.x.values

        # initialize data
        mu_init, E_init, pi_init, v_init = initialize_params_t(self.p, self.k, self.n)

        self.pi_k = pi_init.reshape(-1, 1)  # (K x 1)
        self.mu_k = mu_init * np.max(self.x, axis=0, keepdims=True) + (1 - mu_init) * np.min(x, axis=0,
                                                                                             keepdims=True)  # (K x P)
        self.sigma_k = E_init  # (list of K matrices, each P x P)
        if len(vlist) < self.k:
            self.v = v_init  # (leave as is for now, list of numbers)
        else:
            self.v = vlist

        self.gamma_kn = np.zeros((self.k, self.n))  # (K x N)
        self.n_k = np.sum(self.gamma_kn, axis=1, keepdims=True)  # (K x 1)
        self.u_kn = np.zeros((self.k, self.n))  # (K x N)
        self.update_gamma_u_kn()

        # extra parameters for plotting
        self.current_iter = 0
        self.pi_k_list = []
        for ii, pi_k in enumerate(self.pi_k):
            self.pi_k_list.append([pi_k[0]])

        self.mu_k_list = []
        for kk, mu_k in enumerate(self.mu_k):
            self.mu_k_list.append([mu_k])

        self.sigma_k_list = [np.copy(self.sigma_k)]
        self.ll = self.log_likelihood()
        self.ll_list = [self.ll]
        self.converged = False

    def log_likelihood(self):
        ll = 0
        for nn in range(self.n):
            net_sum = 0
            x_n = self.x[nn, :]
            for kk in range(self.k):
                pi_k = self.pi_k[kk]
                mu_k = self.mu_k[kk, :]
                E_k = self.sigma_k[kk]
                v_k = self.v[kk]
                t_kn = get_t_distribution(x_n, mu_k, E_k, v_k)
                net_sum += pi_k * t_kn
            ll += np.log(net_sum)
        return ll

    def update_gamma_u_kn(self):
        for nn in range(self.n):
            denom = 0
            x_n = self.x[nn, :]
            for kk in range(self.k):
                # update gamma
                pi_k = self.pi_k[kk]
                mu_k = self.mu_k[kk, :]
                E_k = self.sigma_k[kk]
                v_k = self.v[kk]
                t_dist_kn = get_t_distribution(x_n, mu_k, E_k, v_k)
                num = pi_k * t_dist_kn
                denom += num
                self.gamma_kn[kk, nn] = num

                # update u
                delta = x_n - mu_k
                t_squared = np.dot(np.dot(delta, np.linalg.pinv(E_k)), np.transpose(delta))
                self.u_kn[kk, nn] = (v_k + self.p) / (v_k + t_squared)

            self.gamma_kn[:, nn] = self.gamma_kn[:, nn] / denom

        # update n_k
        self.n_k = np.sum(self.gamma_kn, axis=1, keepdims=True)

    def update_mu_k(self):
        # K x P
        beta_kn = self.gamma_kn * self.u_kn
        num = beta_kn @ self.x
        denom = np.sum(beta_kn, axis=1, keepdims=True)
        self.mu_k = num / denom

    def update_sigma_k(self):
        # list of K matrices, each P x P
        beta_kn = self.gamma_kn * self.u_kn
        for kk in range(self.k):
            sigma_k = np.zeros((self.p, self.p))
            for nn in range(self.n):
                x_m = self.x[nn, :].reshape(-1, 1)
                mu_k = self.mu_k[kk, :].reshape(-1, 1)
                sigma_k += beta_kn[kk, nn] * ((x_m - mu_k) @ (x_m - mu_k).T)
            sigma_k = sigma_k / self.n_k[kk]
            self.sigma_k[kk] = sigma_k

    def update_pi_k(self):
        # K x 1
        self.pi_k = self.n_k / self.n

    def fit(self, n_iter=10, conv_tol=1e-5):
        if self.converged:
            print('Already converged.')
            return
        # does n_iter further iterations of the fit, unless log-likelihood stops changing (according to conv_tol)
        for ii in range(n_iter):
            self.update_gamma_u_kn()

            self.update_mu_k()
            for kk in range(self.k):
                self.mu_k_list[kk] = np.vstack((self.mu_k_list[kk], self.mu_k[kk]))

            self.update_sigma_k()
            self.sigma_k_list.append(np.copy(self.sigma_k))

            self.update_pi_k()
            for kk in range(self.k):
                self.pi_k_list[kk].append(self.pi_k[kk][0])

            self.ll = self.log_likelihood()
            self.ll_list.append(self.ll)

            self.current_iter += 1

            # check for convergence
            ll = self.ll
            ll_old = self.ll_list[-2]
            if abs((ll - ll_old) / ll_old) < conv_tol:
                print('====================================================')
                print('Convergence has been reached. Iterations terminated.')
                print('====================================================')
                self.converged = True
                break
