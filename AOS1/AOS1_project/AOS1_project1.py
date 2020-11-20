# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from numpy import *
from numpy.linalg import inv

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from scipy.stats import multivariate_normal
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, MetaEstimatorMixin

from sklearn.utils import check_X_y


######################################
# Quadratic Logistic Regression model#
######################################
def RQLR(penalty, solver='liblinear', degree=2, l1_ratio=None):
    """Generate Regularized Quadratic Logistic Regression model"""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    cls = LogisticRegression(max_iter=2000, solver=solver, penalty=penalty, l1_ratio=l1_ratio)
    pipe = make_pipeline(poly, cls)
    return pipe


##################################
# Gaussian Discriminant Analysis #
##################################
class GDA:
    def __init__(self): pass

    def fit(self, X, y, **kwargs):
        ''' Fisrt function(for training)
        Training with Gaussian DA
        the maximum-likelihood estimated mu and sigma by mean and covariance of data
        '''
        _, _, self.mu_hat_1, self.mu_hat_2, self.sigma_hat_1, self.sigma_hat_2 = \
            Gaussian_empirical_parameters(X, y)

    def predict(self, X):
        p1 = multivariate_normal.pdf(X, mean=self.mu_hat_1, cov=self.sigma_hat_1, allow_singular=True)
        p2 = multivariate_normal.pdf(X, mean=self.mu_hat_2, cov=self.sigma_hat_2, allow_singular=True)
        return (p1 > p2) * -1 + 2

    def get_params(self, deep=False):
        return {}

    def show_params(self, deep=False):
        print(f'mu_hat_1: {self.mu_hat_1}\nmu_hat_2: {self.mu_hat_2}\
        \nsigma_hat_1:\n{self.sigma_hat_1}\nsigma_hat_2:\n{self.sigma_hat_2}')


####################################################################################
# Regularized Gaussian Discriminant Analysis with a Gaussian-inverse-Wishart prior #
####################################################################################
class RGDA:
    def __init__(self, mu, kappa, Lambda, nu=None):
        self.mu, self.kappa, self.Lambda, self.nu = mu, kappa, Lambda, nu

    def fit(self, X, y, **kwargs):
        """ Fisrt function(for training)
        Training with Gaussian DA with Gaussian inverse-Wishart prior
        the maximum-likelihood estimated mu and sigma by posterior distribution
        We suppose these two class have the same prior distribution for the parameters
        """
        n_1, n_2, Xm_1, Xm_2, V_1, V_2 = Gaussian_empirical_parameters(X, y)
        mu_hat_1 = (n_1 * Xm_1 + self.kappa * self.mu) / (n_1 + self.kappa)
        mu_hat_2 = (n_2 * Xm_2 + self.kappa * self.mu) / (n_2 + self.kappa)

        d = X.shape[1]
        if self.nu is None:
            self.nu = X.shape[1] + 1
        m_mu_1 = (Xm_1 - self.mu).reshape(-1, 1)
        m_mu_2 = (Xm_2 - self.mu).reshape(-1, 1)
        sigma_hat_1 = (n_1 * V_1 + ((n_1 * self.kappa) / (n_1 + self.kappa)) \
                       * np.dot(m_mu_1, m_mu_1.T) \
                       + inv(self.Lambda)) / (n_1 + self.nu + d + 2)
        sigma_hat_2 = (n_2 * V_2 + ((n_2 * self.kappa) / (n_2 + self.kappa)) \
                       * np.dot(m_mu_2, m_mu_2.T) \
                       + inv(self.Lambda)) / (n_2 + self.nu + d + 2)
        self.mu_hat_1, self.mu_hat_2, self.sigma_hat_1, self.sigma_hat_2 = \
            mu_hat_1, mu_hat_2, sigma_hat_1, sigma_hat_2

    def predict(self, X):
        p1 = multivariate_normal.pdf(X, mean=self.mu_hat_1, cov=self.sigma_hat_1)
        p2 = multivariate_normal.pdf(X, mean=self.mu_hat_2, cov=self.sigma_hat_2)
        return (p1 > p2) * -1 + 2

    def get_params(self, deep=False):
        return {'mu': self.mu, 'kappa': self.kappa, 'Lambda': self.Lambda, 'nu': self.nu}

    def show_params(self, deep=False):
        print(f'mu_hat_1: {self.mu_hat_1}\nmu_hat_2: {self.mu_hat_2}\
        \nsigma_hat_1:\n{self.sigma_hat_1}\nsigma_hat_2:\n{self.sigma_hat_2}')


#################
# Visualisation #
#################
def models_comparison(data_path, name):
    data = pd.read_csv(data_path)
    y = data.z.to_numpy()
    X = data.drop(columns=["z"]).to_numpy()
    # GDA
    r = cross_validate(GDA(), X, y,
                       scoring='accuracy', return_train_score=True)
    r["model"] = "GDA"
    df = pd.DataFrame(data=r)
    # QLR
    r = cross_validate(RQLR('none', solver='newton-cg'), X, y,
                       scoring='accuracy', return_train_score=True)
    r["model"] = "QLR"
    df = df.append(pd.DataFrame(data=r), ignore_index=True)
    # RGDA
    mu = 0
    kappa = 1
    d = X.shape[1]
    Lambda = np.eye(d)
    r = cross_validate(RGDA(mu, kappa, Lambda), X, y,
                       scoring='accuracy', return_train_score=True)
    r["model"] = "RGDA"
    df = df.append(pd.DataFrame(data=r), ignore_index=True)
    # RQLR
    r = cross_validate(RQLR('l2', 'newton-cg'), X, y,
                       scoring='accuracy', return_train_score=True)
    r["model"] = "RQLR"
    df = df.append(pd.DataFrame(data=r), ignore_index=True)
    # Data frame processing
    df_acc = df.drop(columns=["fit_time", "score_time"])
    df_time = df.drop(columns=["test_score", "train_score"])
    base = min(df_acc["test_score"].min(), df_acc["train_score"].min())
    df_acc["test_score"] = df_acc["test_score"] - base
    df_acc["train_score"] = df_acc["train_score"] - base

    df_acc = df_acc.melt(id_vars=["model"], value_name="accuracy - base", var_name="validation")
    df_time = df_time.melt(id_vars=["model"], value_name="time", var_name="stage")

    sns.set_theme(style="whitegrid")
    plt.figure()
    acc_ax = sns.barplot(x="model", y="accuracy - base", hue="validation", data=df_acc)
    plt.savefig(f'{name}_acc.pdf')

    plt.figure()
    time_ax = sns.barplot(x="model", y="time", hue="stage", data=df_time)
    plt.savefig(f'{name}_time.pdf')
    return df


def plot_gda_boundry(model, name, df=pd.read_csv('data/synth.csv')):
    y = df.z
    X = df.drop(columns=["z"])
    indices = np.arange(len(y))
    sns.scatterplot(x="X1", y="X2", data=df, hue="z")

    x_min, x_max = X['X1'].min() - .5, X['X1'].max() + .5
    y_min, y_max = X['X2'].min() - .5, X['X2'].max() + .5
    h = 0.1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax = plt.gca()
    scout, = ax.plot([], [], linestyle="dashed")
    sns.lineplot([0], [0], label=name, ax=ax, linestyle="dashed")

    ax.contour(xx, yy, Z, linestyles="dashed", antialiased=True)

def add_decision_boundary(model, levels=None, resolution=1000, ax=None, label=None):
    """Trace une frontière de décision sur une figure existante.

    La fonction utilise `model` pour prédire un score ou une classe
    sur une grille de taille `resolution`x`resolution`. Une (ou
    plusieurs frontières) sont ensuite tracées d'après le paramètre
    `levels` qui fixe la valeur des lignes de niveaux recherchées.

    """

    if ax is None:
        ax = plt.gca()


    if isinstance(model, MetaEstimatorMixin):
        return add_decision_boundary(model.best_estimator_, levels=levels, resolution=resolution, ax=ax, label=label)

    elif callable(model):
        if levels is None:
            levels = [0]
        def predict(X):
            return model(X)

    elif isinstance(model, BaseEstimator):
        n_classes = len(model.classes_)
        if "decision_function" in dir(model):
            if n_classes == 2:
                if levels is None:
                    levels = [0]
                def predict(X):
                    return model.decision_function(X)
        else:
            levels = np.arange(n_classes - 1) + .5
            def predict(X):
                pred = model.predict(X)
                _, idxs = np.unique(pred, return_inverse=True)
                return idxs
    else:
        raise Exception("Modèle pas supporté")


    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], resolution)
    yy = np.linspace(ylim[0], ylim[1], resolution)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = predict(xy).reshape(XX.shape)

    scout, = ax.plot([], [], linestyle="dashed")
    sns.lineplot([0], [0], label=label, ax=ax, linestyle="dashed")

    ax.contour(
        XX,
        YY,
        Z,
        levels=levels,
        linestyles="dashed",
        antialiased=True,
    )

#########
# Tools #
#########

def get_pi_hat(pd_serie, z):
    # get the proportion of class z
    c = pd_serie["z"]
    return sum(c == z) / len(pd_serie)


def Gaussian_empirical_parameters(X, Y, z1=1, z2=2):
    # get gaussian empirical parameters from data
    X_1 = X[Y == z1]
    X_2 = X[Y == z2]
    mu1 = X_1.mean(axis=0)
    mu2 = X_2.mean(axis=0)
    n_1 = len(X_1)
    n_2 = len(X_2)
    x1_sub_mu1 = np.array(X_1 - mu1)
    x2_sub_mu2 = np.array(X_2 - mu2)
    V_1 = (x1_sub_mu1.T @ x1_sub_mu1) / n_1
    V_2 = (x2_sub_mu2.T @ x2_sub_mu2) / n_2
    return n_1, n_2, mu1, mu2, V_1, V_2


def is_pos_def(x):
    # verify if x is positive definite
    return np.all(np.linalg.eigvals(x) > 0)


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def add_decision_boundary(model, levels=None, resolution=1000, ax=None, label=None):
    """Trace une frontière de décision sur une figure existante.

    La fonction utilise `model` pour prédire un score ou une classe
    sur une grille de taille `resolution`x`resolution`. Une (ou
    plusieurs frontières) sont ensuite tracées d'après le paramètre
    `levels` qui fixe la valeur des lignes de niveaux recherchées.

    """

    if ax is None:
        ax = plt.gca()

    if isinstance(model, MetaEstimatorMixin):
        return add_decision_boundary(model.best_estimator_, levels=levels, resolution=resolution, ax=ax, label=label)

    elif callable(model):
        if levels is None:
            levels = [0]

        def predict(X):
            return model(X)

    elif isinstance(model, BaseEstimator):
        n_classes = len(model.classes_)
        if "decision_function" in dir(model):
            if n_classes == 2:
                if levels is None:
                    levels = [0]

                def predict(X):
                    return model.decision_function(X)
        else:
            levels = np.arange(n_classes - 1) + .5

            def predict(X):
                pred = model.predict(X)
                _, idxs = np.unique(pred, return_inverse=True)
                return idxs
    else:
        raise Exception("Modèle pas supporté")

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], resolution)
    yy = np.linspace(ylim[0], ylim[1], resolution)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = predict(xy).reshape(XX.shape)

    scout, = ax.plot([], [], linestyle="dashed")
    orig_color = scout.get_color()
    sns.lineplot([0], [0], label=label, ax=ax, color=orig_color, linestyle="dashed")

    ax.contour(
        XX,
        YY,
        Z,
        levels=levels,
        colors=orig_color,
        linestyles="dashed",
        antialiased=True,
    )


def add_decision_boundaries(df, models):
    colors = sns.color_palette()
    for model, name in models:
        y = df.z
        X = df.drop(columns=["z"])
        cls = model()
        cls.fit(X, y)
        add_decision_boundary(cls, label=name)


def main():
    pass


if __name__ == "__main__":
    main()
