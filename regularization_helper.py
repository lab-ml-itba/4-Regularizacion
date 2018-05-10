import math
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression

def nCr(n,r):
    f = math.factorial
    return int(f(n) / f(r) / f(n-r))

def get_polynimial_set(X, degree = 12):
    k = 2
    n = degree + k
    pos = 0
    X_mat = np.zeros((X.shape[0],nCr(n,k)))
    for i in range(degree + 1):
        for j in range(i+1):
            X_mat[:,pos] = (X[:,0]**(i-j))*X[:,1]**j
            pos = pos + 1
    return X_mat[:,1:]

def plot_it(X,y, s = 10, colors = ['r','b'], ax=None):
    if ax is None:
        plt.scatter(*X[y==1].T, color=colors[0], marker='o',s = s)
        plt.scatter(*X[y==0].T, color=colors[1], marker='x', s = s)
    else:
        ax.scatter(*X[y==1].T, color=colors[0], marker='o',s = s)
        ax.scatter(*X[y==0].T, color=colors[1], marker='x', s = s)

def plot_classifier(X, y, predict, degree, N = 500, ax=None):
    # create a mesh to plot in
    if ax is None:
        plt.figure(figsize=(8,8))
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, N),
                             np.linspace(y_min, y_max, N))
    polynomial_set = get_polynimial_set(np.c_[xx.ravel(), yy.ravel()], degree = degree)
    Z = predict(polynomial_set)
    plot_it(X[:,:2], y, s= 20, ax=ax)
    Z = Z.reshape(xx.shape)>0.5
    if ax is None:
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.2)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
    else:
        ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.2)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())

def get_simple_dataset(N_total=200, r_class_1=0.5, inner_radius=1.5):
    N_1 = int(N_total*r_class_1)
    N_2 = int(N_total*(1-r_class_1))
    radius = 10
    slice_witdh = 7
    circle_r = 2*np.random.randn(N_1,1) + radius

    # random angle
    alpha = 2 * math.pi * np.random.rand(N_1,1)
    # random radius
    r = circle_r + slice_witdh*np.random.rand(N_1,1)
    # calculating coordinates
    X = np.zeros((N_1+N_2,2))
    y = np.zeros((N_1+N_2))
    print(X.shape)
    X[:N_1,0] = (r * np.cos(alpha)).reshape(-1)
    X[:N_1,1] = (r * np.sin(alpha)).reshape(-1)
    y[:N_1] = 1

    X[N_1:] = inner_radius*np.sqrt(radius)*np.random.randn(N_2,2)
    y[N_1:] = 0

    X = X/X.max()
    return X,y

def save_dataset(name, X, y):
    dataset = np.zeros((X.shape[0], X.shape[1]+1))
    dataset[:, :X.shape[1]] = X
    dataset[:, X.shape[1]] = y
    np.save(name, dataset)

