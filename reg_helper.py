from sklearn.linear_model import LogisticRegression
import math
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.logistic import _logistic_loss

def nCr(n,r):
    f = math.factorial
    return int(f(n) / f(r) / f(n-r))

def get_polynimial_set(X, degree = 12, bias = True):
    # Recibe el dataset X de numero_de_muestras x features  y devuelve una matriz con todas las combinaciones 
    # De los productos del grado indicado en degree
    k = 2
    n = degree + k
    pos = 0
    X_mat = np.zeros((X.shape[0],nCr(n,k)))
    for i in range(degree + 1):
        for j in range(i+1):
            X_mat[:,pos] = (X[:,0]**(i-j))*X[:,1]**j
            pos = pos + 1
    if bias:
        return X_mat
    else:
        return X_mat[:,1:]

def plot_boundaries(X_train, y_train, score, probability_func, degree = None, n_colors = 100, mesh_res = 1000, ax = None):
    X = X_train #np.vstack((X_test, X_train))
    margin_x = (X[:, 0].max() - X[:, 0].min())*0.05
    margin_y = (X[:, 1].max() - X[:, 1].min())*0.05
    x_min, x_max = X[:, 0].min() - margin_x, X[:, 0].max() + margin_x
    y_min, y_max = X[:, 1].min() - margin_y, X[:, 1].max() + margin_y
    hx = (x_max-x_min)/mesh_res
    hy = (y_max-y_min)/mesh_res
    x_domain = np.arange(x_min, x_max, hx)
    y_domain = np.arange(y_min, y_max, hy)
    xx, yy = np.meshgrid(x_domain, y_domain)

    if ax is None:
        ax = plt.subplot(1, 1, 1)
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if degree is not None:
        polynomial_set = get_polynimial_set(np.c_[xx.ravel(), yy.ravel()], degree = degree)
        Z = probability_func(polynomial_set)[:, 1]
    else:
        Z_aux = probability_func(np.c_[xx.ravel(), yy.ravel()])
        Z = Z_aux[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    
    cf = ax.contourf(xx, yy, Z, n_colors, vmin=0., vmax=1., cmap=cm, alpha=.8)
    plt.colorbar(cf, ax=ax)
    #plt.colorbar(Z,ax=ax)

    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k', s=100, marker='o')
    
    boundary_line = np.where(np.abs(Z-0.5)<0.001)
    
    ax.scatter(x_domain[boundary_line[1]], y_domain[boundary_line[0]], color='k', alpha=0.5, s=1)
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    #ax.set_xticks(())
    #ax.set_yticks(())
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=40, horizontalalignment='right')

def fit_and_get_regions(X_train, y_train, X_test, y_test, degree = 2, lambd = 0, plot_it = True, print_it = False):
    X_train_degree = get_polynimial_set(X_train, degree=degree)
    X_test_degree = get_polynimial_set(X_test, degree=degree)
    # Defino el modelo de clasificación como Regresion Logistica
    if lambd == 0:
        C1 = 10000000000
    else:
        C1 = 1/lambd 
    #C2 = 1
    clf_logist_pol = LogisticRegression(C=C1, fit_intercept=False)

    # Entreno el modelo con el dataset de entrenamiento
    clf_logist_pol.fit(X_train_degree, y_train)

    # Calculo el score (Exactitud) con el dataset de testeo
    score_test_logist_pol = clf_logist_pol.score(X_test_degree, y_test)

    # Calculo tambien el score del dataset de entrenamiento para comparar
    score_train_logist_pol = clf_logist_pol.score(X_train_degree, y_train)
    
    #loss_train = _logistic_loss(clf_logist_pol.coef_, X_train_degree, y_train, 1 / clf_logist_pol.C)
    #loss_test = _logistic_loss(clf_logist_pol.coef_, X_test_degree, y_test, 1 / clf_logist_pol.C)

    # print('Test Accuracy (Exactitud):',score_test_logist_pol)
    # print('Train Accuracy (Exactitud):',score_train_logist_pol)
    # print('coeficientes:', clf_logist_pol.coef_)
    # print('intercept:', clf_logist_pol.intercept_)
    if plot_it:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))
        plot_boundaries(X_train, y_train, score_train_logist_pol, clf_logist_pol.predict_proba, degree=degree, ax=ax1)
        plot_boundaries(X_test, y_test, score_test_logist_pol, clf_logist_pol.predict_proba, degree=degree, ax=ax2)
        print('Regresion Logistica Polinomial de orden '+str(degree) +', con lamdba (regularización L2):' +  str(lambd))
        plt.show()
    if print_it:
        print('Train Accuracy (Exactitud):',score_train_logist_pol)
        print('Test Accuracy (Exactitud):',score_test_logist_pol)
    return score_train_logist_pol, score_test_logist_pol, clf_logist_pol.coef_ #, loss_train, loss_test

def test_options(X_train, y_train, X_test, y_test, options, plot_it=False):
    train_acc_array = []
    test_acc_array = []
    degrees = []
    lambdas = []
    coefs_array_mean = []
    coefs_array_std = []
    coefs_abs_max = []
    for opt in options:
        tr_acc, ts_acc, coefs = fit_and_get_regions(X_train, y_train, X_test, y_test, degree = opt['degree'], lambd = opt['lambd'], plot_it=plot_it)
        train_acc_array.append(tr_acc)
        test_acc_array.append(ts_acc)
        degrees.append(opt['degree'])
        lambdas.append(opt['lambd'])
        coefs_array_mean.append(coefs.mean())
        coefs_array_std.append(coefs.std())
        coefs_abs_max.append(np.max(abs(coefs)))
    return degrees, lambdas, train_acc_array, test_acc_array, coefs_array_mean, coefs_array_std, coefs_abs_max

def plot_boundaries_keras(X_train, y_train, score, probability_func, degree=None, bias=False, h = .02, ax = None, margin=0.5):
    X = X_train
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    if ax is None:
        ax = plt.subplot(1, 1, 1)
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    
    if degree is not None:
        polynomial_set = get_polynimial_set(np.c_[xx.ravel(), yy.ravel()], degree = degree, bias=bias)
        Zaux = probability_func(polynomial_set)
    else:
        Zaux = probability_func(np.c_[xx.ravel(), yy.ravel()])
        # Z = Z_aux[:, 1]
    print(Zaux.shape)
    
    if Zaux.shape[1] == 2:
        Z = Zaux[:, 1]
    else:
        Z = Zaux[:, 0]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    
    cf = ax.contourf(xx, yy, Z, 50, cmap=cm, alpha=.8)
    plt.colorbar(cf, ax=ax)
    #plt.colorbar(Z,ax=ax)

    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k', s=100)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=40, horizontalalignment='right')