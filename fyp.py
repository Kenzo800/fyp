import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

#np.set_printoptions(suppress=True)

def loss_fn(X, Y, beta):
    return cp.norm2(X @ beta - Y)**2

def regularizer(beta):
    return cp.norm1(beta)

def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)

def mse(X, Y, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value

def readAndCutDataset(fileName):
    Assets_Returns_pd = pd.read_excel(fileName,sheet_name="Assets_Returns")
    Index_Returns_pd = pd.read_excel(fileName,sheet_name="Index_Returns")
    X = np.array(Assets_Returns_pd) # Assets_Returns[day][stock]
    Y = np.array(Index_Returns_pd) # Index_Returns[day][0]
    n = X.shape[1] # number of assets in the index
    m = X.shape[0] # number of trading day
    X_train = X[:400, :]
    Y_train = Y[:400]
    X_test = X[400:, :]
    Y_test = Y[400:]
    return X_train,X_test,Y_train,Y_test,n,m

def threshold(w,thresholdValue):
    w = np.array(w.value)
    w = np.where(np.abs(w) <= thresholdValue, 0, w)
    w = w/np.sum(w)
    return w

def plotOneTimeFitting(l1,l2):
    plt.plot(l1, label="Index")
    plt.plot(l2, label="Portfolio")
    plt.xlabel('Trading days')
    plt.ylabel('Return %')
    plt.title('Index tracking')
    plt.legend(loc='upper left')
    plt.show()

def oneTimeFitting(X_train,X_test,Y_train,Y_test,n,m, lbd, thresholdValue):
    w = cp.Variable((n,1)) # weights of the assets in the portfolio
    lambd = cp.Parameter(nonneg=True)
    lambd.value = lbd
    problem = cp.Problem(cp.Minimize(objective_fn(X_train, Y_train, w, lambd)))
    #problem = cp.Problem(cp.Minimize(objective_fn(X_train, Y_train, w, lambd)),[w>=0,cp.sum(w)==1])
    problem.solve()
    sparse_count = 0
    
    #threshold
    w = threshold(w,thresholdValue) # cp.Variable -> np.array

    for e in w:
        if e == 0:
            sparse_count+=1

    pltLineIndex = np.cumprod(1+(Y_test))
    pltLinePortfolio = np.cumprod(1+np.matmul(X_test,w))
    plotOneTimeFitting(pltLineIndex,pltLinePortfolio)
    return problem, w, sparse_count

def printResult(problem, w, sparse_count, X_test, Y_test):
    print("status:",problem.status)
    print("\nThe optimal value is", problem.value)
    print("The norm of the residual is ", cp.norm(X_test @ w - Y_test, p=2).value)
    print("The weight is", w)
    print("The sum of weight is", np.sum(w))
    print("sparsity = ",sparse_count)


def lambdaAnalysis(X_train, X_test, Y_train, Y_test, n, m, thresholdValue):
    train_errors = []
    test_errors = []
    w_values = []
    lambd = cp.Parameter(nonneg=True)
    lambd_values = np.logspace(-2, 3, 50)
    w = cp.Variable((n,1))
    problem = cp.Problem(cp.Minimize(objective_fn(X_train, Y_train, w, lambd)))
    for v in lambd_values:
        lambd.value = v
        problem.solve()
        train_errors.append(mse(X_train, Y_train, w))
        test_errors.append(mse(X_test, Y_test, w))
        w_values.append(w.value)
    return train_errors,test_errors,w_values,lambd_values

def plot_train_test_errors(train_errors, test_errors, lambd_values):
    plt.plot(lambd_values, train_errors, label="Train error")
    plt.plot(lambd_values, test_errors, label="Test error")
    plt.xscale("log")
    plt.legend(loc="upper left")
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.title("Mean Squared Error (MSE)")
    plt.show()

def plot_regularization_path(lambd_values, w_values):
    num_coeffs = len(w_values[0])
    for i in range(num_coeffs):
        plt.plot(lambd_values, [wi[i] for wi in w_values])
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.xscale("log")
    plt.title("Regularization Path")
    plt.show()




X_train,X_test,Y_train,Y_test,n,m = readAndCutDataset('SP500.xlsx')


# (lambda analysis)
# train_errors,test_errors,w_values,lambd_values = lambdaAnalysis(X_train, X_test, Y_train, Y_test, n, m, thresholdValue = 1e-4)
# plot_train_test_errors(train_errors, test_errors, lambd_values)
# plot_regularization_path(lambd_values, w_values)

# (one time fitting with lambda and threshold value)
problem, w, sparse_count = oneTimeFitting(X_train, X_test, Y_train, Y_test, n, m, lbd = 1.7, thresholdValue = 1e-4)
printResult(problem, w, sparse_count, X_test, Y_test)












