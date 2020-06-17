import numpy as np

def sigmoid(x):
    return 1.0/(1.0+np.exp(-1.0*x))

def gaussian_kernel(x, y, sigma=1.0):
    return np.exp(-0.5*np.linalg.norm(x-y)**2/sigma**2)

def logistic_train(X, Y, lr=0.01, max_iter = 100):
    m=len(Y)
    alpha=np.zeros(len(X))
    K = np.zeros((m,m))
    for i in range(m):
        # print(i)
        for j in range(i,m):
            K[i,j] = gaussian_kernel(X[i], X[j])
            K[j,i] = K[i,j]

    for i in range(max_iter):
        # print('iter',i)
        t=np.sum(alpha*K,1)
        t=Y-sigmoid(t)
        t=np.sum(K*t[:,np.newaxis],0)
        alpha=alpha+(lr/m)*t
    return alpha

def logistic_predict(Xtr,X, alpha):
    m=X.shape[0]
    n = Xtr.shape[0]
    K = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            K[i,j] = gaussian_kernel(X[i], Xtr[j])
    t=sigmoid(np.sum(alpha*K,1))
    return np.round(t)

	# return np.round(sigmoid(X @ Weights)

def main():
    fd=open('dataset1.txt')
    x=np.asarray([item.split() for item in fd.readlines()],dtype=float)
    # print(x[0])
    X_train=x[:,0:2]
    Y_train=x[:,2]
    # print(X_train)
    alpha=logistic_train(X_train,Y_train)
    y_predict = logistic_predict(X_train,X_train,alpha)
    cnt = np.sum(y_predict==Y_train)
    print(cnt/len(Y_train))
main()

    