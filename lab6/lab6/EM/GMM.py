import numpy as np
import matplotlib.pyplot as plt
import math


def fit_gmm(data,K=3,max_iter = 30):

    def init(X,K):
        # can change the init
        _,n = X.shape
        
        a = np.full((K),1/K)
        mu = [X[i] for i in range(K)]
        sigma = np.zeros((n,n))
        for i in range(n):
            sigma[i,i] = 1
        
        Sigma = [sigma for i in range(K)]
        return a,mu,Sigma

    def gaussian(x,mean,cov):

        # return gaussian at x
        x_m = x - mean
        # x_m = x_m.reshape((2,1))
        # print(cov)
        c=1. / (np.sqrt((2 * np.pi)**(2) * np.linalg.det(cov)))
        ret_val = (c * np.exp(-(x_m.T@np.linalg.inv(cov)@x_m) / 2))
        return ret_val

    def GMM(data,K,iter_num = 10):
        # fit the data by completing 
        # Refer to tutorial 6
        # can change the code to terminate under log-likelihood convergence
        
        m,_ = data.shape
        a,mu,sigma = init(data,K)
        # a is the weights of gaussians in MM
        # mu is the array of mean
        # sigma is the array of variance

        # cluster=[[]for i in range(K)]
        for _ in range(iter_num):

            # E_step
            ''' write code for E-step here , '''
            p=np.zeros((m,K))
            for i in range(m):
                for j in range(K):
                    # print(np.linalg.inv(sigma[j]))
                    p[i,j]=a[j]*gaussian(data[i],mu[j],sigma[j])
            p/=p.sum(1).reshape(m,1)

            # M_step
            ''' write code for M-step here '''  
            a=p.sum(0)/m
            mu=(p/p.sum(0)).T@data

            for i in range(K):
                temp=np.zeros((2,2))
                for j in range(m):
                    ndata=(data[j]-mu[i]).reshape(2,1)
                    cov_mat=ndata@ndata.T
                    # print(cov_mat)
                    temp+=cov_mat*p[j,i]
                temp/=p[:,i].sum()
                sigma[i]=temp

            for i in range(K):
                plt.plot(mu[i][0],mu[i][1],'o',c = 'r')
        
        return a,mu,sigma



    a,mu,sigma = GMM(data,K,max_iter)
    print(a)
    print(mu)
    print(sigma)

    num = 200
    l = np.linspace(0,15,num)
    X, Y = np.meshgrid(l, l)

    plt_data = np.dstack((X,Y))
    for i in range(K):
        Z = [[gaussian(plt_data[k][j],mu[i],sigma[i]) for j in range(num)] for k in range(num)]
        cs = plt.contour(X,Y,Z)
        plt.clabel(cs)

    plt.show()