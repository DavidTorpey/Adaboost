import numpy as np
from sklearn.tree import DecisionTreeClassifier as clf
from sklearn.datasets import load_breast_cancer as load_data
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

def getData():
    data_map = load_data()
    X = data_map['data']
    y = data_map['target']
    y[np.where(y==0)] = -1
    return scale(X), y

X, y = getData()

K = 20
accs = []
Ts = [1, 5, 10, 25, 50, 100, 250]
for T in Ts:
    acs = []
    for k in range(K):    
        
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.33)
        m = xtrain.shape[0]
        wts = np.ones(m) / m
        C = [None] * T
        alpha = [None] * T
        for i in range(T):
            sampledX = xtrain[np.random.choice(np.arange(0, m), size = m, p = wts.ravel()), :]
            sampledy = ytrain[np.random.choice(np.arange(0, m), size = m, p = wts.ravel())]
            C[i] = clf(max_depth=1).fit(sampledX, sampledy)
            yhat = C[i].predict(xtrain)    
            a = wts
            b = ytrain != yhat
            e = np.matmul(a.reshape(1, -1), b.reshape(-1, 1))
            alpha[i] = 0.5 * np.log((1-e) / e)
            wts = wts * np.exp(-alpha[i] * ytrain * yhat)
            wts = wts / np.sum(wts)
            
        ypred = []
        for i in range(xtest.shape[0]):
            test_sample = xtest[i, :].reshape(1, -1)
            s = 0
            for t in range(T):
                s += alpha[t]*C[t].predict(test_sample)
            ypred.append(np.sign(s)[0, 0])
        acs.append(f1_score(ytest, ypred))
    print T
    accs.append(np.array(acs).mean())

plt.plot(Ts, accs)
plt.xlabel('Number of Weak Learners')
plt.ylabel('F1-Score')
plt.title('Adaboost F1-Score For Different Number Of Weak Learners')
plt.show()

f = clf().fit(xtrain, ytrain)
print "Normal classifier: ", f1_score(ytest, f.predict(xtest))