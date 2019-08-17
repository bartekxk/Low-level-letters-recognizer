import numpy as np
from math import log
def train(Theta1,Theta2,X,y,_lambda,num_labels,alpha):
    m = np.size(X,0)
    a1 = X
    a1 = np.append(np.ones((np.size(X,0),1)),a1,axis=1)
    z2 = np.dot(a1,Theta1.getT())
    a2 = sigmoid(z2)
    a2 = np.append(np.ones((np.size(a2,0),1)),a2,axis=1)
    a3 = sigmoid(np.dot(a2,Theta2.getT()))
    Theta1NoBias = np.delete(Theta1, 0, axis=1)
    Theta2NoBias = np.delete(Theta2, 0, axis=1)
    J = costFunction(Theta1NoBias,Theta2NoBias,a3,y,_lambda,num_labels,m)
    print("Cost:" + str(J))
    delta3 = a3 - y
    delta2 = np.multiply(np.dot(Theta2NoBias.getT(),delta3.getT()),sigmoidGradient(z2).getT())
    Theta2 = np.subtract(Theta2,np.dot(delta3.getT(),a2) * alpha / m)
    Theta1 = np.subtract(Theta1,np.dot(delta2,a1) * alpha / m)
    for i in range(np.size(Theta1NoBias,0)):
        for j in range(np.size(Theta1NoBias,1)):
            Theta1.itemset((i,j+1), Theta1.item((i,j+1)) + (Theta1NoBias.item((i,j)) * _lambda / m))
    for i in range(np.size(Theta2NoBias,0)):
        for j in range(np.size(Theta2NoBias,1)):
            Theta2.itemset((i,j+1), Theta2.item((i,j+1)) + (Theta2NoBias.item((i,j)) * _lambda / m))
    return Theta1,Theta2,J
def costFunction(Theta1NoBias,Theta2NoBias,a,y,_lambda,num_labels,m):
    J = 0
    for i in range(m):
        for k in range(num_labels):
            J = J + (y.item((i,k)) * log(a.item((i,k))) + (1 - y.item((i,k))) * log(1-a.item((i,k)))) / (-m)
    J = J + (np.sum(np.square(Theta1NoBias)) + np.sum(np.square(Theta2NoBias)))*_lambda/(2*m)
    return J

def sigmoidGradient(z):
    z = sigmoid(z)
    return np.multiply(z,1-z)

def sigmoid(z):
    return 1/(1+np.exp(-z))

X = np.loadtxt('train.data',usecols=range(1,17))
y = np.loadtxt('train.data',usecols=range(0,1))
X = np.asmatrix(X)
y = np.asmatrix(y).getT()
num_labels = 26
y_tmp = np.asmatrix(np.zeros([np.size(X,0),num_labels]))
theta1 = np.asmatrix(np.random.rand(30,np.size(X,1)+1))
theta2 = np.asmatrix(np.random.rand(26,31))
for i in range(np.size(y,0)):
    y_tmp[i,int(y[i,0])-1] = 1
y = y_tmp
J = 100000000
while J>0.0001:
    theta1, theta2, J = train(theta1,theta2,X,y,0.1,num_labels,0.1)
np.savetxt('theta1.data', theta1)
np.savetxt('theta2.data', theta2)
