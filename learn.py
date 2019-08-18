import numpy as np
from math import log
def train(Theta1,Theta2,Theta3,X,y,_lambda,num_labels,alpha):
    m = np.size(X,0)
    a1 = X
    a1 = np.append(np.ones((np.size(X,0),1)),a1,axis=1)
    z2 = np.dot(a1,Theta1.getT())
    a2 = sigmoid(z2)
    a2 = np.append(np.ones((np.size(a2,0),1)),a2,axis=1)
    z3 = np.dot(a2,Theta2.getT())
    a3 = sigmoid(z3)
    a3 = np.append(np.ones((np.size(a3,0),1)),a3,axis=1)
    z4 =  np.dot(a3,Theta3.getT())
    a4 = sigmoid(z4)
    Theta1NoBias = np.delete(Theta1, 0, axis=1)
    Theta2NoBias = np.delete(Theta2, 0, axis=1)
    Theta3NoBias = np.delete(Theta3, 0, axis=1)
    J = costFunction(Theta1NoBias,Theta2NoBias,Theta3NoBias,a4,y,_lambda,num_labels,m)
    print("Cost:" + str(J))
    delta4 = a4 - y
    delta3 = np.multiply(np.dot(Theta3NoBias.getT(),delta4.getT()),sigmoidGradient(z3).getT())
    delta2 = np.multiply(np.dot(Theta2NoBias.getT(),delta3),sigmoidGradient(z2).getT())
    Theta3 = np.subtract(Theta3,np.dot(delta4.getT(),a3) * alpha / m)
    Theta2 = np.subtract(Theta2,np.dot(delta3,a2) * alpha / m)
    Theta1 = np.subtract(Theta1,np.dot(delta2,a1) * alpha / m)
    for i in range(np.size(Theta1NoBias,0)):
        for j in range(np.size(Theta1NoBias,1)):
            Theta1.itemset((i,j+1), Theta1.item((i,j+1)) + (Theta1NoBias.item((i,j)) * _lambda / m))
    for i in range(np.size(Theta2NoBias,0)):
        for j in range(np.size(Theta2NoBias,1)):
            Theta2.itemset((i,j+1), Theta2.item((i,j+1)) + (Theta2NoBias.item((i,j)) * _lambda / m))
    for i in range(np.size(Theta3NoBias,0)):
        for j in range(np.size(Theta3NoBias,1)):
            Theta3.itemset((i,j+1), Theta3.item((i,j+1)) + (Theta3NoBias.item((i,j)) * _lambda / m))
    return Theta1,Theta2,Theta3,J
def costFunction(Theta1NoBias,Theta2NoBias,Theta3NoBias,a,y,_lambda,num_labels,m):
    J = 0
    for i in range(m):
        for k in range(num_labels):
            J = J + (y.item((i,k)) * log(a.item((i,k))) + (1 - y.item((i,k))) * log(1-a.item((i,k)))) / (-m)
    J = J + (np.sum(np.square(Theta1NoBias)) + np.sum(np.square(Theta2NoBias)) + np.sum(np.square(Theta3NoBias)))*_lambda/(2*m)
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
theta1 = np.asmatrix(np.random.randn(500,np.size(X,1)+1))*np.sqrt(2/(np.size(X,1)+1))
theta2 = np.asmatrix(np.random.randn(300,501))*np.sqrt(2.0/501.0)
theta3 = np.asmatrix(np.random.randn(26,301))*np.sqrt(2.0/301.0)
for i in range(np.size(y,0)):
    y_tmp.itemset((i,int(y.item((i,0)))-1),1)
y = y_tmp
J = 100000000
J_prev = 0
while abs(J-J_prev)>0.0001:
    J_prev = J
    theta1, theta2,theta3, J = train(theta1,theta2,theta3,X,y,1.2,num_labels,0.1)
np.savetxt('theta1.data', theta1)
np.savetxt('theta2.data', theta2)
np.savetxt('theta3.data', theta3)
