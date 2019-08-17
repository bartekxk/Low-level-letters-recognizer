import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def ver(X,y,Theta1,Theta2,Theta3):
    m = np.size(X,0)
    a1 = X
    a1 = np.append(np.ones((np.size(X, 0), 1)), a1, axis=1)
    z2 = np.dot(a1, Theta1.getT())
    a2 = sigmoid(z2)
    a2 = np.append(np.ones((np.size(a2, 0), 1)), a2, axis=1)
    z3 = np.dot(a2, Theta2.getT())
    a3 = sigmoid(z3)
    a3 = np.append(np.ones((np.size(a3, 0), 1)), a3, axis=1)
    z4 = np.dot(a3, Theta3.getT())
    a4 = sigmoid(z4)
    y_tmp = np.asmatrix(np.zeros([m,1]))
  #  print(a4)
  #  print(y)
    for i in range(m):
        max =0
        imax = 0
        for j in range (26):
            if a4.item((i,j)) > max:
                max = a4.item((i,j))
                imax = j
        y_tmp.itemset((i,0),imax+1)
    print(y_tmp)
    correct = 0
    for i in range(m):
        if y_tmp.item((i,0)) == y.item((i,0)):
            correct = correct + 1
    print(str(correct) + "/" + str(m))

X = np.asmatrix(np.loadtxt('ver.data',usecols=range(1,17)))
y = np.asmatrix(np.loadtxt('ver.data',usecols=range(0,1))).getT()
theta1 = np.asmatrix(np.loadtxt('theta1.data'))
theta2 = np.asmatrix(np.loadtxt('theta2.data'))
theta3 = np.asmatrix(np.loadtxt('theta3.data'))
ver(X,y,theta1,theta2,theta3)
