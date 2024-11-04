#Mini project EE456
#Manas Munjial; mqm6911

import random      
from mat4py import loadmat
import numpy as np 
import matplotlib.pyplot as plt


dt1 = loadmat('/Users/manasmunjial/Downloads/Data set A.mat.mat')
dt2 = loadmat('/Users/manasmunjial/Downloads/Data set B.mat.mat')


def TrainTest_split_pro(x, y, ratio=0.3):
    joined = list(zip(x, y))
    random.shuffle(joined)
    t = len(joined) - int(len(joined) * ratio)
    train = joined[:t]
    test = joined[t:]
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def activation(v,theta=0):
    if v>theta:
        return 1
    elif v < -theta:
        return -1
    else:
        return 0

x1 = np.array(dt1['X'])
y1 = np.array(dt1['Y']).flatten()
x2 = np.array(dt2['X'])
y2 = np.array(dt2['Y']).flatten()

#split into training and testing data, The function by default splits into 70/30 ratio
x1_train, y1_train, x1_test ,y1_test = TrainTest_split_pro(x1,y1)
x2_train, y2_train, x2_test ,y2_test = TrainTest_split_pro(x2,y2)

wt1= [0,0] #hard coded 2 weights since we have have 2D x arrays
wt2= [0,0]
func=None
learning_rate=0.1


def train(x_train, y_train, x_test, y_test, wt, learning_rate=0.1, theta=0):
    all_error = []
    for epoch in range(100):
        for i in range(len(x_train)):
            v = np.dot(wt, x_train[i])
            func = activation(v, theta)
            if func != y_train[i]:
                wt += learning_rate * y_train[i] * x_train[i]

        sum_error = testing(wt, x_test, y_test)
        all_error.append(sum_error)
    return all_error, wt

def testing(weights,x_test,y_test,theta=0):
    errors=0
    for i in range(len(x_test)):
        v= weights[0] * x_test[i][0] + weights[1] * x_test[i][1]   #dot product
        func=activation(v,theta)
        if func!=y_test[i]:
            errors+=1

    return errors


def testing2(x, y, wt):
    group1 = x[y == 1]
    groupN1 = x[y == -1]
    plt.figure()
    x_values = np.linspace(min(x[:, 0]), max(x[:, 0]), 100)
    if wt[1] != 0:
        y_values = -(wt[0] * x_values) / wt[1]
        plt.plot(x_values, y_values, color='red')
    else:
        y_values = -(wt[0] * x_values) / 0.00000001
        plt.plot(x_values, y_values, color='red')
        
    plt.scatter(group1[:, 0], group1[:, 1], color='blue', label='Class 1')
    plt.scatter(groupN1[:, 0], groupN1[:, 1], color='red', label='Class -1')
    plt.xlabel('F1')
    plt.ylabel('F2')
    plt.legend()
    plt.show()




def Graphs(x_train,y_train,x_test,y_test, wt,theta =0):
    all_error,wts=train(x_train, y_train, x_test, y_test, wt, learning_rate=0.1, theta=theta)
    print('Weights: ', wts)
    print('Error Rate: ', (sum(all_error)/len(all_error))*100)

    plt.figure()    
    plt.plot(range(len(all_error)),all_error)
    plt.xlabel("Epochs")
    plt.ylabel("Errors")
    plt.show()



    testing2(x_test,y_test,wts)

    
    


def calculate_theta(x_train, y_train, x_test, y_test, wt):
    dict0={}
    for i in np.arange(-30.0, 30.0, 1):
        all_error,wts=train(x_train, y_train, x_test, y_test, wt, learning_rate=0.1, theta=i)
        dict0[float(i)]=sum(all_error)    

    small=min(dict0.values())
    for i in dict0:
        if dict0[i]==small:
            return i



#Case 1
Graphs(x1_train,y1_train,x1_test,y1_test,wt1) #DT1
Graphs(x2_train,y2_train,x2_test,y2_test,wt2) #DT2


best_theta1=calculate_theta(x1_train, y1_train, x1_test, y1_test, wt1)
best_theta2=calculate_theta(x2_train, y2_train, x2_test, y2_test, wt2)


#Case 2
Graphs(x1_train,y1_train,x1_test,y1_test,wt1,theta=best_theta1) #DT1
Graphs(x2_train,y2_train,x2_test,y2_test,wt2,theta=best_theta2) #DT2


