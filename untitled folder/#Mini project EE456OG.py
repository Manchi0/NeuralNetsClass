#Mini project EE456
#Manas Munjial; mqm6911

import random
from mat4py import loadmat
import numpy as np 
import matplotlib.pyplot as plt


dt1 = loadmat('/Users/manasmunjial/Downloads/Data set A.mat.mat')
dt2 = loadmat('/Users/manasmunjial/Downloads/Data set B.mat.mat')

# def TrainTest_split(dt,ratio=0.3):
#     t=len(dt) -int(len(dt)*ratio)
#     return dt[:t],dt[t:]


def TrainTest_split_pro(x,y,ratio=0.3):
    joined=list(zip(x,y))
    random.shuffle(joined)
    t=len(joined) -int(len(joined)*ratio)

    train=joined[:t]
    test=joined[t:]

    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)


    return list(x_train),list(y_train),list(x_test),list(y_test)



def activation(v,theta):
    if v>theta:
        return 1
    elif v < -theta:
        return -1
    else:
        return 0



x1=dt1['X']
y1=dt1['Y']

x2=dt2['X']
y2=dt2['Y']


#split into training and testing data, The function by default splits into 70/30 ratio

x1_train, y1_train, x1_test ,y1_test = TrainTest_split_pro(x1,y1)

x2_train, y2_train, x2_test ,y2_test = TrainTest_split_pro(x2,y2)





wt1= [0,0] #hard coded 2 weights since we have have 2D x arrays
func=None
learning_rate=0.1




def train(x_train,y_train,x_test,y_test, wt,theta=0):
    wt= [0,0]
    all_error=[]

    for a in range(100):
        sum_error=0
        for i in range(len(x_train)):
            v= wt[0] * x_train[i][0] + wt[1] * x_train[i][1]   #dot product

            func=activation(v,theta)

            if func!=y_train[i][0]:
                wt[0]+= learning_rate*x_train[i][0]*y_train[i][0]
                wt[1]+= learning_rate*x_train[i][1]*y_train[i][0]

            # if func!=y[i][0]:
            #     wt[0]+= learning_rate*x[i][0]*y[i][0]
            #     wt[1]+= learning_rate*x[i][1]*y[i][0]
            #     b+= y[i][0]*learning_rate

                # sum_error+=1
                # error=y[i][0]-func
                # print(error)
                # sum_error+=abs(error)
        
        sum_error=testing(wt,x_test,y_test)
   
        all_error.append(sum_error)

    return all_error, wt





def testing(weights,x_test,y_test,theta=0):
    errors=0
    for i in range(len(x_test)):
        v= weights[0] * x_test[i][0] + weights[1] * x_test[i][1]   #dot product
        func=activation(v,theta)
        if func!=y_test[i][0]:
            errors+=1
    return errors




def testing2(x,y,wt1):
    group1f1=[]
    group1f2=[]
    groupN1f1=[]
    groupN1f2=[]

    for i in range(len(y)):
        if y[i][0]==1:
            group1f1.append(x[i][0])
            group1f2.append(x[i][1])

        else:
            groupN1f1.append(x[i][0])
            groupN1f2.append(x[i][1])

    
    plt.figure()


    # y_values=[]
    # x_values=group1f1+groupN1f1
    # for i in x_values:
    #     y_values+= [-(wt1[0]*i)/wt1[1]]


    first_column = [row[0] for row in x]  # Extract the first column
    x_values = np.linspace(min(first_column), max(first_column), 100)   
    y_values= -(wt1[0]*x_values)/wt1[1]


    plt.scatter(group1f1, group1f2, color='blue', label='Class 1')
    plt.scatter(groupN1f1, groupN1f2, color='red', label='Class -1')

    # x_line = np.linspace(0, 6, 100)


    plt.plot(x_values, y_values, color='red')
    plt.xlabel('F 1')
    plt.ylabel('F 2')
    plt.legend()
    plt.show()


# def predict(x,y):




def ErrorsPerEpochGraph(x_train,y_train,x_test,y_test, wt,theta=0):
    all_error,wts=train(x_train,y_train,x_test,y_test, wt,theta=0)
    print(wts)

    plt.figure()    
    plt.plot(range(len(all_error)),all_error)
    plt.xlabel("Epochs")
    plt.ylabel("Errors")
    plt.show()
    
    # for i in range(len(all_error)):
    #     print(all_error[i],i)
    


avg,wts=train(x2_train,y2_train,x2_test,y2_test,wt1)
testing2(x1_test,y1_test,wts)

ErrorsPerEpochGraph(x1_train,y1_train,x1_test,y1_test,wt1)