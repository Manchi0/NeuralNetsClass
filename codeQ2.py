#problem 2
import random
from mat4py import loadmat
import matplotlib.pyplot as plt




dt1_1 = loadmat('/Users/manasmunjial/Downloads/Two_moons_no_overlap2.mat')
dt1_2 = loadmat('/Users/manasmunjial/Downloads/Two_moons_overlap3.mat')



def TrainTest_split_pro(x,y,ratio=0.3):
    joined=list(zip(x,y))
    random.shuffle(joined)
    t=len(joined) -int(len(joined)*ratio)

    train=joined[:t]
    test=joined[t:]

    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)

    return list(x_train),list(y_train),list(x_test),list(y_test)


x11=dt1_1['X']
y11=dt1_1['Y']

x12=dt1_2['X']
y12=dt1_2['Y']


x11_train, y11_train, x11_test ,y11_test = TrainTest_split_pro(x11,y11)
x12_train, y12_train, x12_test ,y12_test = TrainTest_split_pro(x12,y12)


learning_rate=0.01

def model(input,hidden): # list with weights
    weights=[]
    bias=[]
    for i in range(input):
        by_in=[]
        for j in range(hidden):
            by_in.append(random.randrange(1, 100)*0.0001)
        weights.append(by_in)
    for k in range(hidden+1):
        bias.append(random.randrange(1, 100)*0.0001)

    return weights, bias


def activation(v):
    if v>=0:
        return 1
    elif v < 0:
        return -1
    
def andNN(z_o):
    b1=1
    y_in= b1 + z_o[0]+z_o[1]+z_o[2]+z_o[3]
    if y_in==5:
        y_f=1
    else:
        y_f=-1
    return y_f


def delta_update_weight1(learning_rate,yT,weights,z_in,neuron_index,x_val,bias):
    for i in range(len(weights)):
        weights[i][neuron_index]= weights[i][neuron_index]+ learning_rate*(yT-z_in[neuron_index])*x_val[i]
    bias=bias + learning_rate*(yT-z_in[neuron_index])
    return weights,bias




def delta_update_weightall(learning_rate,yT,weights,z_in,x_val,bias):
    for neuron_index in range(len(z_in)):
        for i in range(len(weights)):
            weights[i][neuron_index]= weights[i][neuron_index]+ learning_rate*(yT-z_in[neuron_index])*x_val[i]
        bias[neuron_index]=bias[neuron_index] + learning_rate*(yT-z_in[neuron_index])
    return weights,bias



def testing(wt,x_test,y_test,bias,output_neuron):
    if output_neuron==1:
        change=-1
    else:
        change=1
    errors=0
    z_i=[None,None,None,None]
    z_o=[None,None,None,None]
    
    for i in range(len(x_test)):
        for j in range(len(z_i)):
            z_i[j]=bias[j]+x_test[i][0]*wt[0][j]+x_test[i][1]*wt[1][j]
            z_o[j]= activation(z_i[j])
       
        #and gate
        y_f=andNN(z_o)

        if y_f!=y_test[i][0]*change:
            errors+=1
    
    return errors




        



def training(x,y,x_test,y_test, epochs,wt,bias,output_neuron):
    if output_neuron==0:
        change=1
    elif output_neuron ==1:
        change=-1

    all_error=[]
    z_i=[None,None,None,None]
    z_o=[None,None,None,None]
    for epoch in range(epochs):
        for i in range(len(x)):
            for j in range(len(z_i)):
                z_i[j]=bias[j]+x[i][0]*wt[0][j]+x[i][1]*wt[1][j]
                z_o[j]= activation(z_i[j])
            #and gate

            y_f=andNN(z_o)
            
            if y_f!=y[i][0]*change:
                if min(z_i)<0.25 and min(z_i)>-0.25:
                    
                    z_abs=[abs(ele) for ele in z_i]
                    sorted_abs=sorted(z_abs)
                    for lo in range(len(sorted_abs)):
                        l=z_abs.index(sorted_abs[lo])
                        if z_i[l]<0.25 and z_i[l]>-0.25:
                            z_o[l]*= -1
                            y_fu=andNN(z_o)
                            if y_fu==y[i][0]*change:
                                wt,bias[l]=delta_update_weight1(learning_rate,y[i][0]*change,wt,z_i,l,x[i],bias[l]);
                            else:
                                z_o[l]*= -1
                else:
                    wt,bias= delta_update_weightall(learning_rate,y[i][0]*change,wt,z_i,x[i],bias)

        sum_error=testing(wt,x_test,y_test,bias,output_neuron)         
        all_error.append(sum_error)

    return all_error,wt
           

import matplotlib.pyplot as plt


def decisionL(x, y, wt):
    ny = [yi[0] for yi in y]

    group1 = [xi for xi, yi in zip(x, ny) if yi == 1]
    groupN1 = [xi for xi, yi in zip(x, ny) if yi == -1]

    plt.figure(figsize=(8, 6))

    x_feature = [xi[0] for xi in x]
    min_x0 = min(x_feature)
    max_x0 = max(x_feature)

    num_points = 100
    if num_points == 1:
        x_values = [min_x0]
    else:
        step = (max_x0 - min_x0) / (num_points - 1)
        x_values = [min_x0 + step * i for i in range(num_points)]

    num_weights = len(wt[0])  # Assuming wt[0] and wt[1] have the same length

    for n in range(num_weights):
        w0 = wt[0][n]
        w1 = wt[1][n]
        
        if w1 != 0:
            y_values = [-(w0 * xv) / w1 for xv in x_values]
        else:
            # Handle the case where w1 is zero to avoid division by zero
            y_values = [-(w0 * xv) / 1e-8 for xv in x_values]
        
        plt.plot(x_values, y_values, label=f'Line {n+1}')

    group1_x = [xi[0] for xi in group1]
    group1_y = [xi[1] for xi in group1]

    groupN1_x = [xi[0] for xi in groupN1]
    groupN1_y = [xi[1] for xi in groupN1]



    plt.scatter(group1_x, group1_y, color='blue', label='Class 1', edgecolor='k')
    plt.scatter(groupN1_x, groupN1_y, color='red', label='Class -1', edgecolor='k')
    plt.xlabel('F1')
    plt.ylabel('F2')
    plt.title('Decision Boundaries')
    plt.legend()
    plt.grid(True)
    plt.show()



def Graphs(x_train,y_train,x_test,y_test, wt, bias,output):
    if output==2:
        all_error1,wts1=training(x_train,y_train,x_test,y_test, 20, wt, bias,0)
        all_error2,wts2=training(x_train,y_train,x_test,y_test, 20, wt, bias,1)
     

    print('Output Neuron 1:\nWeights: ', wts1)
    print('Error Rate: ', (sum(all_error1)/len(all_error1)))

    print('Output Neuron 2:\nWeights: ', wts2)
    print('Error Rate: ', (sum(all_error2*-1)/len(all_error2)))

    plt.figure()    
    plt.plot(range(len(all_error1)),all_error1, color='red',label="Neuron 1")
    plt.plot(range(len(all_error2)),all_error2, color='blue',label="Neuron 2")
    plt.xlabel("Epochs")
    plt.ylabel("Errors")
    plt.show()

    

    decisionL(x_test,y_test,wts1)








wts,bi=model(2,4)
output=2
Graphs(x11_train,y11_train,x11_test,y11_test,wts,bi,output)
Graphs(x12_train,y12_train,x12_test,y12_test,wts,bi,output)










