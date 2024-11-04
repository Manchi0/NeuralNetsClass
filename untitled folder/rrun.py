from mat4py import loadmat
import numpy as np 
import matplotlib.pyplot as plt



fp1 = '/Users/manasmunjial/Downloads/Data set A.mat.mat'
fp2= '/Users/manasmunjial/Downloads/Data set B.mat.mat'
dt1 = loadmat(fp1)
dt2 = loadmat(fp2)



x_1=dt1['X']
y_1=dt1['Y']

x=dt2['X']
y=dt2['Y']

wt= [0,0]
theta=0
func=None
learning_rate=0.5
b=0
avr_error=[]


for a in range(100):
    sum_error=0
    for i in range(len(x)):
        v= wt[0] * x[i][0] + wt[1] * x[i][1]

        if v>theta:
            func=1
        elif v< theta:
            func= -1
        elif v==0:
            func=0
        
        if func!=y[i][0]:
            wt[0]+= learning_rate*x[i][0]*y[i][0]
            wt[1]+= learning_rate*x[i][1]*y[i][0]
            sum_error+= abs(y[i][0]-func)

    avr_error.append(sum_error/1000)
    
plt.figure()    
for i in range(len(avr_error)):
    print(avr_error[i],i)

plt.plot(range(len(avr_error)),avr_error)

plt.show()
    
    

    


