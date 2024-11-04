import math
import random
import matplotlib.pyplot as plt


def generate_complex_data(n_samples):
    y=[]
    step = 20 / (n_samples-1) if n_samples > 1 else 0
    x = [-10 + i * step for i in range(n_samples)]
    print(len(x))
    for x_val in x:
        y_val=(0.1*x_val**3)-(0.5*x_val**2)+(0.2*x_val)+3+ 2*math.sin(2*x_val)+random.gauss(0, 10)
        y.append(y_val)
    return x,y

def real_val(x_val):
    return (0.1*x_val**3)-(0.5*x_val**2)+(0.2*x_val)+3+ 2*math.sin(2*x_val)


def plotG(n_samples):
    x,y=generate_complex_data(n_samples)
    plt.figure()
    plt.title('Generated Complex Non-linear Dataset')
    plt.scatter(x, y)
    plt.grid(True)        
    plt.show()


def rand_weight_bias(inputL,next_layer): # list with weights
    weights=[]
    bias=[]
    for i in range(inputL):
        by_in=[]
        for j in range(next_layer):
            by_in.append(random.randrange(-100, 100)*0.0001)
        weights.append(by_in)
    for k in range(next_layer):
        bias.append(0)

    return weights, bias


def mean_squared_error(y_true,y_predicted):
    val=0
    for i in range(len(y_true)):
        val+=(y_true-y_predicted)**2
    return val/len(y_true)
        

def TrainTest_split_pro(x,y,ratio=0.3):
    joined=list(zip(x,y))
    random.shuffle(joined)
    t=len(joined) -int(len(joined)*ratio)

    train=joined[:t]
    test=joined[t:]

    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)

    return list(x_train),list(y_train),list(x_test),list(y_test)

def relu(val):
    return max(0,val)

def diff_relu(val):
    return 0 if val<=0 else 1

print(diff_relu(2))

def tanh(val):
    return math.tanh(val)

def diff_tanh(val):
    return 1 - math.tanh(val)**2


class TwoLayerMLP:
    def __init__(self):
        self.w_l1,self.bias_l1=rand_weight_bias(1,128)
        self.w_l2,self.bias_l2=rand_weight_bias(128,64)
        self.w_l3,self.bias_l3=rand_weight_bias(64,1)
    
    def forward(self,X):
        s3=0
        x1_vals=[]
        x2_vals=[]

        for i in range(len(self.w_l1[0])):
            s1=x*self.w_l1[0][i]
            x1_vals.append(relu(s1+self.bias_l1))


        for j in range(len(self.w_l2[0])):
            s2=0
            for k in range(len(x1_vals)):
                s2+=x1_vals[k]*self.w_l2[k][j]
            x2_vals.append(tanh(s2+self.bias_l2))
        
        for L in range(len(x2_vals)):
            s3+=x2_vals[L]*self.w_l3[L][0]
        output=s3+self.bias_l3

        return output


    def backward(self,y,output,lr):
        pass
        

    def train(self,y,epochs,lr,batch_size):
        pass

    def predict(self):
        pass


# print(plotG(100))
# print(len(rand_weight_bias(1,128)[1]))
hey=TwoLayerMLP(1)
print(hey.forward())