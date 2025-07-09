import pandas as pd
import numpy as np  
import scipy as sp
import matplotlib.pyplot as plt
#loading csv as a panda df and separating input columns from output columns 
df = pd.read_csv('train.csv')
indf=df.drop(df.columns[0], axis=1)
outdf=df.iloc[0:,:1]

#converting to matrices
inp=indf.to_numpy()
out=outdf.to_numpy()

Y=np.zeros((42000,10))
for i in range(0,42000):
    num=out[i][0]
    Y[i][num]=1

#transposing so dimensions are correct when feeding forward 
A0=inp.T 
Y=Y.T 
m=42000

#initialising weights and biases where L is number of layers(excluding input layer) and n contains no. of neurons in each layer
L = 4
n = [784,280,280,280,10]

listweights=[0]*L
listbiases=[0]*L
for i in range(L):
    listweights[i]=np.random.randn(n[i+1],n[i])
    listbiases[i]=np.random.randn(n[i+1],1)

#activation functions 
def actf(z):
    return 1 / (1 + np.exp(-1 * z)) #sig
   #return                         #ReLu

#can code softmax but will require a time consuming loop(for the column wise summations) or multiple numpy statements with broadcasting 
#def smax(z):
 #   return np.exp(z)/np.sum(np.exp(z))

#training
epochs=1000
alpha=0.01 #learning rate 
costs=[]

for i in range(epochs):


    #feed forward
    listpreac=[0]*L
    listafac=[0]*L

    listpreac[0]=listweights[0]@A0 + listbiases[0]
    listafac[0]=actf(listpreac[0])
    for j in range(1,L):
        listpreac[j]=listweights[j]@listafac[j-1] + listbiases[j]
        listafac[j]=actf(listpreac[j])

    listafac[L-1]=sp.special.softmax(listpreac[L-1],axis=0)

    #calculating cost using categorical cross entropy loss function. 
    losses=np.sum(-1*Y*np.log(listafac[L-1]),axis=0, keepdims=True) #basically doing -y*log(yhat) for each element in the matrix after softmax activation and then summing along the columns to get the loss for each sample. losses should be a 1 by 60000 matrix. we use * because we want corresponding elementwise multiplication and not matrix multiplication

    cost = (1/m)*np.sum(losses) #cost is average loss over all the samples

    #calculating cost using Mean squared error, to be used only in regression scenario ie when there is only one node in the final output layer which is supposed to predict/print the number in the image
    costs.append(cost)



    #Backpropogation
    #for output layer
    dc_dzL=(1/m)*(listafac[L-1]-Y)
    assert dc_dzL.shape==(n[L],m)
    dzL_dwL=listafac[L-2]

    dc_dwL=(dc_dzL)@((dzL_dwL).T)
    assert dc_dwL.shape==(n[L],n[L-1])
    dc_dbL=np.sum(dc_dzL,axis=1, keepdims=True)
    assert dc_dbL.shape==(n[L],1)

    dc_dal=(listweights[L-1].T)@dc_dzL  #propogator ie used to calculate dwl and dbl for the next layer

    #making a list of the gradients for weights and biases for each layer
    wegrads=[0]*L
    bigrads=[0]*L
    wegrads[L-1]=dc_dwL
    bigrads[L-1]=dc_dbL

    for k in range(L-1):
        dc_dzl=dc_dal*(listafac[L-2-k]*(1-listafac[L-2-k]))
        assert dc_dzl.shape==(n[L-1-k],m)
        if L-3-k >= 0:
            dzl_dwl = listafac[L-3-k]
        else:
            dzl_dwl = A0

        dc_dwl=(dc_dzl)@((dzl_dwl).T)
        assert dc_dwl.shape==(n[L-1-k],n[L-2-k])
        dc_dbl=np.sum(dc_dzl,axis=1, keepdims=True)
        assert dc_dbl.shape==(n[L-1-k],1)

        wegrads[L-2-k]=dc_dwl
        bigrads[L-2-k]=dc_dbl

        dc_dal=(listweights[L-2-k].T)@dc_dzl  #propogator ie used to calculate dwl and dbl for the next layer
    #weights and biases gradiants are now present in the lists for each layer 

    #updating parameters
    for p in range(L):
        listweights[p]=listweights[p]-(alpha*wegrads[p])
        listbiases[p]=listbiases[p]-(alpha*bigrads[p])
    print(costs)

plt.plot(costs,range(epochs))






