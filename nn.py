import pandas as pd
import numpy as np  
import scipy as sp
import matplotlib.pyplot as plt

m=36000 #number of training samples, m is used to calculate the average cost and also in backpropogation to calculate the gradients
v=6000 #number of validation samples
activation_function='sigmoid'   #can be 'sigmoid', 'relu' or 'tanh'. change accordingly 


#loading csv as a panda df and separating input columns from output columns 
df = pd.read_csv('train.csv')
indf=df.iloc[0:m,1:]  #leaving 6000 samples for validation
outdf=df.iloc[0:m,:1]

#converting to matrices
inp=indf.to_numpy()
out=outdf.to_numpy()

Y=np.zeros((m,10))
for i in range(0,m):
    num=out[i][0]
    Y[i][num]=1

#transposing so dimensions are correct when feeding forward 
A0=inp.T 
Y=Y.T 

#initialising weights and biases where L is number of layers(excluding input layer) and n contains no. of neurons in each layer
L = 4
n = [784,280,280,280,10]

listweights=[0]*L
listbiases=[0]*L
for i in range(L):
    listweights[i]=np.random.randn(n[i+1],n[i])
    listbiases[i]=np.random.randn(n[i+1],1)

#activation functions and their derivatives
def actf(z):
    if activation_function=='sigmoid':
        return 1 / (1 + np.exp(-1 * z))
    elif activation_function=='relu':
        return np.maximum(0, z)
    elif activation_function=='tanh':
        return np.tanh(z)
    else:
        print("Invalid activation function. Choose 'sigmoid', 'relu', or 'tanh'")
        quit()
 
#can code softmax but will require a time consuming loop(for the column wise summations) or multiple numpy statements with broadcasting 

def actf_derivative(z):
    if activation_function=='sigmoid':
        return z*(1-z) 
    elif activation_function=='relu':
        return np.where(z>0,1,0)
    elif activation_function=='tanh':
        return 1-z**2




#some prerequisites for testing/cross validation
indfv=df.iloc[m:,1:] 
outdfv=df.iloc[m:,:1]

#converting to matrices
inpv=indfv.to_numpy()
outv=outdfv.to_numpy()

Yv=np.zeros((v,10))
for i in range(0,v):
    num=outv[i][0]
    Yv[i][num]=1

#transposing so dimensions are correct when feeding forward 
A0v=inpv.T 
Yv=Yv.T




def feedforward(A0,listweights,listbiases,L):
    listpreac=[0]*L
    listafac=[0]*L

    listpreac[0]=listweights[0]@A0 + listbiases[0]
    listafac[0]=actf(listpreac[0])
    for j in range(1,L):
        listpreac[j]=listweights[j]@listafac[j-1] + listbiases[j]
        listafac[j]=actf(listpreac[j])

    listafac[L-1]=sp.special.softmax(listpreac[L-1],axis=0)
    
    return listpreac, listafac

def costcalc(Y,listafac,m):
    #calculating cost using categorical cross entropy loss function.
    y_hat_clipped = np.clip(listafac[L-1], 1e-15, 1 - 1e-15)  #clipping to avoid log(0) which is undefined and log(1) to prevent overfitting/overconfidence
 
    losses=np.sum(-1*Y*np.log(y_hat_clipped),axis=0, keepdims=True) #basically doing -y*log(yhat) for each element in the matrix after softmax activation and then summing along the columns to get the loss for each sample. losses should be a 1 by 60000 matrix. we use * because we want corresponding elementwise multiplication and not matrix multiplication

    cost = (1/m)*np.sum(losses) #cost is average loss over all the samples
    #calculating cost using Mean squared error is to be used only in regression scenario ie when there is only one node in the final output layer which is supposed to predict/print the number in the image
    return cost





#training
epochs=500
alpha=0.1 #learning rate 
costs=[] #to store the cost for each epoch
costsv=[] #to store the cost for each epoch for validation set

for i in range(epochs):


    #feed forward
    listpreac,listafac=feedforward(A0,listweights,listbiases,L)

    #calculating cost
    cost=costcalc(Y,listafac,m)

    costs.append(cost)



    #feed forward for validation
    listpreacv,listafacv=feedforward(A0v,listweights,listbiases,L)

    #calculating cost
    costv=costcalc(Yv,listafacv,v)

    costsv.append(costv)




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
        dc_dzl=dc_dal*actf_derivative(listafac[L-2-k])
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
    if i%50==0:
      print(costs[i])
      print(costsv[i])

plt.plot(range(epochs),costs)
plt.plot(range(epochs),costsv)





#outputing predictions for test.csv
df_test=pd.read_csv('test.csv')
inpt=df_test.to_numpy()
A0t=inpt.T 

#feed forward
listpreact,listafact=feedforward(A0t,listweights,listbiases,L)

predictions=np.argmax(listafact[L-1], axis=0)
submission_df=pd.DataFrame({'ImageId': range(1,len(predictions)+1),'Label': predictions})
submission_df.to_csv('submission.csv', index=False)
