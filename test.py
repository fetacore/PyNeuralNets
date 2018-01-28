import NN
import auxiliary as aux
import numpy as np

# Various test Cases

##Serial Correlation Case
#n = 100
#x_1 = np.random.normal(5,3,size=(n,1))
#x_2 = 5 * x_1 + np.random.normal(2,1,size=(n,1))
#x = comb_arrays([x_1,x_2])
#target = x_1**2 + x_2 + np.random.normal(0,2,size=(n,1))
#eta = 0.005
#iterations=400
#hlayers = [10,50,500,3]
#active_fnct = ['tanh','tanh','Sigmoid','tanh','linear']

##Serial Correlation and extra variables
#n = 100
#x_1 = np.random.normal(5,3,size=(n,1))
#x_2 = 5 * x_1 + np.random.normal(2,1,size=(n,1))
#x_3 = np.random.normal(0,3,size=(n,1))
#x = comb_arrays([x_1,x_2,x_3])
#target = x_1**2 + x_2 + np.random.normal(0,2,size=(n,1))
#eta = 0.005
#iterations=100
#hlayers = [10,50,500,3]
#active_fnct = ['tanh','tanh','Sigmoid','tanh','linear']

##Ommitted Variables and Irrelevant
#n = 100
#x_1 = np.random.normal(5,3,size=(n,1))
#x_2 = 5 * x_1 + np.random.normal(2,1,size=(n,1))
#x_3 = np.random.normal(0,3,size=(n,1))
#x = comb_arrays([x_1,x_3])
#target = x_1**2 + x_2 + np.random.normal(0,2,size=(n,1))
#eta = 0.005
#iterations=100
#hlayers = [10,50,500,3]
#active_fnct = ['tanh','tanh','Sigmoid','tanh','linear']

#Ommitted Variables
#n = 100
#x_1 = np.random.normal(5,3,size=(n,1))
#x_2 = 5 * x_1 + np.random.normal(2,1,size=(n,1))
#x_3 = np.random.normal(0,3,size=(n,1))
#x = x_2
#target = x_1**2 + x_2 + np.random.normal(0,2,size=(n,1))
#eta = 0.005
#iterations=100
#hlayers = [10,50,500,3]
#active_fnct = ['tanh','tanh','Sigmoid','tanh','linear']


#n = 100
#x_1 = np.random.normal(5,3,size=(n,1))
#x_2 = np.random.normal(1,3,size=(n,1))
#true_model = x_1**2 + np.sign(x_2)
#error = np.random.normal(0,1,size=(n,1))
#target = true_model + error
#x = comb_arrays([x_1,x_2])
#n = 1000
#x = np.random.normal(5,3,size=(n,1))
##x_2 = np.random.normal(1,3,size=(n,1))
#true_model = x**2
#error = np.random.normal(0,5,size=(n,1))
#target = true_model + error

n = 1000
x_1 = np.random.normal(5,3,size=(n,1))
x_2 = np.random.normal(1,3,size=(n,1))
true_model = x_1**2 + np.sign(x_2)
error = np.random.normal(0,1,size=(n,1))
Y = true_model + error
X = aux.comb_arrays([x_1,x_2])

threshold = 1e-24
eta = 0.005
hlayers = [100,50,50,30,20,5,30,5]
active_fnct = [
    'tanh',
    'Arctan',
    'Softsign',
    'Sigmoid',
    'tanh',
    'tanh',
    'tanh',
    'tanh',
    'linear'
]

[zreg,w,b,loss,iterations] = NN.neural_sifu(Y,X,threshold,eta,hlayers,active_fnct)

print('zreg = ',zreg,' | w = ',w,' | b = ',b,' | loss = ',loss,' | iterations = ',iterations)
