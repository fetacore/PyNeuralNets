import auxiliary as aux
import numpy as np

def neural_sifu(target,x,threshold,eta,hlayers,active_fnct):
    x = aux.normalize(x)
    w = aux.w_list_creator(x,hlayers)
    b = aux.bias_list_creator(hlayers)
    loss = []
    l = 10e+5
    iterations = 0
    while l > threshold:
        print(l)
        new_z = np.zeros((len(x),1))
 #       #Plots
  #      if(len(x[0,:])==1):
   #         plt.figure(iterations)
    #        plt.plot(x,new_z)
     #       plt.xlabel('X')
      #      plt.ylabel('Y')
       #     plt.title('Regression Curve')
        #    plt.grid(True)
         #   plt.show('hold')
          #  plt.pause(0.05)
        for i in range(0,len(x[:,0])):
            x_i = x[i,:]
            x_i.shape = (1,len(x_i))
            target_i = target[i,:]
            z = aux.neurons(x_i,w,b,hlayers,active_fnct)
            deltas = aux.delta_rule(target_i,z,w,hlayers,active_fnct)
            [w,b] = aux.update(x_i,z,w,b,deltas,eta,hlayers)
            z=aux.neurons(x_i,w,b,hlayers,active_fnct)
            new_z[i] = z[1][len(hlayers)]
        l = aux.pred_loss(target,new_z)
        loss.append(l)
        iterations = iterations + 1
    return([new_z,w,b,loss,iterations])