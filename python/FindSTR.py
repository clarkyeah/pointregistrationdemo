import numpy as np
from numpy.core.defchararray import endswith
import cmath


def FindSTR3(c1, c0):
    rE, cE = np.shape(c1)
    rF, cF = np.shape(c0)
    if(cE != 2) or (cF != 2):
        raise ValueError('E, F should be an nx2matrix')
    if(rE != rF):
        raise ValueError('matrices E and F are of different size')
    
    A = c1[:,0]+c1[:,1]*1j
    B = c0[:,0]+c0[:,1]*1j

    meanA = sum(A)/rE
    meanB = sum(B)/rF
    A = A - meanA
    B = B - meanB
    x = np.linalg.pinv(B) * A
    Theta = np.angle(x)
    S = np.abs(x)
    v = meanA - (x/S)*meanB
    fcr = v/(1-x/S)
    t = [np.real(v), np.imag(v)]
    c = [np.real(fcr), np.imag(fcr)]

    return S, Theta, t, c 

if __name__== "__main__":

    c1 =np.array([[4.3368,-53.6819],[4.7889,-56.6994],[5.2542,-59.7153],[-3.9465,-64.4640],[-4.5972, -60.8229],[-5.2092, -57.1722]])
    c0 =np.array([[4.2622,-53.6927], [4.7088,  -56.7115], [5.1730,  -59.7280], [-3.9670,  -64.4679], [-4.6131,  -60.8256], [-5.2059, -57.1717]])   

    [S,Theta,t,c]=FindSTR3(c1,c0)