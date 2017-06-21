import numpy as np
import matplotlib.pyplot as plt

def d_waist(z):
    d0 = 1e-4
    d1 = 1e-4
    z1 = 5.0
    f = 1.0
    lambda_ = 1064e-9

#    d2 = (((1-z1/f)**2)/(d1**2)+(np.pi*d1/(2*lambda_)**2/f**2))**(-1/2)
    d = d0*(1+(4*lambda_**2*(z**2))/(np.pi**2*d0**4))
    d2 = (((1-z1/f)**2)/(d1**2)+(np.pi*d1/(2*lambda_)**2/f**2))**(-1/2)
    z2 = f + f**2*(z1-f)/((z1-f)**2+(np.pi*d1/2*lambda_)**2)

    return d, d2, z2

if __name__ == "__main__":
    z = np.arange(-4,0,4/100)
    d, d2, z2 = d_waist(z)
    print(d2, z2)

    plt.plot(z, d)
    plt.show()
