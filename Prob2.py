import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def toeplitz(i,T):
        for k in range(T.shape[1]):
            T[i,k] = 1/5.6*np.exp(-0.1*(i-k)**2)
        return T

def blurring(path):

    im = Image.open(path) 
    im = im.convert('L')
    img = np.array(im)
    img = img.astype(np.float64) / 255
    # plt.imshow(img,cmap = 'gray')
    # plt.axis('off')
    # plt.show()

    T = np.zeros([img.shape[0],img.shape[1]])

    for i in range(T.shape[0]):
        T = toeplitz(i,T)

    B = T @ img @ T.T

    # plt.gray()
    # plt.imshow(B,cmap = 'gray')
    # plt.axis('off')
    # plt.show()

    return img, B, T