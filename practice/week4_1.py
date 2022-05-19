import cv2
import numpy as np
from matplotlib import pyplot as plt

src = cv2.imread('./data/h1.jpg',cv2.IMREAD_GRAYSCALE)

def drawHist(input_src):
    h, w = input_src.shape

    x = np.zeros(256)


    for i in range(h-1):
        for j in range(w-1):
            x[src[i,j]] += 1

    plt.title('hist')
    binX = np.arange(256)
    plt.bar(binX,x, color ='b')
    plt.show()

drawHist(src)