import numpy as np
import matplotlib.pyplot as plt


def linear(x1,y1,x2):
    y2 = []
    
    for i in range (len(x2)):
        for j in range (len(x1)):
            if x2[i] <= x1[j]:
                w1 = x1[j] - x2[i]
                w2 = x2[i] - x1[j-1]
                temp = y1[j] * w2 + y1[j-1] * w1
                y2.append(float(temp))
                break

    return y2


def closest_neightbour(matrix_x, matrix_y):
    return


def main():
    x1 = np.linspace(0,5,10)
    y1 = np.sin(x1)
    x2 = np.linspace(0,5,100)


    plt.plot(x1, y1, '.r')
    plt.show()

    y2 = linear(x1,y1,x2)
    print(len(y2))
    print(y2)

    plt.plot(x2, y2, '-g')
    plt.show()

if __name__ == "__main__":
    main()