import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Interpolacja liniowa
def linear(x1,y1,x2):
    y2 = []
    
    # Porównaj każdą wartość z wektora x2 z wartościami wektora x1 aby sprawdzić w którym przedziale znajduje się punkt.
    for i in range (len(x2)):
        for j in range (len(x1)):
            if x2[i] <= x1[j]:

                # Oblicz wagi na podstawie odległości nowego punktu od 2 sąsiadujących
                w1 = x1[j] - x2[i]
                w2 = x2[i] - x1[j-1]

                # Oblicz wartość dla punktu z wektora x2 i dodaj ją do macierzy y2
                temp = y1[j] * w2 + y1[j-1] * w1

                y2.append(float(temp * 1.8))
                break

    return y2


# Interpolacja najbliższy-sąsiad
def closest_neightbour(x1,y1,x2):
    y2 = []

    # Porównaj każdą wartość z wektora x2 z wartościami wektora x1 aby sprawdzić do którego punktu z macierzy x1 jest bliżej
    for i in range (len(x2)):
        for j in range (len(x1)):
            if x2[i] <= x1[j]:
                w1 = x1[j] - x2[i]
                w2 = x2[i] - x1[j-1]

                # Zapisz odpowiednią wartość dla puntku z macierzy x2
                if w1 >= w2:
                    y2.append(y1[j-1])
                else:
                    y2.append(y1[j])

                break

    return y2


def main():
    x1 = np.linspace(0,5,10)
    y1 = np.sin(x1)
    x2 = np.linspace(0,5,100)

    y2 = linear(x1,y1,x2)
    plt.plot(x2, y2, '.g')

    y3 = closest_neightbour(x1,y1,x2)
    plt.plot(x2, y3, '.b')

    plt.plot(x1, y1, '.r')
    plt.show()

    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(2, 2)   

    ax = fig.add_subplot(gs[0, :])
    ax.plot(x1, y1, '.r') 
    plt.title("Oryginalny sinus")
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(x2, y3, '.b')
    plt.title("Najbliższy-sąsiad")
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(x2, y2, '.g')
    plt.title("Interpolacja liniowa")
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    fig.align_labels()
    plt.show()

if __name__ == "__main__":
    main()