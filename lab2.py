import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.linalg import inv


# Interpolacja liniowa
def linear(x1, y1, x2):
    y2 = []

    # Porównaj każdą wartość z wektora x2 z wartościami wektora x1 aby sprawdzić w którym przedziale znajduje się punkt.
    for i in range(len(x2)):
        for j in range(len(x1)):
            if x2[i] <= x1[j]:
                # Oblicz wagi na podstawie odległości nowego punktu od 2 sąsiadujących
                w1 = x1[j] - x2[i]
                w2 = x2[i] - x1[j - 1]

                # Oblicz wartość dla punktu z wektora x2 i dodaj ją do macierzy y2
                temp = (y1[j] * w2 + y1[j - 1] * w1) / (w1 + w2)

                y2.append(float(temp))
                break

    return y2


# Interpolacja najbliższy-sąsiad
def closest_neightbour(x1, y1, x2):
    y2 = []

    # Porównaj każdą wartość z wektora x2 z wartościami wektora x1 aby sprawdzić do którego punktu z macierzy x1 jest bliżej
    for i in range(len(x2)):
        for j in range(len(x1)):
            if x2[i] <= x1[j]:
                w1 = x1[j] - x2[i]
                w2 = x2[i] - x1[j - 1]

                # Zapisz odpowiednią wartość dla puntku z macierzy x2
                if w1 >= w2:
                    y2.append(y1[j - 1])
                else:
                    y2.append(y1[j])

                break

    return y2

def err(y1,y2):

    sum = 0
    length = len(y2)

    for i in range(length):
        sum += (y2[i] - y1[i])**2

    return sum/length


def square_func(x1,y1,x2):
    '''
    wybierz kolejne 3 wartosci x1 i y1
    Zapakuj je do macierzy i wektora
    pomnóż odwrotność wektora x z wektorem y
    wektor wynikowy zawiera wartosci dla funkcji kwadratowej
    pomnóż wartości x2 z funkcją kwadratową
    wpisz wyniki do macierzy y2
    '''
    # Tworzenie zmiennych
    y2 = []
    length1 = len(x1)
    length2 = len(x2)
    shift = False

    for i in range (0, length1, 2):
        if ((i + 2) >= length1):
            i -= 1
            shift = True

        # Tworzenie tymczasowych macierzy i wektorów do obliczeń
        temp_y = []
        temp_x = [[1,1,1],
                  [1,1,1],
                  [1,1,1]]

        # Wypełnnianie macierzy i wektorów. Inkrementacja zmiennej
        temp_x[0][1] = x1[i]
        temp_x[0][0] = temp_x[0][1] ** 2
        temp_y.append(y1[i])
        i += 1

        temp_x[1][1] = x1[i]
        temp_x[1][0] = temp_x[1][1] ** 2
        temp_y.append(y1[i])
        i += 1

        temp_x[2][1] = x1[i]
        temp_x[2][0] = temp_x[2][1] ** 2
        temp_y.append(y1[i])

        # Odwracanie macierzy X-ów
        xinv = inv(temp_x)

        # Mnożenie macierzy
        abc = xinv@temp_y

        for j in range(length2):
            if x1[i-2] <= x2[j] and x1[i] > x2[j] and shift == False:
                y2.append((x2[j]**2)*abc[0] + x2[j]*abc[1] + abc[2])
            elif x1[i-1] <= x2[j] and x1[i] >= x2[j] and shift == True:
                y2.append((x2[j]**2)*abc[0] + x2[j]*abc[1] + abc[2])
    return y2


def main():
    x1 = np.linspace(0, 5, 10)
    x2 = np.linspace(0, 5, 100)

    # Funkcja interpolowana
    y1 = np.sin(x1)
    y100 = np.sin(x2)
    # Interpolacja liniowa
    y2 = linear(x1, y1, x2)
    # Interpolacja najbliższy-sąsiad
    y3 = closest_neightbour(x1, y1, x2)

    # Stworzenie płótna i podzielenie go na sekcje
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(2, 2)

    # Tworzenie oryginalnego sinusa
    ax = fig.add_subplot(gs[0, :])
    ax.plot(x1, y1, '.r')
    plt.title("Oryginalny sinus")
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    # Tworzenie wykresu interpolacji najbliższy-sąsiad
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(x2, y100, '.b')
    plt.title("Najbliższy-sąsiad")
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    y200 = square_func(x1,y1,x2)
    # Tworzenie wykresu interpolacji liniowej
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(x2, y200, '.g')
    plt.title("Interpolacja liniowa")
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    fig.align_labels()
    plt.show()

    #plt.plot(x2, y200, '-g')
    #plt.show()

if __name__ == "__main__":
    main()