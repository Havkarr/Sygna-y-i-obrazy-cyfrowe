import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.linalg import inv
from PIL import Image
import cv2  

'''
OpenCV korzysta z reprezentacji kolorów BGR zamiast RGB. !!!

Plan dalszych działań:
- Implementacja filtru Fuji
- Pobranie zdjęć R, G, B, RGB dla filtra fuji
- Poprawienie filtru Bayera aby wyświetlał kolory w kompozycji RGB a nie BGR!
- Interpolacja koloró R, G, B
- Pobrać obrazy po interpolacji
- Suma obrazów

'''


def Bayer_filter(image, matrix):
    # resize matrix
    resized_matrix = np.tile(matrix, ( image.shape[0] // matrix.shape[0], image.shape[1] // matrix.shape[1], 1))

    # Pomnóż obie macierze przez siebie (element-wise)
    result = image*resized_matrix

    # Tworzenie obrazu
    result_image = (result).astype(np.uint8)

    # Wyświetlanie obrazu
    cv2.imshow("Result Image", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.imsave('test.png', result_image)

    return result


def demosaic():
    with Image.open("image.png") as im:
        photo_array = np.array(im)

        red_Bayer_filtr = np.array([[[0,0,0], [0,0,0]],
                                [[0,0,1], [0,0,0]]])
        
        green_Bayer_filtr = np.array([[[0,1,0], [0,0,0]],
                                  [[0,0,0], [0,1,0]]])
        
        blue_Bayer_filtr = np.array([[[0,0,0], [1,0,0]],
                                 [[0,0,0], [0,0,0]]])

        red_Fuji_filtr = np.array([[[0,0,0], [0,0,0], [0,0,1], [0,0,0], [0,0,1], [0,0,0]],
                                [[0,0,1], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]],
                                [[0,0,0], [0,0,0], [0,0,0], [0,0,1], [0,0,0], [0,0,0]],
                                [[0,0,0], [0,0,1], [0,0,0], [0,0,0], [0,0,0], [0,0,1]],
                                [[0,0,0], [0,0,0], [0,0,0], [0,0,1], [0,0,0], [0,0,0]],
                                [[0,0,1], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]])

        print("Czerwo")
        red_channel = Bayer_filter(photo_array, red_Bayer_filtr)
        print("green")
        green_channel = Bayer_filter(photo_array, green_Bayer_filtr)
        print("blue")
        blue_channel = Bayer_filter(photo_array, blue_Bayer_filtr)

        return


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


def err(y1, y2):
    sum = 0
    length = len(y2)

    for i in range(length):
        sum += (y2[i] - y1[i]) ** 2

    return sum / length


def square_func(x1, y1, x2):
    # Tworzenie zmiennych
    y2 = []
    length1 = len(x1)
    length2 = len(x2)
    shift = False

    for i in range(0, length1 - 1, 2):
        if ((i + 2) >= length1):
            i -= 1
            shift = True
        elif ((i + 1) >= length1):
            return y2

        # Tworzenie tymczasowych macierzy i wektorów do obliczeń
        temp_y = []
        temp_x = [[x1[i] ** 2, x1[i], 1],
                  [x1[i + 1] ** 2, x1[i + 1], 1],
                  [x1[i + 2] ** 2, x1[i + 2], 1]]

        # Uzupełnianie macierzy y
        temp_y.append(y1[i])
        temp_y.append(y1[i + 1])
        temp_y.append(y1[i + 2])

        # Odwracanie macierzy X-ów
        xinv = inv(temp_x)

        # Mnożenie macierzy
        factors = xinv @ temp_y

        # Obliczanie odpowiednich wartości dla argumentów z odpowiedniego przedziału x2.
        for j in range(length2):
            if x1[i] <= x2[j] and x1[i + 2] > x2[j] and shift == False:
                y2.append((x2[j] ** 2) * factors[0] + x2[j] * factors[1] + factors[2])
            elif x1[i + 1] <= x2[j] and x1[i + 2] >= x2[j] and shift == True:
                y2.append((x2[j] ** 2) * factors[0] + x2[j] * factors[1] + factors[2])

        if i == (length1 - 2) and shift == False:
            y2.append((x2[j] ** 2) * factors[0] + x2[j] * factors[1] + factors[2])

    return y2


def cubic_func(x1, y1, x2):
    # Tworzenie zmiennych
    y2 = []
    length1 = len(x1)
    length2 = len(x2)
    shift = False
    opt = False

    for i in range(0, length1 - 1, 3):
        if ((i + 2) >= length1):
            i -= 2
            shift = True
            opt = False
        elif ((i + 3) >= length1):
            i -= 1
            shift = True
            opt = True
        # Tworzenie tymczasowych macierzy i wektorów do obliczeń
        temp_y = []
        temp_x = [[x1[i] ** 3, x1[i] ** 2, x1[i], 1],
                  [x1[i + 1] ** 3, x1[i + 1] ** 2, x1[i + 1], 1],
                  [x1[i + 2] ** 3, x1[i + 2] ** 2, x1[i + 2], 1],
                  [x1[i + 3] ** 3, x1[i + 3] ** 2, x1[i + 3], 1]]

        # Uzupełnianie macierzy y
        temp_y.append(y1[i])
        temp_y.append(y1[i + 1])
        temp_y.append(y1[i + 2])
        temp_y.append(y1[i + 3])

        # Odwracanie macierzy X-ów
        xinv = inv(temp_x)

        # Mnożenie macierzy
        factors = xinv @ temp_y

        # Obliczanie odpowiednich wartości dla argumentów z odpowiedniego przedziału x2.
        for j in range(length2):
            if x1[i] <= x2[j] and x1[i + 3] > x2[j] and shift == False:
                y2.append((x2[j] ** 3) * factors[0] + (x2[j] ** 2) * factors[1] + x2[j] * factors[2] + factors[3])
            elif x1[i + 2] <= x2[j] and x1[i + 3] >= x2[j] and shift == True and opt == False:
                y2.append((x2[j] ** 3) * factors[0] + (x2[j] ** 2) * factors[1] + x2[j] * factors[2] + factors[3])
            elif x1[i + 1] <= x2[j] and x1[i + 3] >= x2[j] and shift == True and opt == True:
                y2.append((x2[j] ** 3) * factors[0] + (x2[j] ** 2) * factors[1] + x2[j] * factors[2] + factors[3])

        if i == (length1 - 4) and shift == False:
            y2.append((x2[j] ** 3) * factors[0] + (x2[j] ** 2) * factors[1] + x2[j] * factors[2] + factors[3])

    return y2


def main():
    # x1 = np.linspace(0, 5, 10)
    # x2 = np.linspace(0, 5, 100)

    # # Funkcja interpolowana
    # y1 = np.sin(x1)
    # y100 = np.sin(x2)
    # # Interpolacja liniowa
    # y2 = linear(x1, y1, x2)
    # # Interpolacja najbliższy-sąsiad
    # y3 = closest_neightbour(x1, y1, x2)

    # # Stworzenie płótna i podzielenie go na sekcje
    # fig = plt.figure(tight_layout=True)
    # gs = gridspec.GridSpec(2, 2)

    # # # Tworzenie oryginalnego sinusa
    # # ax = fig.add_subplot(gs[0, :])
    # # ax.plot(x1, y1, '.r')
    # # plt.title("Oryginalny sinus")
    # # ax.set_ylabel('Y')
    # # ax.set_xlabel('X')

    # # # Tworzenie wykresu interpolacji najbliższy-sąsiad
    # # ax = fig.add_subplot(gs[1, 0])
    # # ax.plot(x2, y3, '.b')
    # # plt.title("Najbliższy-sąsiad")
    # # ax.set_ylabel('Y')
    # # ax.set_xlabel('X')

    # # # Tworzenie wykresu interpolacji liniowej
    # # ax = fig.add_subplot(gs[1, 1])
    # # ax.plot(x2, y2, '.g')
    # # plt.title("Interpolacja liniowa")
    # # ax.set_ylabel('Y')
    # # ax.set_xlabel('X')

    # # Tworzenie oryginalnego sinusa 100 punktów
    # ax = fig.add_subplot(gs[0, :])
    # ax.plot(x1, y1, '.r')
    # plt.title("Oryginalny sinus")
    # ax.set_ylabel('Y')
    # ax.set_xlabel('X')

    # # Tworzenie wykresu interpolacji funkcją kwadratową
    # y4 = square_func(x1, y1, x2)
    # ax = fig.add_subplot(gs[1, 0])
    # ax.plot(x2, y4, '-b')
    # plt.title("Kwadratowa")
    # ax.set_ylabel('Y')
    # ax.set_xlabel('X')

    # # Tworzenie wykresu funkcją sześcienną
    # y5 = cubic_func(x1, y1, x2)
    # ax = fig.add_subplot(gs[1, 1])
    # ax.plot(x2, y5, '-g')
    # plt.title("Sześcienna")
    # ax.set_ylabel('Y')
    # ax.set_xlabel('X')

    # # fig.align_labels()
    # # plt.show()

    demosaic()

if __name__ == "__main__":
    main()