import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.linalg import inv
from PIL import Image

'''
OpenCV korzysta z reprezentacji kolorów BGR zamiast RGB. !!!

Plan dalszych działań:
- porównać zdjęcia za pomocą odejmowania obrazów
- Interpolacja koloró R, G, B
- Pobrać obrazy po interpolacji
- Suma obrazów

'''

def filter(image, matrix):
    # resize matrix
    resized_matrix = np.tile(matrix, ( (image.shape[0] // matrix.shape[0])+1, (image.shape[1] // matrix.shape[1])+1, 1))
    resized_matrix = resized_matrix[:image.shape[0], :image.shape[1]]

    # Pomnóż obie macierze przez siebie (element-wise)
    result = image*resized_matrix

    # Tworzenie obrazu
    result_image = (result).astype(np.uint8)

    plt.imshow(result_image)
    plt.show()

    return result


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

def linear_interpolation(x1, y1, x2):
    y2 = []

    # Indeks dla poprzedniego punktu w x1
    prev_index = 0

    for x2_val in x2:
        # Znajdź indeks dla x2_val w x1, zaczynając od poprzedniego indeksu
        while prev_index < len(x1) - 1 and x1[prev_index + 1] < x2_val:
            prev_index += 1

        # Sprawdź czy x2_val znajduje się pomiędzy x1[prev_index] i x1[prev_index + 1]
        if prev_index < len(x1) - 1:
            # Oblicz wagi na podstawie odległości nowego punktu od sąsiednich punktów
            w1 = x1[prev_index + 1] - x2_val
            w2 = x2_val - x1[prev_index]

            # Oblicz wartość dla punktu z x2 i dodaj ją do listy y2
            interpolated_value = (y1[prev_index] * w1 + y1[prev_index + 1] * w2) / (w1 + w2)

            y2.append(float(interpolated_value))
        else:
            # Jeśli x2_val przekracza największą wartość w x1, użyj ostatniej wartości y1
            y2.append(float(y1[-1]))

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
    # Funkcja interpolowana
    # y1 = np.sin(x1)
    # y100 = np.sin(x2)
    # Interpolacja liniowa
    #y2 = linear(x1, y1, x2)
    # Interpolacja najbliższy-sąsiad
    #y3 = closest_neightbour(x1, y1, x2)

    # Stworzenie płótna i podzielenie go na sekcje
    # fig = plt.figure(tight_layout=True)
    # gs = gridspec.GridSpec(2, 2)

    # Tworzenie oryginalnego sinusa
    # ax = fig.add_subplot(gs[0, :])
    # ax.plot(x1, y1, '.r')
    # plt.title("Oryginalny sinus")
    # ax.set_ylabel('Y')
    # ax.set_xlabel('X')

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

    # Tworzenie oryginalnego sinusa 100 punktów
    # ax = fig.add_subplot(gs[0, :])
    # ax.plot(x1, y1, '.r')
    # plt.title("Oryginalny sinus")
    # ax.set_ylabel('Y')
    # ax.set_xlabel('X')

    # Tworzenie wykresu interpolacji funkcją kwadratową
    # y4 = square_func(x1, y1, x2)
    # ax = fig.add_subplot(gs[1, :])
    # ax.plot(x2, y4, '.b')
    # plt.title("Kwadratowa")
    # ax.set_ylabel('Y')
    # ax.set_xlabel('X')

    # # Tworzenie wykresu funkcją sześcienną
    # y5 = cubic_func(x1, y1, x2)
    # ax = fig.add_subplot(gs[1, :])
    # ax.plot(x2, y5, '.g')
    # plt.title("Sześcienna")
    # ax.set_ylabel('Y')
    # ax.set_xlabel('X')

    # fig.align_labels()
    # plt.show()

    with Image.open("image.png") as im:
        photo_array = np.array(im)

        red_Bayer_filtr = np.array([[[0,0,0], [0,0,0]],
                                [[1,0,0], [0,0,0]]])
        
        green_Bayer_filtr = np.array([[[0,1,0], [0,0,0]],
                                  [[0,0,0], [0,1,0]]])
        
        blue_Bayer_filtr = np.array([[[0,0,0], [0,0,1]],
                                 [[0,0,0], [0,0,0]]])

        red_Fuji_filtr = np.array([[[0,0,0], [0,0,0], [1,0,0], [0,0,0], [1,0,0], [0,0,0]],
                                [[1,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]],
                                [[0,0,0], [0,0,0], [0,0,0], [1,0,0], [0,0,0], [0,0,0]],
                                [[0,0,0], [1,0,0], [0,0,0], [0,0,0], [0,0,0], [1,0,0]],
                                [[0,0,0], [0,0,0], [0,0,0], [1,0,0], [0,0,0], [0,0,0]],
                                [[1,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]])

        green_Fuji_filtr = np.array([[[0,1,0], [0,0,0], [0,0,0], [0,1,0], [0,0,0], [0,0,0]],
                                [[0,0,0], [0,1,0], [0,1,0], [0,0,0], [0,1,0], [0,1,0]],
                                [[0,0,0], [0,1,0], [0,1,0], [0,0,0], [0,1,0], [0,1,0]],
                                [[0,1,0], [0,0,0], [0,0,0], [0,1,0], [0,0,0], [0,0,0]],
                                [[0,0,0], [0,1,0], [0,1,0], [0,0,0], [0,1,0], [0,1,0]],
                                [[0,0,0], [0,1,0], [0,1,0], [0,0,0], [0,1,0], [0,1,0]]])
        
        blue_Fuji_filtr = np.array([[[0,0,0], [0,0,1], [0,0,0], [0,0,0], [0,0,0], [0,0,1]],
                                [[0,0,0], [0,0,0], [0,0,0], [0,0,1], [0,0,0], [0,0,0]],
                                [[0,0,1], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]],
                                [[0,0,0], [0,0,0], [0,0,1], [0,0,0], [0,0,1], [0,0,0]],
                                [[0,0,1], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]],
                                [[0,0,0], [0,0,0], [0,0,0], [0,0,1], [0,0,0], [0,0,0]]])

        red = filter(photo_array, red_Bayer_filtr)
        green = filter(photo_array, green_Bayer_filtr)
        blue = filter(photo_array, blue_Bayer_filtr)

        

        # # Działanie na wierszach
        # temp1 = np.linspace(0, photo_array.shape[1], photo_array.shape[1])
        # temp2 = np.linspace(0, photo_array.shape[1], photo_array.shape[1])

        # for i in range(1, photo_array.shape[0], 2):
        #     new_row = linear_interpolation(temp2,red[i, :, 0],temp1)
        #     for j in range(0,photo_array.shape[1]):
        #         red[i,j,0] = new_row[j]

        # # Działanie na kolumnach
        # temp1 = np.linspace(0, photo_array.shape[0], photo_array.shape[0])
        # temp2 = np.linspace(0, photo_array.shape[0], photo_array.shape[0])

        # for i in range(1, photo_array.shape[1]):
        #     new_column = linear_interpolation(temp2,red[:, i, 0],temp1)
        #     for j in range(0,photo_array.shape[0]):
        #         red[j,i,0] = new_column[j]
        
        # Działanie na wierszach
        temp1 = np.linspace(0, photo_array.shape[1], photo_array.shape[1])
        temp2 = np.linspace(0, photo_array.shape[1], int(photo_array.shape[1]/2))

        for i in range(1, photo_array.shape[0], 2): 
            row = []
            for j in range(0,photo_array.shape[1], 2):
                row.append(red[i, j, 0])
            new_row = linear(temp2,row,temp1)
            red[i,:,0] = new_row

        # Działanie na kolumnach
        temp1 = np.linspace(0, photo_array.shape[0], photo_array.shape[0])
        temp2 = np.linspace(0, photo_array.shape[0], int(photo_array.shape[0]/2))

        for i in range(0, photo_array.shape[1]): 
            column = []
            for j in range(1,photo_array.shape[0], 2):
                column.append(red[j, i, 0])
            new_column = linear(temp2,column,temp1)
            red[:,i,0] = new_column

        result_image = (red).astype(np.uint8)
        plt.imshow(result_image)
        plt.show()
############################################################################################33
        # Działanie na wierszach
        temp1 = np.linspace(0, photo_array.shape[1], photo_array.shape[1])
        temp2 = np.linspace(0, photo_array.shape[1], photo_array.shape[1])

        for i in range(0, photo_array.shape[0], 1): 
            row = []
            for j in range(0,photo_array.shape[1], 1):
                row.append(green[i, j, 1])
            new_row = linear(temp2,row,temp1)
            green[i,:,1] = new_row

        # Działanie na kolumnach
        temp1 = np.linspace(0, photo_array.shape[0], photo_array.shape[0])
        temp2 = np.linspace(0, photo_array.shape[0], int(photo_array.shape[0]/2))

        for i in range(0, photo_array.shape[1]): 
            column = []
            for j in range(0,photo_array.shape[0]):
                column.append(green[j, i, 1])
            new_column = linear(temp2,column,temp1)
            green[:,i,1] = new_column

        result_image = (green).astype(np.uint8)
        plt.imshow(result_image)
        plt.show()
##################################################################################################
         # Działanie na wierszach
        temp1 = np.linspace(0, photo_array.shape[1], photo_array.shape[1])
        temp2 = np.linspace(0, photo_array.shape[1], int(photo_array.shape[1]/2))

        for i in range(0, photo_array.shape[0], 2): 
            row = []
            for j in range(1,photo_array.shape[1], 2):
                row.append(blue[i, j, 2])
            new_row = linear(temp2,row,temp1)
            blue[i,:,2] = new_row

        # Działanie na kolumnach
        temp1 = np.linspace(0, photo_array.shape[0], photo_array.shape[0])
        temp2 = np.linspace(0, photo_array.shape[0], int(photo_array.shape[0]/2))

        for i in range(0, photo_array.shape[1]): 
            column = []
            for j in range(0,photo_array.shape[0], 2):
                column.append(blue[j, i, 2])
            new_column = linear(temp2,column,temp1)
            blue[:,i,2] = new_column
        
        result_image = (blue).astype(np.uint8)
        plt.imshow(result_image)
        plt.show()

        rm = red + blue + green

        result_image = (rm).astype(np.uint8)
        plt.imshow(result_image)
        plt.show()


if __name__ == "__main__":
    main()