import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.linalg import inv
from PIL import Image

'''
Plan dalszych działań:
- maska do filtru FUji
'''


# Nakładanie filtru na obraz
def filter(image, matrix):
    # resize matrix
    resized_matrix = np.tile(matrix,
                             ((image.shape[0] // matrix.shape[0]) + 1, (image.shape[1] // matrix.shape[1]) + 1, 1))
    resized_matrix = resized_matrix[:image.shape[0], :image.shape[1]]

    # Pomnóż obie macierze przez siebie (element-wise)
    result = image * resized_matrix

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


# Interpolacja najbliższy-sąsiad
def closest_neighbour(x1, y1, x2):
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


# Funkcja do pomiaru błędu
def err(y1, y2):
    sum = 0
    length = len(y2)

    for i in range(length):
        sum += (y2[i] - y1[i]) ** 2

    return sum / length


# Interpolacja wielomianem 2-ego stopnia
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


# Interpolacja wielomianem 3-ego stopnia
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


# Interpolacja maską
# rgb = 0 => red
# rgb = 1 => green
# rgb = 2 => blue
def mask(img, rgb):
    # mask = [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
    #         [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
    #         [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
    #         [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]]
    
    # Stwórz tymczasową macierz która zostanie uzupełniona wartośćiami z interpolacji
    temp = np.array([[[0, 0, 0]]*img.shape[1]]*img.shape[0])

    for row in range(1, img.shape[0]-2):
        for col in range(1, img.shape[1]-2):
            # Zsumuj wszystkie wartości z maski
            # Dla kolorów czerwonego i niebieskiego dzielnik zawsze będzie wynosił 4
            if img[row][col][rgb] == 0 and (rgb == 0 or rgb == 2):
                val = (img[row-1][col-1][rgb] + img[row-1][col][rgb] + img[row-1][col+1][rgb] + img[row-1][col+2][rgb] +
                       img[row][col-1][rgb] +img[row][col][rgb] +img[row][col+1][rgb] +img[row][col+2][rgb] +
                       img[row+1][col-1][rgb] +img[row+1][col][rgb] +img[row+1][col+1][rgb] +img[row+1][col+2][rgb] +
                       img[row+2][col-1][rgb] +img[row+2][col][rgb] +img[row+2][col+1][rgb] +img[row+2][col+2][rgb])/4
                temp[row][col][rgb] = val
            # Zsumuj wszystkie wartości z maski
            # Dla koloru zielonego dzielnik będzie różny dlatego trzeba zliczać wartośći niezerowe
            elif img[row][col][rgb] == 0 and rgb == 1:
                divider = 0
                values = [img[row - 1][col - 1][rgb], img[row - 1][col][rgb], img[row - 1][col + 1][rgb], img[row - 1][col + 2][rgb],
                       img[row][col - 1][rgb], img[row][col][rgb], img[row][col + 1][rgb], img[row][col + 2][rgb],
                       img[row + 1][col - 1][rgb], img[row + 1][col][rgb], img[row + 1][col + 1][rgb],
                       img[row + 1][col + 2][rgb],
                       img[row + 2][col - 1][rgb], img[row + 2][col][rgb], img[row + 2][col + 1][rgb],
                       img[row + 2][col + 2][rgb]]
                for value in values:
                    if value != 0:
                        divider += 1

                val = sum(values) / divider
                temp[row][col][rgb] = val
            else:
                temp[row][col][rgb] = img[row][col][rgb]
    
    return temp


def main():
    # x1 = np.linspace(0, 5, 10)
    # x2 = np.linspace(0, 5, 100)
    # Funkcja interpolowana
    # y1 = np.sin(x1)
    # y100 = np.sin(x2)
    # Interpolacja liniowa
    # y2 = linear(x1, y1, x2)
    # Interpolacja najbliższy-sąsiad
    # y3 = closest_neightbour(x1, y1, x2)

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

    # Nakładnanie filtru Bayera na obraz, interpolacja i połączenie obrazów.
    with Image.open("plik.jpg") as im:
        photo_array = np.array(im)

        red_Bayer_filtr = np.array([[[0, 0, 0], [1, 0, 0]],
                                    [[0, 0, 0], [0, 0, 0]]])

        green_Bayer_filtr = np.array([[[0, 1, 0], [0, 0, 0]],
                                      [[0, 0, 0], [0, 1, 0]]])

        blue_Bayer_filtr = np.array([[[0, 0, 0], [0, 0, 0]],
                                     [[0, 0, 1], [0, 0, 0]]])

        red_Fuji_filtr = np.array([[[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]],
                                   [[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                   [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]],
                                   [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]],
                                   [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]],
                                   [[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]])

        green_Fuji_filtr = np.array([[[0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]],
                                     [[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0]],
                                     [[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0]],
                                     [[0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]],
                                     [[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0]],
                                     [[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0]]])

        blue_Fuji_filtr = np.array([[[0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1]],
                                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]],
                                    [[0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                    [[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 1], [0, 0, 0]],
                                    [[0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]]])

        red = filter(photo_array, red_Bayer_filtr)
        green = filter(photo_array, green_Bayer_filtr)
        blue = filter(photo_array, blue_Bayer_filtr)

        
        red_Fuji = filter(photo_array, red_Fuji_filtr)
        green_Fuji = filter(photo_array, green_Fuji_filtr)
        blue_Fuji = filter(photo_array, blue_Fuji_filtr)

        red_F = mask(red_Fuji, 0)
        green_F = mask(green_Fuji, 1)
        blue_F = mask(blue_Fuji, 2)

        result_image = (red_F).astype(np.uint8)
        plt.imshow(result_image)
        plt.show()

        result_image = (green_F).astype(np.uint8)
        plt.imshow(result_image)
        plt.show()

        result_image = (blue_F).astype(np.uint8)
        plt.imshow(result_image)
        plt.show()

        res = red_F + green_F + blue_F

        result_image = (res).astype(np.uint8)
        plt.imshow(result_image)
        plt.show()


        # Interpolacja z filtrów bayera
        ############################################################################################33
        # Kolor czerwony
        # Działanie na wierszach

        temp1 = np.linspace(0, photo_array.shape[1], photo_array.shape[1])
        temp2 = np.linspace(0, photo_array.shape[1], int(photo_array.shape[1]/2))

        for i in range(0, photo_array.shape[0], 2):
            row = []
            for j in range(1,photo_array.shape[1], 2):
                row.append(red[i, j, 0])
            new_row = linear(temp2,row,temp1)
            red[i,:,0] = new_row

        # # Działanie na kolumnach
        temp3 = np.linspace(0, photo_array.shape[0], photo_array.shape[0])
        temp4 = np.linspace(0, photo_array.shape[0], int(photo_array.shape[0]/2))

        for i in range(0, photo_array.shape[1]):
            column = []
            for j in range(0,photo_array.shape[0], 2):
                column.append(red[j, i, 0])
            new_column = linear(temp4,column,temp3)
            red[:,i,0] = new_column

        result_image = (red).astype(np.uint8)
        plt.imshow(result_image)
        plt.show()

        ############################################################################################33
        # Kolor zielony (odpowiedni zmodyfikowany)
        # Działanie na wierszach

        for i in range(0, photo_array.shape[0], 2):
            row = []
            for j in range(0,photo_array.shape[1], 2):
                row.append(green[i, j, 1])
            new_row = linear(temp2,row,temp1)
            green[i,:,1] = new_row

        # Działanie na kolumnach
        for i in range(1, photo_array.shape[0], 2):
            row = []
            for j in range(1,photo_array.shape[1], 2):
                row.append(green[i, j, 1])
            new_row = linear(temp2,row,temp1)
            green[i,:,1] = new_row

        result_image = (green).astype(np.uint8)
        plt.imshow(result_image)
        plt.show()

        ##################################################################################################
        # Kolor niebieski (również odpowiednio zmodyfikowany)
        # Działanie na wierszach

        for i in range(1, photo_array.shape[0], 2):
            row = []
            for j in range(0,photo_array.shape[1], 2):
                row.append(blue[i, j, 2])
            new_row = linear(temp2,row,temp1)
            blue[i,:,2] = new_row

        # # Działanie na kolumnach

        for i in range(0, photo_array.shape[1]):
            column = []
            for j in range(1,photo_array.shape[0], 2):
                column.append(blue[j, i, 2])
            new_column = linear(temp4,column,temp3)
            blue[:,i,2] = new_column

        result_image = (blue).astype(np.uint8)
        plt.imshow(result_image)
        plt.show()

        # Suma obrazów
        rm = red + blue + green

        result_image = (rm).astype(np.uint8)
        plt.imshow(result_image)
        plt.show()

        # Interpolacja z filtrów Fuji
        ############################################################################################33
        # Kolor czerwony
        # Działanie na wierszach
        temp = np.linspace(0, photo_array.shape[1], photo_array.shape[1])
        for i in range(photo_array.shape[0]):
            tempY = []
            tempX = []
            for j in range(photo_array.shape[1]):
                if red_Fuji[i][j][0] != 0:
                    tempY.append(red_Fuji[i][j][0])
                    tempX.append(j)
            tempX.append(j + 1)
            tempY.append(tempY[-1])
            new_row = linear(tempX, tempY, temp)
            red_Fuji[i, :, 0] = new_row

        result_image = (red_Fuji).astype(np.uint8)
        plt.imshow(result_image)
        plt.show()

        ############################################################################################33
        # Kolor zielony
        # Działanie na wierszach
        for i in range(photo_array.shape[0]):
            tempY = []
            tempX = []
            for j in range(photo_array.shape[1]):
                if green_Fuji[i][j][1] != 0:
                    tempY.append(green_Fuji[i][j][1])
                    tempX.append(j)
            tempX.append(j + 1)
            tempY.append(tempY[-1])
            new_row = linear(tempX, tempY, temp)
            green_Fuji[i, :, 1] = new_row

        result_image = (green_Fuji).astype(np.uint8)
        plt.imshow(result_image)
        plt.show()

        ############################################################################################33
        # Kolor niebieski
        # Działanie na wierszach
        temp = np.linspace(0, photo_array.shape[1], photo_array.shape[1])
        for i in range(photo_array.shape[0]):
            tempY = []
            tempX = []
            for j in range(photo_array.shape[1]):
                if blue_Fuji[i][j][2] != 0:
                    tempY.append(blue_Fuji[i][j][2])
                    tempX.append(j)
            tempX.append(j + 1)
            tempY.append(tempY[-1])
            new_row = linear(tempX, tempY, temp)
            blue_Fuji[i, :, 2] = new_row

        result_image = (blue_Fuji).astype(np.uint8)
        plt.imshow(result_image)
        plt.show()

        result = red_Fuji + blue_Fuji + green_Fuji

        result_image = (result).astype(np.uint8)
        plt.imshow(result_image)
        plt.show()


if __name__ == "__main__":
    main()