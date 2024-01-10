import cv2 as cv
import numpy as np
import math
import time
from matplotlib import pyplot as plt
from PIL import Image


print("SELECT THE ENHANCEMENT TECHNIQUES :")
print("1. Fuzzy set ")
print("2. Histogram Equalization ")
print("3. Comparison ")
n= int(input("Enter Choice -> "))
imagee = "chest_xray.jpeg"




#clahe technique
#histogram specify
#contrast streching
#threshold




# FUZZY SET

if n==1:

    n = 2  # number of rows (windows on columns)
    m = 2  # number of colomns (windows on rows)
    EPSILON = 0.00001
    # GAMMA, IDEAL_VARIANCE 'maybe' have to changed from image to another
    GAMMA = 1  # Big GAMMA >> Big mean >> More Brightness
    IDEAL_VARIANCE = 0.35  # Big value >> Big variance >> Big lamda >> more contrast

    # img_name = 'R'
    # img = cv.imread('C:/Users/arora/PycharmProjects/pythonProject'+img_name)
    img = cv.imread(imagee)
    # img = cv.resize(img, (200, 200))
    layer = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    WIDTH = layer.shape[1]
    HEIGHT = layer.shape[0]
    x0, x1, y0, y1 = 0, WIDTH - 1, 0, HEIGHT - 1


    # split the image to windows
    def phy(value):  # phy: E --> R
        # if ((1+value)/((1-value)+0.0001)) < 0:
        # print(value)
        return 0.5 * np.log((1 + value) / ((1 - value) + EPSILON))


    def multiplication(value1, value2):  # ExE --> R
        return phy(value1) * phy(value2)


    def norm(value):
        return abs(phy(value))


    def scalar_multiplication(scalar, value):  # value in E ([-1,1])
        s = (1 + value) ** scalar
        z = (1 - value) ** scalar
        res = (s - z) / (s + z + EPSILON)
        return res


    def addition(value1, value2):  # value1,value2 are in E ([-1,1])
        res = (value1 + value2) / (1 + (value1 * value2) + EPSILON)
        return res


    def subtract(value1, value2):  # value1,value2 are in E ([-1,1])
        res = (value1 - value2) / (1 - (value1 * value2) + EPSILON)
        return res


    def C(m, i):
        return math.factorial(m) / ((math.factorial(i) * math.factorial(m - i)) + EPSILON)


    def qx(i, x):  # i: window index in rows, x: number of current pixel on x-axis
        if (x == WIDTH - 1):
            return 0
        return C(m, i) * (np.power((x - x0) / (x1 - x), i) * np.power((x1 - x) / (x1 - x0),
                                                                      m))  # This is the seconf implementation
        # return C(m,i)*((np.power(x-x0,i) * np.power(x1-x,m-i)) / (np.power(x1-x0,m)+EPSILON))


    def qy(j, y):
        '''
        The second implementation for the formula does not go into overflow.
        '''
        if (y == HEIGHT - 1):
            return 0
        return C(n, j) * (np.power((y - y0) / (y1 - y), j) * np.power((y1 - y) / (y1 - y0),
                                                                      n))  # This is the seconf implementation
        # return C(n,j)*((np.power((y-y0),j) * np.power((y1-y),n-j))/ (np.power(y1-y0,n)+EPSILON))


    def p(i, j, x, y):
        return qx(i, x) * qy(j, y)


    def mapping(img, source, dest):
        return (dest[1] - dest[0]) * ((img - source[0]) / (source[1] - source[0])) + dest[0]


    e_layer_gray = mapping(layer, (0, 255), (-1, 1))


    def cal_ps_ws(m, n, w, h, gamma):
        ps = np.zeros((m, n, w, h))
        for i in range(m):
            for j in range(n):
                for k in range(w):
                    for l in range(h):
                        ps[i, j, k, l] = p(i, j, k, l)

        ws = np.zeros((m, n, w, h))
        for i in range(m):
            for j in range(n):
                ps_power_gamma = np.power(ps[i, j], gamma)
                for k in range(w):
                    for l in range(h):
                        ws[i, j, k, l] = ps_power_gamma[k, l] / (np.sum(ps[:, :, k, l]) + EPSILON)
        return ps, ws


    print('Ps and Ws calculation is in progress...')
    start = time.time()
    ps, ws = cal_ps_ws(m, n, WIDTH, HEIGHT, GAMMA)
    end = time.time()
    print('Ps and Ws calculation has completed successfully in ' + str(end - start) + ' s')


    def cal_means_variances_lamdas(w, e_layer):
        means = np.zeros((m, n))
        variances = np.zeros((m, n))
        lamdas = np.zeros((m, n))
        taos = np.zeros((m, n))

        def window_card(w):
            return np.sum(w)

        def window_mean(w, i, j):
            mean = 0
            for k in range(HEIGHT):
                for l in range(WIDTH):
                    mean = addition(mean, scalar_multiplication(w[i, j, l, k], e_layer[k, l]))
            mean /= window_card(w[i, j])
            return mean

        def window_variance(w, i, j):
            variance = 0
            for k in range(HEIGHT):
                for l in range(WIDTH):
                    variance += w[i, j, l, k] * np.power(norm(subtract(e_layer[k, l], means[i, j])), 2)
            variance /= window_card(w[i, j])
            return variance

        def window_lamda(w, i, j):
            return np.sqrt(IDEAL_VARIANCE) / (np.sqrt(variances[i, j]) + EPSILON)

        def window_tao(w, i, j):
            return window_mean(w, i, j)

        for i in range(m):
            for j in range(n):
                means[i, j] = window_mean(ws, i, j)
                variances[i, j] = window_variance(ws, i, j)
                lamdas[i, j] = window_lamda(ws, i, j)
        taos = means.copy()

        return means, variances, lamdas, taos


    print('means, variances, lamdas and taos calculation is in progress...')
    start = time.time()
    means, variances, lamdas, taos = cal_means_variances_lamdas(ws, e_layer_gray)
    end = time.time()
    print('means, variances, lamdas and taos calculation is finished in ' + str(end - start) + ' s')


    def window_enh(w, i, j, e_layer):
        return scalar_multiplication(lamdas[i, j], subtract(e_layer, taos[i, j]))


    def image_enh(w, e_layer):
        new_image = np.zeros(e_layer.shape)
        width = e_layer.shape[1]
        height = e_layer.shape[0]
        for i in range(m):
            for j in range(n):
                win = window_enh(w, i, j, e_layer)
                w1 = w[i, j].T.copy()
                for k in range(width):
                    for l in range(height):
                        new_image[l, k] = addition(new_image[l, k], scalar_multiplication(w1[l, k], win[l, k]))
        return new_image


    def one_layer_enhacement(e_layer):
        # card_image = layer.shape[0]*layer.shape[1]
        new_E_image = image_enh(ws, e_layer)
        res_image = mapping(new_E_image, (-1, 1), (0, 255))
        res_image = np.round(res_image)
        res_image = res_image.astype(np.uint8)
        return res_image


    res_img = one_layer_enhacement(e_layer_gray)

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(res_img, cmap='gray')
    plt.title('Fuzzy Grayscale image enhacement.')
    plt.show()


# HISTOGRAM EQUALIZATION

elif n==2:
    save_filename = 'Equalized_Hawkes_Bay_NZ.jpg'

    # load file as pillow Image
    img = Image.open("chest_xray.jpeg")

    # convert to grayscale
    imgray = img.convert(mode='L')

    # convert to NumPy array
    img_array = np.asarray(imgray)

    # flatten image array and calculate histogram via binning
    histogram_array = np.bincount(img_array.flatten(), minlength=256)

    # normalize
    num_pixels = np.sum(histogram_array)
    histogram_array = histogram_array / num_pixels

    # cumulative histogram
    chistogram_array = np.cumsum(histogram_array)

    transform_map = np.floor(255 * chistogram_array).astype(np.uint8)

    # flatten image array into 1D list
    img_list = list(img_array.flatten())

    # transform pixel values to equalize
    eq_img_list = [transform_map[p] for p in img_list]

    # reshape and write back into img_array
    eq_img_array = np.reshape(np.asarray(eq_img_list), img_array.shape)

    # Let's plot the histograms

    # histogram and cumulative histogram of original image has been calculated above
    ori_cdf = chistogram_array
    ori_pdf = histogram_array

    # calculate histogram and cumulative histogram of equalized image
    eq_histogram_array = np.bincount(eq_img_array.flatten(), minlength=256)
    num_pixels = np.sum(eq_histogram_array)
    eq_pdf = eq_histogram_array / num_pixels
    eq_cdf = np.cumsum(eq_pdf)




    eq_img = Image.fromarray(eq_img_array, mode='L')
    #eq_img.show()
   # eq_img.save(save_filename)

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(eq_img, cmap='gray')
    plt.title('Histogram Equalization')
    plt.show()

elif n==3:

    n = 2  # number of rows (windows on columns)
    m = 2  # number of colomns (windows on rows)
    EPSILON = 0.00001
    # GAMMA, IDEAL_VARIANCE 'maybe' have to changed from image to another
    GAMMA = 1  # Big GAMMA >> Big mean >> More Brightness
    IDEAL_VARIANCE = 0.35  # Big value >> Big variance >> Big lamda >> more contrast

    # img_name = 'R'
    # img = cv.imread('C:/Users/arora/PycharmProjects/pythonProject'+img_name)
    img = cv.imread(imagee)
    # img = cv.resize(img, (200, 200))
    layer = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    WIDTH = layer.shape[1]
    HEIGHT = layer.shape[0]
    x0, x1, y0, y1 = 0, WIDTH - 1, 0, HEIGHT - 1


    # split the image to windows the basi need lf the bosy
    def phy(value):  # phy: E --> R
        # if ((1+value)/((1-value)+0.0001)) < 0:
        # print(value)
        return 0.5 * np.log((1 + value) / ((1 - value) + EPSILON))


    def multiplication(value1, value2):  # ExE --> R
        return phy(value1) * phy(value2)


    def norm(value):
        return abs(phy(value))


    def scalar_multiplication(scalar, value):  # value in E ([-1,1])
        s = (1 + value) ** scalar
        z = (1 - value) ** scalar
        res = (s - z) / (s + z + EPSILON)
        return res


    def addition(value1, value2):  # value1,value2 are in E ([-1,1])
        res = (value1 + value2) / (1 + (value1 * value2) + EPSILON)
        return res


    def subtract(value1, value2):  # value1,value2 are in E ([-1,1])
        res = (value1 - value2) / (1 - (value1 * value2) + EPSILON)
        return res


    def C(m, i):
        return math.factorial(m) / ((math.factorial(i) * math.factorial(m - i)) + EPSILON)


    def qx(i, x):  # i: window index in rows, x: number of current pixel on x-axis
        if (x == WIDTH - 1):
            return 0
        return C(m, i) * (np.power((x - x0) / (x1 - x), i) * np.power((x1 - x) / (x1 - x0),
                                                                      m))  # This is the seconf implementation
        # return C(m,i)*((np.power(x-x0,i) * np.power(x1-x,m-i)) / (np.power(x1-x0,m)+EPSILON))


    def qy(j, y):
        '''
        The second implementation for the formula does not go into overflow.
        '''
        if (y == HEIGHT - 1):
            return 0
        return C(n, j) * (np.power((y - y0) / (y1 - y), j) * np.power((y1 - y) / (y1 - y0),
                                                                      n))  # This is the seconf implementation
        # return C(n,j)*((np.power((y-y0),j) * np.power((y1-y),n-j))/ (np.power(y1-y0,n)+EPSILON))


    def p(i, j, x, y):
        return qx(i, x) * qy(j, y)


    def mapping(img, source, dest):
        return (dest[1] - dest[0]) * ((img - source[0]) / (source[1] - source[0])) + dest[0]


    e_layer_gray = mapping(layer, (0, 255), (-1, 1))


    def cal_ps_ws(m, n, w, h, gamma):
        ps = np.zeros((m, n, w, h))
        for i in range(m):
            for j in range(n):
                for k in range(w):
                    for l in range(h):
                        ps[i, j, k, l] = p(i, j, k, l)

        ws = np.zeros((m, n, w, h))
        for i in range(m):
            for j in range(n):
                ps_power_gamma = np.power(ps[i, j], gamma)
                for k in range(w):
                    for l in range(h):
                        ws[i, j, k, l] = ps_power_gamma[k, l] / (np.sum(ps[:, :, k, l]) + EPSILON)
        return ps, ws


    print('Ps and Ws calculation is in progress...')
    start = time.time()
    ps, ws = cal_ps_ws(m, n, WIDTH, HEIGHT, GAMMA)
    end = time.time()
    print('Ps and Ws calculation has completed successfully in ' + str(end - start) + ' s')


    def cal_means_variances_lamdas(w, e_layer):
        means = np.zeros((m, n))
        variances = np.zeros((m, n))
        lamdas = np.zeros((m, n))
        taos = np.zeros((m, n))

        def window_card(w):
            return np.sum(w)

        def window_mean(w, i, j):
            mean = 0
            for k in range(HEIGHT):
                for l in range(WIDTH):
                    mean = addition(mean, scalar_multiplication(w[i, j, l, k], e_layer[k, l]))
            mean /= window_card(w[i, j])
            return mean

        def window_variance(w, i, j):
            variance = 0
            for k in range(HEIGHT):
                for l in range(WIDTH):
                    variance += w[i, j, l, k] * np.power(norm(subtract(e_layer[k, l], means[i, j])), 2)
            variance /= window_card(w[i, j])
            return variance

        def window_lamda(w, i, j):
            return np.sqrt(IDEAL_VARIANCE) / (np.sqrt(variances[i, j]) + EPSILON)

        def window_tao(w, i, j):
            return window_mean(w, i, j)

        for i in range(m):
            for j in range(n):
                means[i, j] = window_mean(ws, i, j)
                variances[i, j] = window_variance(ws, i, j)
                lamdas[i, j] = window_lamda(ws, i, j)
        taos = means.copy()

        return means, variances, lamdas, taos


    print('means, variances, lamdas and taos calculation is in progress...')
    start = time.time()
    means, variances, lamdas, taos = cal_means_variances_lamdas(ws, e_layer_gray)
    end = time.time()
    print('means, variances, lamdas and taos calculation is finished in ' + str(end - start) + ' s')


    def window_enh(w, i, j, e_layer):
        return scalar_multiplication(lamdas[i, j], subtract(e_layer, taos[i, j]))


    def image_enh(w, e_layer):
        new_image = np.zeros(e_layer.shape)
        width = e_layer.shape[1]
        height = e_layer.shape[0]
        for i in range(m):
            for j in range(n):
                win = window_enh(w, i, j, e_layer)
                w1 = w[i, j].T.copy()
                for k in range(width):
                    for l in range(height):
                        new_image[l, k] = addition(new_image[l, k], scalar_multiplication(w1[l, k], win[l, k]))
        return new_image


    def one_layer_enhacement(e_layer):
        # card_image = layer.shape[0]*layer.shape[1]
        new_E_image = image_enh(ws, e_layer)
        res_image = mapping(new_E_image, (-1, 1), (0, 255))
        res_image = np.round(res_image)
        res_image = res_image.astype(np.uint8)
        return res_image


    res_img = one_layer_enhacement(e_layer_gray)

    #########################################################################
    ###########################################################################
    ##########################################################################

    save_filename = 'Equalized_Hawkes_Bay_NZ.jpg'

    # load file as pillow Image
    img = Image.open("chest_xray.jpeg")

    # convert to grayscale
    imgray = img.convert(mode='L')

    # convert to NumPy array
    img_array = np.asarray(imgray)

    # flatten image array and calculate histogram via binning
    histogram_array = np.bincount(img_array.flatten(), minlength=256)

    # normalize
    num_pixels = np.sum(histogram_array)
    histogram_array = histogram_array / num_pixels

    # cumulative histogram
    chistogram_array = np.cumsum(histogram_array)

    transform_map = np.floor(255 * chistogram_array).astype(np.uint8)

    # flatten image array into 1D list
    img_list = list(img_array.flatten())

    # transform pixel values to equalize
    eq_img_list = [transform_map[p] for p in img_list]

    # reshape and write back into img_array
    eq_img_array = np.reshape(np.asarray(eq_img_list), img_array.shape)

    # Let's plot the histograms

    # histogram and cumulative histogram of original image has been calculated above
    ori_cdf = chistogram_array
    ori_pdf = histogram_array

    # calculate histogram and cumulative histogram of equalized image
    eq_histogram_array = np.bincount(eq_img_array.flatten(), minlength=256)
    num_pixels = np.sum(eq_histogram_array)
    eq_pdf = eq_histogram_array / num_pixels
    eq_cdf = np.cumsum(eq_pdf)

    eq_img = Image.fromarray(eq_img_array, mode='L')
    # eq_img.show()
# eq_img.save(save_filename)

        ################################################################################################
        ################################################################################################
        ################################################################################################

    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(res_img, cmap='gray')
    plt.title('Fuzzy Grayscale')


    plt.subplot(1, 3, 3)
    plt.imshow(eq_img, cmap='gray')
    plt.title('Histogram Equalization')


else:
    print("       Invalid Choice  !!      ")