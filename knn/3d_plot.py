from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import csv


def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

if __name__ == '__main__':
    #Reading in the file via csv library
    filepath = 'C:\\Users\\Joash\Desktop\\University Stuff\\4B uni stuff\\SYDE 522\\522 Project\\SMS_spam_or_ham\\' \
               'neural network\Archive\\Hidden Combination NN_v2'
    csvfile = open(filepath + '.csv', "rt", encoding="utf8")
    reader = csv.reader(csvfile)
    x = []
    y = []
    z = []
    First = True
    for row in reader:
        if First:
            First = False
            continue
        x.append(float(row[0]))
        y.append(float(row[1]))
        z.append(float(row[2]))

    # x = np.array(x[1:])
    # y = np.array(y[1:])
    # z = np.array(z[1:])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    # print(x)
    # print(y)
    # print(z)


    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z)
    ax.set_xlabel('# Neurons in 1st Hidden Layer')
    ax.set_ylabel('# Neurons in 2nd Hidden Layer')
    ax.set_zlabel('Validation Accuracy')
    ax.set_title('Hidden Layer Combinations vs Accuracy')
    plt.show()




    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # # ax.contour3D(x, y, z, 50, cmap='binary')
    # ax.scatter3D(x, y, z, c=z, cmap='Greens')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.show()

    X, Y = np.meshgrid(x, y)
    print(x)
    # x = np.linspace(-6, 6, 30)
    # y = np.linspace(-6, 6, 30)
    #
    # print(x)
    # X, Y = np.meshgrid(x, y)
    # Z = f(X, Y)
    #
    # print(Z)
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.contour3D(X, Y, Z, 50, cmap='binary')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.show()