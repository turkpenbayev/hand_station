import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
mypath = 'data/'
images = [f for f in listdir(mypath) if isfile(join(mypath, f))]

x_data = []
y_data = []

for image in images:
    img = cv2.imread("data/{}".format(image), cv2.COLOR_BGR2GRAY)
    x_data.append(img)
    y_data.append(image.split('-')[0])

print(images[:10])
print(y_data[:10])

x_ = np.asarray(x_data)
y_ = np.asarray(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size = 0.15, random_state = 5)

print(len(x_train), len(y_train))
print(len(x_test), len(y_test))

# print("Accuracy {:.2%}".format(acc))
np.save('X_train', x_train)
np.save('y_train', y_train)
np.save('X_test', x_test)
np.save('y_test', y_test)

    