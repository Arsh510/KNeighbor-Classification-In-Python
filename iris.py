from cProfile import label
from pyexpat import features
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

# print(iris.DESCR)
# {
#     - sepal length in cm
#     - sepal width in cm
#     - petal length in cm
#     - petal width in cm
#     - class:
#           - Iris-Setosa[0]
#           - Iris-Versicolour[1]
#           - Iris-Virginica[2]
# }

features = iris.data
label = iris.target

clf = KNeighborsClassifier()
clf.fit(features,label)

predict = clf.predict([[9.1,9.5,6.4,0.2]])
print(predict)
