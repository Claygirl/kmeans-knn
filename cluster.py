# -*- coding: utf-8 -*-

import data_loader
import data_writer
import numpy as np

data = data_loader.read_data('iris.data', skip_class=True, skip_header=False)
data = np.array(data)
data = data.astype(np.float)


def k_means(data, k):

    means = np.zeros((k, data.shape[1]))
    prev = np.zeros((k, data.shape[1]))

    indices = np.array(range(data.shape[0]))
    np.random.shuffle(indices)

    for x in range(means.shape[0]):
        means[x, :] = data[indices[x], :]

    while(not np.array_equal(means, prev)):

        prev = means

        classes = np.zeros((data.shape[0], 1))
        distances = np.zeros((data.shape[0], k))

        for x in range(data.shape[0]):
            for y in range(means.shape[0]):
                diff = data[x] - means[y]
                dist = sum(diff**2)
                dist = np.sqrt(dist)
                distances[x, y] = dist
            classes[x] = np.argmin(distances[x])

        classes = classes.astype(np.integer)
        means = np.zeros((k, data.shape[1]))

        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                means[classes[x], y] += data[x, y]

        for x in range(means.shape[0]):
            means[x] = means[x] / (classes == x).sum()

    return classes


def k_nn(data, data_classes, k, element):
    #TODO sprawdzic k

    distances = np.zeros((data.shape[0], 1))

    for x in range(data.shape[0]):
        diff = data[x] - element
        dist = sum(diff**2)
        dist = np.sqrt(dist)
        distances[x] = dist

    results = np.hstack((distances, data_classes))

    results_indices = np.lexsort((results[:, 1], results[:, 0]))

    print results[results_indices]

    n = np.unique(data_classes).size
    classes = np.zeros((n, 1), dtype=np.integer)

    for x in range(k):
        classes[results[x, 1].astype(np.integer)] += 1

    return np.argmax(classes)


k_m = 0
k_n = 0

while(k_m < 2 or k_m >= data.shape[0] / 3):
    k_m = raw_input("Input number of k-means:")
    k_m = int(k_m)
    if k_m < 2:
        print "K too small. Try k > 1"
    elif k_m >= data.shape[0] / 3:
        print "K too large. Try k < " + str(data.shape[0] / 3)

new_classes = k_means(data, k_m)
data_writer.write_data('out.data', "K-means", k_m, data, new_classes)
print("Output of k-means written to out.data file")

while(k_n >= data.shape[0]):
    k_n = raw_input("Input number of k-NN:")
    k_n = int(k_n)
    if k_n >= data.shape[0]:
        print "K too large. Try k < " + str(data.shape[0])

new_element = []
coo = 0

print("Add coordinates of new element")

for x in range(data.shape[1]):
    while(not isinstance(coo, float)):
        coo = raw_input("Enter float for " + str(x) + ". coordinate:")
        
    new_element.append(coo)

new_element = np.array(new_element)
new_class = k_nn(data, new_classes, k_n, new_element)

data_writer.write_data('out.data', "K-NN", k_n, new_element, new_class)

