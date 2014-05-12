# -*- coding: utf-8 -*-

import data_loader
import data_writer
import numpy as np


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

    return (means, classes)


def k_nn(data, data_classes, k, element):

    distances = np.zeros((data.shape[0], 1))

    for x in range(data.shape[0]):
        diff = data[x] - element
        dist = sum(diff**2)
        dist = np.sqrt(dist)
        distances[x] = dist

    results = np.hstack((distances, data_classes))

    results_indices = np.lexsort((results[:, 1], results[:, 0]))

    results[results_indices]

    n = np.unique(data_classes).size
    classes = np.zeros((n, 1), dtype=np.integer)

    for x in range(k):
        classes[results[x, 1].astype(np.integer)] += 1

    return np.argmax(classes)


def k_means_test(data, means, classes, org_classes):

    org_means = np.zeros((3, data.shape[1]))
    ans = np.zeros((data.shape[0], 1))
    ans = ans.astype(np.integer)

    for x in range(org_classes.shape[0]):
        if org_classes[x] == "Iris-setosa":
            ans[x] = 0
        elif org_classes[x] == "Iris-versicolor":
            ans[x] = 1
        elif org_classes[x] == "Iris-virginica":
            ans[x] = 2

    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            org_means[ans[x], y] += data[x, y]

    for x in range(org_means.shape[0]):
        org_means[x] = org_means[x] / (ans == x).sum()

    distances = np.zeros((org_means.shape[0], 3))
    mappings = np.zeros((org_means.shape[0], 1))

    for x in range(means.shape[0]):
        for y in range(org_means.shape[0]):
            diff = means[x] - org_means[y]
            dist = sum(diff**2)
            dist = np.sqrt(dist)
            distances[x, y] = dist
        mappings[x] = np.argmin(distances[x])

    mappings = mappings.astype(np.string0)

    for x in range(mappings.shape[0]):
        if mappings[x] == '0.0':
            mappings[x] = "Iris-setosa"
        elif mappings[x] == '1.0':
            mappings[x] = "Iris-versicolor"
        elif mappings[x] == '2.0':
            mappings[x] = "Iris-virginica"

    new_classes = np.empty((classes.shape[0], 1), dtype=np.dtype('a25'))

    for x in range(classes.shape[0]):
        new_classes[x] = mappings[classes[x]]

    matches = 0.0

    for x in range(classes.shape[0]):
        if org_classes[x] == new_classes[x]:
            matches += 1

    matches = matches / classes.shape[0]

    return (mappings, new_classes, matches)


def interact():

    skip_class = False

    data = data_loader.read_data('in.data', skip_class=skip_class,
                                 skip_header=False)
    data = np.array(data)

    if not skip_class:
        split = np.split(data, [-1], axis=1)
        data = split[0]
        org_classes = split[1]

    data = data.astype(np.float)

    k_m = 0
    k_n = 10000

    while(k_m < 1 or k_m > data.shape[0]):

        k_m = raw_input("Input number of k-means: ")
        k_m = int(k_m)
        if k_m < 1:
            print "K too small. Try k > 0"
        elif k_m > data.shape[0]:
            print "K too large. Try k <= " + str(data.shape[0])

    means, new_classes = k_means(data, k_m)

    if k_m == 3:
        mappings, coded_classes, matches = k_means_test(data, means, new_classes,
                                               org_classes)
        data_writer.write_tests('out.data', "K-means", k_m, data,
                                coded_classes, org_classes, matches)
        print("Output of k-Means with test written to out.data file")
    else:
        data_writer.write_data('out.data', "K-means", k_m, data, new_classes)
        print("Output of k-Means written to out.data file")

    while(k_n >= data.shape[0]):

        k_n = raw_input("Input number of k-NN: ")
        k_n = int(k_n)
        if k_n >= data.shape[0]:
            print "K too large. Try k < " + str(data.shape[0])

    new_element = []
    coo = 0

    print("Add coordinates of the new element")

    for x in range(data.shape[1]):

        coo = raw_input("Enter float for " + str(x + 1) + ". coordinate: ")
        if coo == "":
            coo = 0
        coo = float(coo)
        new_element.append(coo)

    new_element = np.array([new_element])
    new_class = k_nn(data, new_classes, k_n, new_element)

    if k_m == 3:
        new_class = mappings[new_class]

    new_class = np.array([[new_class]])

    data_writer.write_data('out.data', "K-NN", k_n, new_element, new_class)

    print("Output of k-NN written to out.data file")

interact()
