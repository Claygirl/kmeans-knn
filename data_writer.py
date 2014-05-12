# -*- coding: utf-8 -*-


def write_data(filename, algorithm, k, data, classes):

    if algorithm == "K-means":
        attr = 'w'
    else:
        attr = 'a'

    with open(filename, attr) as f:
        f.write("Output of " + algorithm + ", with k=" + str(k) + "\n\n")
        for x in range(data.shape[0]):
            f.write(str(data[x]) + " calculated class: " + str(classes[x]) + "\n")

        f.write("\n\n")


def write_tests(filename, algorithm, k, data, classes, org_classes):

    with open(filename, 'w') as f:
        f.write("Output of " + algorithm + ", with k=" + str(k) + "\n\n")
        for x in range(data.shape[0]):
            f.write(str(data[x]) + " calculated class: " + str(classes[x]))
            f.write(" original class: " + str(org_classes[x]) + "\n")

        f.write("\n\n")
