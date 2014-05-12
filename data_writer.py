# -*- coding: utf-8 -*-


def write_data(filename, algorithm, k, data, classes):

    if algorithm == "K-means":
        attr = 'w'
    else:
        attr = 'a'

    with open(filename, attr) as f:
        f.write("Output of " + algorithm + ", with k=" + str(k) + "\n\n")
        for x in range(data.shape[0]):
            f.write(str(data[x]) + " class: " + str(classes[x]) + "\n")

        f.write("\n\n")
