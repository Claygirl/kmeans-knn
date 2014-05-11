import data_loader
import numpy as np

data = data_loader.read_data('iris.data', skip_class=True, skip_header=False)


def k_means(data, k):
    #TODO sprawdzic k
    data = np.array(data)
    data = data.astype(np.float)
    means = np.zeros((k, data.shape[1]))
    prev = np.zeros((k, data.shape[1]))

    indices = np.array(range(data.shape[0]))
    np.random.shuffle(indices)

    for x in range(means.shape[0]):
        means[x, :] = data[indices[x], :]

    counter = 0

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

        counter += 1

    print classes

    return classes


def k_nn(data, classes, k, element):
    #TODO sprawdzic k


new_element = np.array([5.1, 4.2, 6.1, 5.9])
classes = k_means(data, 3)
k_nn(data, classes, 3, new_element)
