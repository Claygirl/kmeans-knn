import data_loader
import numpy as np

data = data_loader.read_data('iris.data', skip_class=True, skip_header=False)
data = np.array(data)
data = data.astype(np.float)

def k_means(data, k):
    
    if k < 2:
        print "K too small. Try k > 1."
        return
    elif k > data.shape[0] / 3:
        print "K too large. Try k < " + str(data.shape[0] / 3)
    
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
    results = np.partition(results, k, axis=0)
    
    n = np.unique(data_classes).size    
    classes = np.zeros((n, 1), dtype=np.integer)
    
    for x in range(k):
        classes[results[x, 1].astype(np.integer)] += 1
        
    return np.argmax(classes)
        

new_element = np.array([5.1, 4.2, 6.1, 5.9])
classes = k_means(data, 3)
k_nn(data, classes, 3, new_element)
