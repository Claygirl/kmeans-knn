import csv


def read_data(filename, skip_class=False, skip_header=True):

    with open(filename, 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        if(skip_header):
            next(datareader)
        traindata = []

        for row in datareader:
            if(skip_class):
                traindata.append(row[:-1])
            else:
                traindata.append(row)

    return traindata
