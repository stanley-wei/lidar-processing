import numpy as np
import sys

def readInFile():
    file_name = getFile();
    arr = fileToNumpy(file_name);
    return arr;

def getFile():
    if len(sys.argv) > 1:
        file_name = sys.argv[1];
    else:
        file_name = input("File name: ");

    return file_name;

def fileToNumpy(file_name, delimiter = " "):
    file = open(file_name, 'r');
    count = sum(1 for _ in file);
    file.close();

    arr = np.loadtxt(file_name, delimiter = delimiter, dtype = float);
    arr.reshape(count, -1);
    arr = np.delete(arr, list(range(3, arr.shape[1])), axis = 1);

    return arr;

def writeToFile(point_array, file_name = "output.txt", resolution = 1.0, edit_type = 'w'):
    if(len(file_name.split('/')) > 0):
        file_name = file_name.split('/')[len(file_name.split('/')) - 1]

    if(len(point_array.shape) == 2 and point_array.shape[1] == 3):
        writeCloudToFile(point_array, file_name = file_name, edit_type = edit_type)
    elif(len(point_array.shape) == 2):
        writeGridToFile(point_array, file_name = file_name, resolution = resolution, edit_type = edit_type);

def writeGridToFile(point_array, file_name, resolution = 1.0, edit_type = 'w'):
    file = open("Output/" + file_name, edit_type)

    for i in range(point_array.shape[0]):
        for j in range(point_array.shape[1]):
            if point_array[i][j] != 0:
                file.write("%f %f %f \n" % (j * resolution, i * resolution, point_array[i][j]))

    file.close();

    return file_name;

def writeCloudToFile(point_array, file_name, edit_type = 'w'):
    file = open("Output/" + file_name, edit_type);

    for i in range(point_array.shape[0]):
        file.write("%f %f %f \n" % (point_array[i][0], point_array[i][1], point_array[i][2]))
    
    file.close();

    return file_name;
