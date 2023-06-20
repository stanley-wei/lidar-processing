import numpy as np

def read_in_array(message = ""):
    file_name = get_file(message);
    arr = file_to_array(file_name);
    return arr;

def get_file(message = ""):
    if(message == ""):
        file_name = input("File name: ");
    else:
        file_name = input(message + ": ")

    return file_name;

def file_to_array(file_name, delimiter = " "):
    file = open(file_name, 'r');
    count = sum(1 for _ in file);
    file.close();

    arr = np.loadtxt(file_name, delimiter = delimiter, dtype = float);
    arr.reshape(count, -1);
    arr = np.delete(arr, list(range(3, arr.shape[1])), axis = 1);

    return arr;

def write_to_file(point_array, file_name = "output.txt", resolution = 1.0, edit_type = 'w'):
    if(len(file_name.split('/')) > 0):
        file_name = file_name.split('/')[len(file_name.split('/')) - 1]

    if(len(point_array.shape) == 2 and point_array.shape[1] == 3):
        write_pcloud_to_file(point_array, file_name = file_name, edit_type_ = edit_type)
    elif(len(point_array.shape) == 2):
        write_grid_to_file(point_array, file_name = file_name, resolution = resolution, edit_type_ = edit_type);

def write_grid_to_file(point_array, file_name, resolution = 1.0, edit_type_ = 'w'):
    file = open(file_name, edit_type_)

    for i in range(point_array.shape[0]):
        for j in range(point_array.shape[1]):
            if point_array[i][j] != 0:
                file.write("%f %f %f \n" % (j * resolution, i * resolution, point_array[i][j]))

    file.close();

    return file_name;

def write_pcloud_to_file(point_array, file_name, edit_type_ = 'w'):
    file = open(file_name, edit_type_);

    for i in range(point_array.shape[0]):
        file.write("%f %f %f \n" % (point_array[i][0], point_array[i][1], point_array[i][2]))
    
    file.close();

    return file_name;
