from scipy import interpolate
import numpy as np

def roundMultiple(to_round, round_resolution):
    if to_round % round_resolution > round_resolution / 2:
        return (int(to_round / round_resolution) + 1) 
    else:
        return int(to_round / round_resolution) 

def findCoordsMinMax(point_array):
    min_x = round(point_array[0][0], 2)
    max_x = round(point_array[0][0], 2)

    min_y = round(point_array[0][1], 2)
    max_y = round(point_array[0][1], 2)

    min_z = round(point_array[0][2], 2)
    max_z = round(point_array[0][2], 2)

    for i in range(1, point_array.shape[0]):
        point_array[i][0] = round(point_array[i][0], 2)
        point_array[i][1] = round(point_array[i][1], 2)
        point_array[i][2] = round(point_array[i][2], 2)

        if point_array[i][0] > max_x:
            max_x = point_array[i][0]
        elif point_array[i][0] < min_x:
            min_x = point_array[i][0]

        if point_array[i][1] > max_y:
            max_y = point_array[i][1]
        elif point_array[i][1] < min_y:
            min_y = point_array[i][1]

        if point_array[i][2] < min_z:
            min_z = point_array[i][2]
        elif point_array[i][2] > max_z:
            max_z = point_array[i][2]

    return max_x, min_x, max_y, min_y, max_z, min_z;

def pointCloudToGrid(pointCloud, resolution, resolution_z, mins = []):
    if(len(mins) == 0):
        max_x, min_x, max_y, min_y, max_z, min_z = findCoordsMinMax(pointCloud)
    else:
        min_x = mins[0]
        min_y = mins[1]

    arr = np.zeros([int((max_y - min_y) / resolution) + 2, int((max_x - min_x) / resolution) + 2])

    for i in range(pointCloud.shape[0]):
        arr[roundMultiple(pointCloud[i][1] - min_y, resolution)][roundMultiple(pointCloud[i][0] - min_x, resolution)] = roundMultiple(pointCloud[i][2], resolution_z) * resolution_z;

    return arr;

def interpolateHoles(point_array):
    nonzeros = np.nonzero(point_array)
    zeros = np.argwhere(point_array == 0);
    fills = interpolate.griddata(nonzeros, point_array[nonzeros], zeros);

    for i in range(fills.shape[0]):
        point_array[zeros[i][0]][zeros[i][1]] = fills[i];

    return point_array

def createWalls(point_array, step = 1, resolution = 1):
    file = open("Output/output3.txt", 'w')

    for i in range(1, point_array.shape[0] - 1):
        for j in range(1, point_array.shape[1] - 1):
            maxZ = point_array[i][j];

            i_index = i;
            j_index = j;

            if(maxZ == 0):
                continue;

            if(point_array[i][j-1] > maxZ):
                maxZ = point_array[i][j-1]
                i_index = i;
                j_index = j-1;

            if(point_array[i][j+1] > maxZ):
                maxZ = point_array[i][j+1]
                i_index = i;
                j_index = j+1;

            if(point_array[i-1][j] > maxZ):
                maxZ = point_array[i-1][j]
                i_index = i-1;
                j_index = j;

            if(point_array[i+1][j] > maxZ):
                maxZ = point_array[i+1][j]
                i_index = i+1;
                j_index = j;

            if(maxZ - point_array[i][j] > max(step, resolution)):
                for k in range(int(point_array[i][j]) + 1, int(maxZ), step):
                    file.write("%f, %f, %f \n" % (j_index * resolution, i_index * resolution, k))

    file.close()

    return point_array
