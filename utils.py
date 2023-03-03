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

def writeToFile(point_array, file_name = "output.txt", resolution = 1.0, edit_type = 'w'):
    file = open("Output/" + file_name, edit_type);

    for i in range(point_array.shape[0]):
        for j in range(point_array.shape[1]):
            if point_array[i][j] != 0:
                file.write("%f, %f, %f \n" % (j * resolution, i * resolution, point_array[i][j]))
    file.close();

    return file_name;

def fillHoles(point_array, numIterations = 1):
    for i in range(1, point_array.shape[0] - 1):
        for j in range(1, point_array.shape[1] - 1):
            counter = 0
            numNums = 0
            if(point_array[i][j] == 0):
                if(point_array[i+1][j] == 0):
                    counter += point_array[i+1][j]
                    if(point_array[i+1][j] > 0):
                        numNums += 1
                else:
                    continue;

                if(point_array[i-1][j] == 0 or counter == 0 or abs(point_array[i-1][j] - counter/numNums) <= 10):
                    counter += point_array[i-1][j]
                    if(point_array[i-1][j] > 0):
                        numNums += 1
                else:
                    continue;

                if(point_array[i][j-1] == 0 or counter == 0 or abs(point_array[i][j-1] - counter/numNums) <= 10):
                    counter += point_array[i][j-1]
                    if(point_array[i][j-1] > 0):
                        numNums += 1
                else:
                    continue;

                if(point_array[i][j+1] == 0 or counter == 0 or abs(point_array[i][j+1] - counter/numNums) <= 10):
                    counter += point_array[i][j+1]
                    if(point_array[i][j+1] > 0):
                        numNums += 1
                else:
                    continue;

                if numNums > 0:
                    point_array[i][j] = counter / numNums;

    if(numIterations > 1):
        point_array = fillHoles(point_array, numIterations - 1);

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

    writeToFile(point_array, "output3.txt", resolution, 'a')
