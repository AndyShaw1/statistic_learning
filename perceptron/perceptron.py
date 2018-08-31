#!/usr/bin/python3
# -*- coding:utf-8 -*-

import copy
import numpy as np

# learning step 
ETA = 1

weight  = [1, 2]
bias    = -1000

training_data = []
testing_data  = []

def general_data(count):
    '''
    use for randomly general training and testing data
    param: count: total data count
    return: nothing
    '''
    # the real weight and bias value
    weight_correct = [3, 5]
    bias_correct = -1000

    # randomly general 200 nodes,
    tmparray = np.random.randint(low = 0, high = 300, size = (count, 2))
    
    # set y
    total_data = []
    for x in tmparray:
        y = x[0] * weight_correct[0] + x[1] * weight_correct[1] + bias_correct
        if y > 0:
            total_data.append([[x[0], x[1]], 1])
        else:
            total_data.append([[x[0], x[1]], -1])

    # choose the front 80% data as training data, and the least data as testing data
    global training_data, testing_data
    cnt = int(len(total_data) - (len(total_data) * 0.2))
    training_data = copy.deepcopy(total_data[0:cnt])
    testing_data  = copy.deepcopy(total_data[cnt:len(total_data)])

def update(item):
    '''
    update weight and bias
    param: item: a misclassification point
    return: nothing
    '''
    global weight, bias, history
    weight[0] = weight[0] + ETA * item[1] * item[0][0]
    weight[1] = weight[1] + ETA * item[1] * item[0][1]
    bias = bias + ETA * item[1]

def cal(item):
    '''
    calculate loss function
    param: item: an instance point
    return: result: loss
    '''
    result = 0
    for i in range(len(item[0])):
        result += weight[i] * item[0][i]
    result += bias
    result *= item[1]

    return result

def check(data):
    '''
    check data set has any point is a misclassification point
    and use this point update weight and bias
    param: data: training data set
    return: flag: if training data set has any misclassification point, return False,
                  else return True
    '''
    flag = True
    for item in data:
        if cal(item) < 0:
            update(item)
            flag = False

    return flag

def train(data):
    '''
    training model, use training data set
    param: data: training data set
    return: noting
    '''
    # rapeat 1000 times at most
    for i in range(1000):
        if check(data):
            break
        else:
            #print("Training times:", i)
            #print(weight, bias)
            pass

def test(data):
    '''
    testing model, use testing data set
    param: data: testing data
    return: accuracy
    '''
    correct = 0
    for item in data:
        if cal(item) > 0:
            correct += 1

    return correct / len(data)

if __name__ == "__main__":

    general_data(1000)

    train(training_data)
    accuracy = test(testing_data)

    print("The accuracy is ", accuracy)
