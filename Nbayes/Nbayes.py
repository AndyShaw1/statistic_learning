#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np

type_num   = 10
fature_num = 10
LAMBDA     = 1

def Train(dataset):
    if len(dataset) <= 0:
        return

    type_count = np.zeros(type_num, dtype = int)
    type_prob  = np.zeros(type_num)

    cond_type_cnt = np.zeros([type_num, fature_num, 2], dtype = int)
    cond_prob     = np.zeros([type_num, fature_num, 2])

    for data in dataset:
        type_count[data[1]]++

        for i in range(fature_num):
            cond_type_count[data[1]][i][data[0][i]]++

    for i in range(type_num):
        type_prob[i] = (type_count[i] + LAMBDA) / (len(dataset) + type_num * LAMBDA)

        for j in range(fature_num):
            cond_prob[i][j][0] = (cond_type_count[i][j][0] + LAMBDA) / \
                    (type_count[i] + fature_num * LAMBDA)
            cond_prob[i][j][1] = (cond_type_count[i][j][2] + LAMBDA) / \
                    (type_count[i] + fature_num * LAMBDA)

    return type_prob, cond_prob

def Test(node):
    if len(node) != fature_num:
        return -1

    prob = np.zeros(type_num)
    for i in range(type_num):
        for j in range(fature_num):
            prob[i] = type_prob[i] * cond_prob[i][j][node[i]]

    return np.where(np.amax(prob))


