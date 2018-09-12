#!/usr/bin/python3
# -*- coding:utf-8 -*-

import sys
import numpy as np
import operator as opt

class Kdtree(object):
    '''
    class: Kdtree
    used to save kd-tree nodes which generated through train data

    member:
    __value: train data node value
     __type: train data type
      __dim: split plane dimension
       left: left child
      right: right child
    '''
    def __init__(self, node = None, node_type = -1, dim = 0, left = None, right = None):
        self.__value = node
        self.__type  = node_type
        self.__dim   = dim
        self.left    = left
        self.right   = right

    @property
    def type(self):
        return self.__type
    
    @property
    def value(self):
        return self.__value

    @property
    def dim(self):
        return self.__dim

    def distance(self, node):
        '''
        calculate distance between self-node and node
        param: node: a test or train node value
        '''
        if node == None:
            return sys.maxsize

        dis = 0
        for i in range(len(self.__value)):
            dis = dis + (self.__value[i] - node.__value[i]) ** 2
        return dis

    def build_tree(self, nodes, dim = 0):
        '''
        build a kd-tree use train data set
        param: nodes: train data set
                 dim: split plane dimension
        return: a kd-tree
        '''
        if len(nodes) == 0:
            return None
        elif len(nodes) == 1:
            self.__dim  = dim
            self.__value = nodes[0][:-1]
            self.__type  = nodes[0][-1]
            return self
    
        #sort nodes
        sortNodes = sorted(nodes, key = lambda x:x[dim], reverse = False)
    
        #get node
        midNode      = sortNodes[len(sortNodes) // 2]
        self.__value = midNode[:-1]
        self.__type  = midNode[-1]
        self.__dim   = dim
    
        leftNodes  = list(filter(lambda x: x[dim] < midNode[dim], sortNodes[:len(sortNodes) // 2]))
        rightNodes = list(filter(lambda x: x[dim] >= midNode[dim], sortNodes[len(sortNodes) // 2 + 1:]))
        nextDim    = (dim + 1) % (len(midNode) - 1)
    
        self.left  = Kdtree().build_tree(leftNodes, nextDim)
        self.right = Kdtree().build_tree(rightNodes, nextDim)
    
        return self

    def find_type(self, fnode):
        '''
        find the fnode's type 
        param: fnode: which node needed to find type
        return: fnode's nearest node and type
        '''
        if fnode == None:
            return self, -1

        fNode = Kdtree(fnode)

        #first get the leaf node
        path = []
        currentNode = self
        while currentNode != None:
            path.append(currentNode)
    
            dim   = currentNode.__dim
            if fNode.value[dim] < currentNode.value[dim]:
                currentNode = currentNode.left
            else:
                currentNode = currentNode.right
    
        #the last node in path is the leaf node
        nearestNode = path[-1]
        nearestDist = fNode.distance(nearestNode)
        path = path[:-1]
    
        while path != None and len(path) > 0:
            currentNode = path[-1]
            path = path[:-1]
            dim  = currentNode.__dim
    
            if fNode.distance(currentNode) < nearestDist:
                nearestNode = currentNode
                nearestDist = fNode.distance(currentNode)
 
            #find another child node
            brotherNode = currentNode.left
            if fNode.value[dim] < currentNode.value[dim]:
                brotherNode = currentNode.right

            if brotherNode == None:
                continue

            bdim = brotherNode.__dim
            if np.abs(fnode[bdim] - brotherNode.__value[bdim]) < nearestDist:
                cNode, _ = brotherNode.find_type(fnode)
                if fNode.distance(cNode) < nearestDist:
                    nearestDist = fNode.distance(cNode)
                    nearestNode = cNode

        return nearestNode, nearestNode.type

if __name__ == "__main__":

   #train data set
   trainArray = [[1.0, 1.0, 'a'], [1.1, 1.1, 'a'], [1.5, 1.5, 'a'], \
           [5.0, 5.0, 'b'], [5.2, 5.2, 'b'], [5.5, 5.5, 'b']]

   kdtree = Kdtree().build_tree(trainArray)

   #test1
   testNode = [1.6, 1.5]
   _, testType = kdtree.find_type(testNode)
   print("the type of ", testNode, "is ", testType)

   #test2
   testNode = [3.5, 2.7]
   _, testType = kdtree.find_type(testNode)
   print("the type of ", testNode, "is ", testType)

   #test3
   testNode = [4.3, 5.1]
   _, testType = kdtree.find_type(testNode)
   print("the type of ", testNode, "is ", testType)

