# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:23:13 2020

@author: Shaku
"""

import numpy as np

class Neuronal_Network(object):
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize=3
        
        self.w1 = np.random.randn(self.inputSize,self.hiddenSize) #Matrice 2x3
        self.w2 = np.random.randn(self.hiddenSize,self.outputSize) #Matrice 3x1
        
        def forward(self,x):
            # produit matriciel entre la valeur des neuronnes d entré et les poids
            # w1 entre les neuronnes d entré et les neuronne caché
            # pour obtenir la valeur d entré des neuronnes caché
            self.z = np.dot(x,self.w1)
            
            # sigmoid des neuronnes caché en fonction des valeur d entré des
            # neuronne caché
            self.z2 = self.sigmoid(self.z)
            
            # produit matriciel entre la valeur des neuronnes caché et les poids
            # w2 entre les neuronnes caché et le neuronne d output
            # pour obtenir la valeur d entré du neuronne output
            self.z3 = np.dot(self.z2,self.w2)
            
            # sigmoid du neuronne d output en fonction des valeur d entré du 
            # neuronne de sorti output
            output = self.sigmoid(self.z3)
            return output
            
        def sigmoid(self,x):
            sig = 1/(1+np.exp(-x))
            return sig