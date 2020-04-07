# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:23:13 2020

@author: Shaku
"""

import numpy as np
import os.path

class Neuronal_Network(object):
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize=3
        
        if(os.path.isfile("poid.npz")):
            print("Loading file")
            container = np.load("poid.npz")
            self.w1 = container['w1']
            self.w2 = container['w2']
        else:
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
    
    def deriveSigmoid(self,x):
        #deriver de la fonction sigmoide
        deriv = x*(1-x)
        return deriv
    
    # fonction de retro propagation 3 parametre
    # X_entrer : valeur d entré du reseau de neuronne
    # Y_sortie_prevu : valeur de sortie qu'on devrait normalement avoir
    # O_sortie_predic : valeur de la sortie calculer
    def backward(self,X_entrer , Y_sortie_prevu ,O_sorti_predict):
                
        #calcule de l erreur soustraction de la valeur de sortie 
        self.error_o = (Y_sortie_prevu)-(O_sorti_predict)
        
        #calcul de l erreur Delta
        #l erreur multiplié par la derivé sigmoide de la valeur de sortie calculé
        #la derivé sygmoide permet de recuperer la valeur d entré du neuronne
        # avant d appliquer la sygmoide pour affecter la valeur au neuronne
        self.error_output = self.error_o * self.deriveSigmoid(O_sorti_predict)
        
        #calcule de l erreur des neuronne caché par produit matriciel
        self.errorW2 = self.error_output.dot(self.w2.T)
        
        #calcule de l erreur delta des neuronne caché
        # z2 represente la valeur des neuronnes caché calculer dans la fonction forward
        self.error_delta = self.errorW2 * self.deriveSigmoid(self.z2)
                
        #mise a jour des poids W1 en fonction des entrées :
        # produit matriciel des entrées par l erreur delta des neuronnes caché
        self.w1 += X_entrer.T.dot(self.error_delta)
        
        
        #mise a jour des poids W2 en fonction des valeur des neuronnes caché
        # multiplication des valeurs des neuronnes par l erreur delta de sorti        
        self.w2 += self.z2.T.dot(self.error_output)
        
    #fonctino d entrainement 
    def train(self,X_entrer,Y_output):
        for i in range(100000):
            output = self.forward(X_entrer)
            self.backward(X_entrer,Y_output,output)
        np.savez('poid.npz',w1=self.w1,w2=self.w2)
        
      
    def prediction(self, x_entrer):
        print("donnée prédite après entrainement")
        print("valeur d'entré : " +str(x_entrer))
        print("valeur de sortie : "+str(self.forward(x_entrer)))
        
        if(self.forward(x_entrer)<0.5):
            print("la fleur est violette")
        else:
            print("la fleur est rose")
            
        


        