# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 08:18:25 2020

@author: Shaku
"""

"""
Pour se test on part d'un jeu de donné sur des pétales de fleurs
suivant la longeur et largeur ils ont une couleur différente
le ? est la couleur a déterminé

couleur | rose | violet | rose | violet | rose | violet | rose | violet |  ?   
_-----------------------------------------------------------------------------
longueur|  3   |    2   |  4   |  3     |  3.5 |    2   |  5.5 |    1   |  4.5 
_------------------------------------------------------------------------------
largeur |  1.5 |    1   |  1.5 |  1     |  0.5 |   0.5  |  1   |    1   |  1   
"""

import numpy as np

# tableau d'entrée
x_input = np.array(([3,1.5],
                    [2,1],
                    [4,1.5],
                    [3,1],
                    [3.5,0.5],
                    [2,0.5],
                    [5.5,1],
                    [1,1],
                    [4.5,1]),dtype=float)

#tableau de sortie;  correspondance 1 = Rose / 0 =Violet
y_output = np.array(([1],
                    [0],
                    [1],
                    [0],
                    [1],
                    [0],
                    [1],
                    [0]),dtype=float)


#transformation du tableau d'entrée en pourcentage pour avoir des valeur entre 0 et 1
# on divise chaque nombre par la valeur max de la ligne
#pour longeur la valeur max c est 5.5
#pour largeur la valeur max c est 1.5

#np.amax(NomDuTableau, AXIS) recupere la valeur max de chaque ligne du tableau
x_enter = x_input/np.amax(x_input,axis=0)

print(x_enter)

#recuperation des données Test du tableau dont on connait toute les valeurs
#(couleur longeur largeur) soit les 8 premiere données
enter =np.split(x_enter,[8])[0]

print(enter)

#recuperation de la valeur a determiné 
XaDeterminer = np.split(x_enter,[8])[1]


