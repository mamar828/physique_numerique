#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TP reconstruction TDM (CT)
# Prof: Philippe Després
# programme: Dmitri Matenine (dmitri.matenine.1@ulaval.ca)


# libs
import numpy as np
import time

# local files
import geo
import util
import CTfilter

## créer l'ensemble de données d'entrée à partir des fichiers
def readInput():
    # lire les angles
    [nbprj, angles] = util.readAngles(geo.dataDir+geo.anglesFile)

    print("nbprj:",nbprj)
    print("angles min and max (rad):")
    print("["+str(np.min(angles))+", "+str(np.max(angles))+"]")

    # lire le sinogramme
    [nbprj2, nbpix2, sinogram] = util.readSinogram(geo.dataDir+geo.sinogramFile)

    if nbprj != nbprj2:
        print("angles file and sinogram file conflict, aborting!")
        exit(0)

    if geo.nbpix != nbpix2:
        print("geo description and sinogram file conflict, aborting!")
        exit(0)

    return [nbprj, angles, sinogram]


## reconstruire une image TDM en mode rétroprojection
def laminogram():
    
    [nbprj, angles, sinogram] = readInput()

    # initialiser une image reconstruite
    image = np.zeros((geo.nbvox, geo.nbvox))

    # "etaler" les projections sur l'image
    # ceci sera fait de façon "voxel-driven"
    # pour chaque voxel, trouver la contribution du signal reçu
    for j in range(geo.nbvox): # colonnes de l'image
        print("working on image column: "+str(j+1)+"/"+str(geo.nbvox))

        # Rajouté par Thomas le 4 mars 2025
        x = (j-0.5*(geo.nbvox-1))*geo.voxsize
        
        for i in range(geo.nbvox): # lignes de l'image

            # Rajouté par Thomas le 4 mars 2025
            y = (i-0.5*(geo.nbvox-1))*geo.voxsize
            r = np.sqrt(x**2+y**2)
            if x == 0:
                phi = np.pi/2
                if y < 0:
                    phi *= -1
            else:
                phi = np.arctan(y/x)
                if x < 0:
                    phi += np.pi

            for a in range(len(angles)):
                #votre code ici...
                #le défi est simplement géométrique;
                #pour chaque voxel, trouver la position par rapport au centre de la
                #grille de reconstruction et déterminer la position d'arrivée
                #sur le détecteur d'un rayon partant de ce point et atteignant
                #le détecteur avec un angle de 90 degrés. Vous pouvez utiliser
                #le pixel le plus proche ou interpoler linéairement...Rappel, le centre
                #du détecteur est toujours aligné avec le centre de la grille de
                #reconstruction peu importe l'angle.

                # Rajouté par Thomas le 4 mars 2025
                d = r*np.cos(phi-angles[a])
                nb_pixels = d//geo.pixsize
                pixel = int(geo.nbpix/2+nb_pixels)
                image[i][-j] += sinogram[a][pixel] #Pourquoi le - est nécessaire pour avoir une image droite?
                # Fin de ce que j'ai rajouté    

    util.saveImage(image, "lam")

## reconstruire une image TDM en mode retroprojection filtrée
def backproject():
    
    [nbprj, angles, sinogram] = readInput()
    
    # initialiser une image reconstruite
    image = np.zeros((geo.nbvox, geo.nbvox))
    
    ### option filtrer ###
    CTfilter.filterSinogram(sinogram)
    ######
    
    # "etaler" les projections sur l'image
    # ceci sera fait de façon "voxel-driven"
    # pour chaque voxel, trouver la contribution du signal reçu
    for j in range(geo.nbvox): # colonnes de l'image
        print("working on image column: "+str(j+1)+"/"+str(geo.nbvox))

        x = (j-0.5*(geo.nbvox-1))*geo.voxsize

        for i in range(geo.nbvox): # lignes de l'image

            y = (i-0.5*(geo.nbvox-1))*geo.voxsize
            r = np.sqrt(x**2+y**2)
            if x == 0:
                phi = np.pi/2
                if y < 0:
                    phi *= -1
            else:
                phi = np.arctan(y/x)
                if x < 0:
                    phi += np.pi

            for a in range(len(angles)):
                #votre code ici
               #pas mal la même chose que prédédemment
            #mais avec un sinogramme qui aura été préalablement filtré

                d = r*np.cos(phi-angles[a])
                nb_pixels = d//geo.pixsize
                pixel = int(geo.nbpix/2+nb_pixels)
                image[i][-j] += sinogram[a][pixel] #Pourquoi le - est nécessaire pour avoir une image droite?

    util.saveImage(image, "fbp")

## reconstruire une image TDM en mode retroprojection
def reconFourierSlice():
    
    [nbprj, angles, sinogram] = readInput()

    # initialiser une image reconstruite, complexe
    # pour qu'elle puisse contenir sa version FFT d'abord
    IMAGE = np.zeros((geo.nbvox, geo.nbvox), 'complex')
    
    # conteneur pour la FFT du sinogramme
    SINOGRAM = np.zeros(sinogram.shape, 'complex')

    #image reconstruite
    image = np.zeros((geo.nbvox, geo.nbvox))
    #votre code ici
    #ici le défi est de remplir l'IMAGE avec des TF des projections (1D)
    #au bon angle.
    #La grille de recon est cartésienne mais le remplissage est cylindrique,
    #ce qui fait qu'il y aura un bon échantillonnage de IMAGE
    #au centre et moins bon en périphérie. Un TF inverse de IMAGE vous
    #donnera l'image recherchée.

   
    
    util.saveImage(image, "fft")


## main ##
start_time = time.time()
#laminogram()
backproject()
#reconFourierSlice()
print("--- %s seconds ---" % (time.time() - start_time))
