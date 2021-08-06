from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import skimage.io
import skimage.external.tifffile
import skimage.measure
import skimage.morphology
import imageio
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from torch.autograd import Variable
from PIL import Image
from scipy import ndimage
import sys
import numpy
import subprocess
numpy.set_printoptions(threshold=sys.maxsize)


#############-------- Fonctions Utiles ----------##############


#fonction qui va me permettre de crop mes images (centrées sur leur barycentre)
#paramètres : l'image à crop, barycentre x, barycentre y, la taille souhaité (size*size)
#retourne : l'image cropé
def crop_from_center(img,cx,cy,size):
    return img[cy-size//2:cy+size//2, cx-size//2:cx+size//2]
    
    
#Fonction qui renvoie une liste des barycentres de toutes les composantes connexes d'une image
#paramètre : l'image, les composantes connexes labelisées, Liste énumérant les lables [1,2...,33]
#retourne : la liste des barycentres de composantes connexes d'une image
def barycentres(img,labels,liste_num_labels):
    return ndimage.measurements.center_of_mass(img,labels,L_num_labels)
    
    
#Fonction qui renvoie un booleen : True si le pixel est assez proche de l'image
#labélisée (défini par tresh) pour pouvoir être accepté, False sinon
def neighbors_labeled(img,x,y,tresh,num_label,size):
    for i in range (2*tresh):
        for j in range (2*tresh):
            if 1<(x-tresh+i)<size-1 and 1<(y-tresh+j)<size-1 : 
                if img[x-tresh+i][y-tresh+j] == num_label:
                    return True
    return False
            
    
    
 
############ Importation de l'image tiff à analyser ###########
print("Importing Datas...")

file_root  = "../comp_connexes/data/i3_endoplasmic_reticulum.tif"
data = skimage.external.tifffile.imread(file_root)

file2 = "../comp_connexes/data/i3.tif"
img2 = skimage.external.tifffile.imread(file2)



#note : data.shape = (79, 1360, 2120)
#note : data[0].shape = (1360, 2120)


compteur = 0
print("Starting Labeling and Croping...")
for l in range (data.shape[0]) :
    working_img = data[l]
    labels,num_labels = skimage.measure.label(working_img,return_num=True)

	#On créé une liste qui va nous servir à avoir le barycentre de chaque labels
    L_num_labels = []
    for i in range (num_labels) :
        L_num_labels.append(i+1)
		

		
		
    center_of_mass = barycentres(working_img,labels,L_num_labels)


    for k in range(num_labels):
        compteur+=1
        barycentre_i = center_of_mass[k]
        yb,xb = int(barycentre_i[0]), int(barycentre_i[1]) #on récupère les coordonneés du barycentre
	   
        size_crop = 250
	   
        tmp_labels = np.copy(labels)
        labels_croped = crop_from_center(tmp_labels,xb,yb,size_crop)
        

	
	
        tmp_img = np.copy(img2[l])
        img_croped = crop_from_center(tmp_img,xb,yb,size_crop)
        
        
        img_croped_morpho = np.copy(img_croped)   

        #NOTE : Les plt.imshow() sont très interessant pour voir les étapes de la dilatation !!!

        #Boucle pour effacer les autres composantes qui ne nous interessent pas sur l'image croppé
        for i in range(img_croped.shape[0]):
            for j in range(img_croped.shape[1]):
                if labels_croped[i][j] != k+1 : #si le pixel n'est pas dans le masque binaire, ciao
                    #dire que si la distance entre ce pixel et un du label est sup à tresh, alors on le met en noir   
                    labels_croped[i][j] = 0
		
        if compteur%100 == 0 :
            print("image numéro", compteur)
        
        #img_croped_dilate = skimage.morphology.dilation(img_croped, skimage.morphology.square(5))
        
        #plt.imshow(labels_croped)
        #plt.show()
        
        label_dilate = skimage.morphology.dilation(labels_croped, skimage.morphology.disk(10))
        
        #plt.imshow(label_dilate)
        #plt.show()
        
        for i in range(img_croped.shape[0]):
            for j in range(img_croped.shape[1]):
                if label_dilate[i][j] != k+1:
                    img_croped_morpho[i][j] = 0
        
        plt.gray()
        #plt.imshow(img_croped_morpho)
        #plt.show()
        
        try:
            datasave = "dilat_disk_10_test/"
            plt.imsave(datasave+str(compteur)+".png", img_croped_morpho)
        except RuntimeError : 
        	print("N'a pas foncitonné pour l'image numéro", compteur, "Cela est du au size_crop que vous pouvez réduire")
        	cmd = "rm "+datasave+str(compteur)+".png"
        	process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        	output, error = process.communicate()
        	print("L'image a été automatiquement supprimée")
        	


