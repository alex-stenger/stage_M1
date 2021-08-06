# -*- coding: utf-8 -*-

from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
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
import matplotlib

matplotlib.use('Agg') #to avoid XIO error



####################################################################################################################################
#********************************** Variables Utiles, Importation et pré-traitement des données ***********************************#
####################################################################################################################################



ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


###############################################
#------ IMPORTANT ADVICE WITH DATAROOT -------#
# Be careful, the way dataloader from pytorch #
# is working could be weird at first sight.   #
# Take a look at pytorch.org for more details.#
###############################################


#dataroot = "data/dilat_sorted_disk_10/dilat/"  #reticulum contourés texturés dilatés disk(10)

dataroot_mask = "data/pandley_mask/rec"       #reticulum contouré binarisé (masque binaire)

dataroot_img = "data/pandley_img/rec"        #patch complet, associé au masque ci dessus

#dataroot = "data/croped_patch_full/text"                  #reticulum non contouré texturés

#dataroot = "data/croped_texture_contour/cont"         #Reticulum contourés texturés

result_root_stage_1_batch = "result/batch_result/batch_epoch"              #Chemin où sauvegarder des échantillons des masques binaires créés

result_root_stage_1_patch = "result/fake_mask_generated/rec/mask"          #Chemin où sauvegarder l'ensemble des masques binaires générés
                                                                                 #Les masques sauvegardés seront uniquement ceux générés à la dernière époque

result_root_stage_2_batch = "result/batch_result_stage_2/batch_epoch_"     #Chemin où sauvegarder les échantillons des images complètes générées

root_data_mask_generated = "result/fake_mask_generated/rec"                #Chemin pour charger les masques générés plus haut 
                                                                                 # ATTENTION : Pas exactement le même chemin qu'au dessus, c'est à cause du dataloader !!!


img_mult = 1 #Pour essayer de passer à une résolution supérieure plus tard

image_size = 64 * img_mult 

batch_size = 128 

workers = 2

ngpu = 1

nz = 100

# Attention, on est sur du noir et blanc pour l'instant
nc = 1  #nombre de channel (1 = grayscale / 3 = RGB)

ngf = 64 
ndf = 64 

num_epoch_stage_1 = 100
num_epoch_stage_2 = 200


############################################################
#********************* DATALOADER *************************#
############################################################

dataset_mask = dset.ImageFolder(root=dataroot_mask,
                           transform=transforms.Compose([
                               transforms.Grayscale(num_output_channels=1),  #on passe en nuances de gris !
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5), (0.5)),  #Attention, on change ici parce qu'on est passé en grayscale
                           ]))

#Dataloader_mask va contenir les masques binaires
dataloader_mask = torch.utils.data.DataLoader(dataset_mask, batch_size=batch_size, shuffle=False, num_workers=workers)  ##Shuffle=False important pour correctement pouvoir associer un masque à son image


dataset_img = dset.ImageFolder(root=dataroot_img,
                           transform=transforms.Compose([
                               transforms.Grayscale(num_output_channels=1),  #on passe en nuances de gris !
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5), (0.5)),  #Attention, on change ici parce qu'on est passé en grayscale
                           ]))

#Dataloader_img va contenir les patchs complets associés au masque binaire
dataloader_img = torch.utils.data.DataLoader(dataset_img, batch_size=batch_size, shuffle=False, num_workers=workers)


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader_mask))
print(real_batch[0][0].size())
#print(real_batch[0])
plt.figure(figsize=(14,14))
plt.axis("off")
plt.title("Training Images")
plt.gray()
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
#plt.imshow(real_batch[0][0][0])
plt.show()




##################################################################################################################################################################
#*********************** Archi 1st Stage GAN : On reprends l'archi DCGAN pour générer les masques puisque cela fonctionne plutôt bien ***************************#
##################################################################################################################################################################



# Initialisation des poids
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8 * img_mult, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8 * img_mult),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8 * img_mult, ngf * 4 * img_mult, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4 * img_mult ),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4 * img_mult, ngf * 2 * img_mult, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2 * img_mult),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2 * img_mult, ngf * img_mult, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * img_mult),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8 * img_mult, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            #state size. 1
        )

    def forward(self, input):
        return self.main(input)

G = Generator(1).to(device) #pour le faire tourner sur le GPU
D = Discriminator(1).to(device) #de même

G.apply(weights_init)
D.apply(weights_init)

print(G)
print(D)

BCE_loss = nn.BCELoss()

learning_rate = 0.0002
beta1 = 0.5

G_optimizer = optim.Adam(G.parameters(), lr=learning_rate, betas=(beta1, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=learning_rate, betas=(beta1, 0.999))

real_label = 1
fake_label = 0 

G_loss_l = []
D_loss_l = []



###############################################################################################
#**************************** Entrainement du 1st Stage GAN **********************************#
###############################################################################################



print("Starting 1st stage GAN Training Loop...")

#!cd /content
#!mkdir fake_generated

compteur = 0



noise_for_print = torch.randn(batch_size, nz, 1, 1, device=device)

for epoch in range(num_epoch_stage_1) :
  for i, data in enumerate(dataloader_mask, 0):  #enumerate retourne une liste d'énumérations, exemple :
                                            #seasons = ['Spring', 'Summer', 'Fall', 'Winter']
                                            #list(enumerate(seasons))
                                            #[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
  
  ##############################################################################################################################
  #--------------------On commence par améliorer le Discriminateur (maximize log(D(x)) + log(1 - D(G(z))))---------------------#
  ##############################################################################################################################

  #D'abbord avec les vraies données
    D.zero_grad()
    real = data[0].to(device)  #on prends la donnée et on la met sur le GPU


    b_size = real.size(0) 
    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)  #créé un tenseur de taille batch_size contenant les lables (ici real)  (b_size normalement)

    ##print(real.size())

    outputD_real = D(real).view(-1) #La on fait passé les vraies données dans le discriminateur qui va nous renvoyer une proba entre 0 et 1 (proche de 1 vrai, sinon fake)

    #print("output_d : ", outputD_real[0].size())
    #print("label : ", label.size())

    loss_D_real = BCE_loss(outputD_real, label)  #Les labels nous permettent de calculer la loss fonction de "D_real"

    loss_D_real.backward()
    D_x = outputD_real.mean().item ()

    #Ensuite avec les fausses données du générateur (pour avoir la loss fonction complète de D)

    #mini_batch = data[0].size()[0]
    #noise = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
    #noise = Variable(noise.cuda())
    
    noise_2 = torch.randn(b_size, nz, 1, 1, device=device) #on génère un vecteur de l'espace latent sur lequel le générateur va travailler
                                                        #Au final, si j'ai bien compris, il va y avoir b_size vecteur de taille z (d'où l'utilité du tenseur)
    if epoch%5==0 and i == 0:
      noise = noise_for_print  #En gros, permet de voir si en donnant le même bruit, ça va nous donner des choses comparables et similaires 
    else :
      noise = noise_2


    fake = G(noise) #On fait passer le bruit dans le générateur pour qu'il nous sorte un chiffre fake
    label.fill_(fake_label) #On remplace les real_labels par les fake_labels 

    outputD_fake = D(fake.detach()).view(-1) 
    
    loss_D_fake = BCE_loss(outputD_fake, label)
    loss_D_fake.backward()

    D_G_z1 = outputD_fake.mean().item()

    loss_D = loss_D_real + loss_D_fake  #Calcul de la loss function "globale" de D

    D_optimizer.step()  #On améliore D



    ##########################################################################################
    #-----------------Puis on améliore le Générateur (maximize log(D(G(z))))-----------------#
    ##########################################################################################

    G.zero_grad() #"In PyTorch, we need to set the gradients to zero before starting to do backpropragation 
          #because PyTorch accumulates the gradients on subsequent backward passes. Because of this, when you start 
          #your training loop, ideally you should zero out the gradients."

    label.fill_(real_label) #Du point de vue de la loss function du générateur, on a des vrais labels

    outputD2 = D(fake).view(-1)  #.view(-1) permet de renvoyer un tenseur mais "applatit", en gros un tenseur de taille n*1

    loss_G = BCE_loss(outputD2, label)  #Loss function de G

    loss_G.backward()

    D_G_z2 = outputD2.mean().item()  #.mean() nous donne la moyenne et .item() permet de convertir le tenseur en un réel standard
                                  # Juste ici, il me semble que dans output, il n'y a qu'une seule valeur, donc je ne comprends pas le .mean()

    G_optimizer.step()  #on améliore G

            #Faut vraiment voir ça en 5 étapes :
          #   1) On regarde ce que le discriminateur nous dis sur un fake du generator (la proba qu'il sort) en gros
          #   2) ON calcule la loss fonction (qu'on veut optimiser)
          #   3) Calcul du gradient de la loss function de G (mais je sais pas ce qu'on en fait ensuite)
          #   4) Ennsuite, on calcule D(G(z))
          #   5) On utilie l'optimizer pour update le générateur


    #NOTE : ce qui va nous intéresser (pour l'affichage), c'est l'output de G(noise), c'est à dire "fake"


    G_loss_l.append(loss_G.item())
    D_loss_l.append(loss_D.item())

    if i%50==0 :
      print("iteration ", i+1, "/", len(dataloader_mask),"-- epoch", epoch+1, "/", num_epoch_stage_1, "---- Loss_D : ", loss_D.item(), " Loss_G : ", loss_G.item(),
            " D(x) : ", D_x, " D(G(z1)) : ", D_G_z1, " D(G(z2)) : ", D_G_z2)
      
      #note : fake.size() = torch.Size([128,1,64,64])
      #note : data[0].size() = torch.Size([128,3,64,64])
      #note : fake[0,:,:,:].size() = torch.Size([1,64,64])
      #note : data[0][0,:,:,:].size() = torch.Size([3,64,64])

    if epoch%5==0 and i==0:

      plt.figure(figsize=(14,14))
      plt.axis("off")
      plt.title("Training Images a l'époque "+str(epoch)+" et à l'itération "+str(i))
      img_cpu = np.transpose(vutils.make_grid(fake.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0))
      img_to_save = img_cpu.detach().numpy()
      plt.imsave(result_root_stage_1_batch+str(epoch)+".png", img_to_save)
      plt.cla()
      plt.clf()
      plt.close()

      #plt.imshow(np.transpose(vutils.make_grid(fake.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
      #plt.show()


    if epoch > num_epoch_stage_1 - 2 :  #in order not to save too much images
      for i in range(fake.size()[0]) :
        compteur+=1
        to_cpu = fake.to(device='cpu')
        to_numpy = to_cpu.detach().numpy()
            #print(to_numpy.shape) #(128, 1, 64, 64)
            #print(to_numpy[i][0].shape) #(64, 64)
        plt.gray()
        plt.imsave(result_root_stage_1_patch+str(compteur)+".png", to_numpy[i][0]) #on fait attention à convertir en numpy
      



plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During 1st Stage GAN Training")
plt.plot(G_loss_l,label="G")
plt.plot(D_loss_l,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()




###########################################################################################################################
#**************************** 2nd Stage GAN : Implémentation du 2nd GAN du papier de Pandley *****************************#
###########################################################################################################################


# Pour plus de détail, se référer au papier de Pandley et l'architecture de son "2nd Stage GAN".
# A noter que je n'ai pas repris exactement son architecture, j'y ai fait quelques modifications,
# Dans les grandes lignes, ça reste similaire.


class Generator_2(nn.Module): 
    def __init__(self):

        super(Generator_2,self).__init__()
        
            
        self.seq1 = nn.Sequential(
            #conv(1,16) because we are in grayscale
            nn.Conv2d(1,16,3),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1),
            #1*16*64*64 (N,C,H,W)
            nn.Conv2d(16,32,3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1)
            )
            #1*32*32*32 (N,C,H,W)
            #Here, we have to keep the result to merge it later
            
        self.seq2 = nn.Sequential(
            nn.Conv2d(32,64,3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1)
            )
            #keep result
        
        self.seq3 = nn.Sequential(
            nn.Conv2d(64,128,3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1)
            )
            #keep result
            
        self.seq4 = nn.Sequential(
            nn.Conv2d(128,256,3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25)
            )
            #1*256*8*8 (N,C,H,W)
            
        self.variational_out = nn.Sequential(
            nn.Conv2d(256,64,3, padding=1),
            #1*64*8*8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Upsample((16,16)) #same here, i'm not sure
            )
            #1*64*16*16 (N,C,H,w)
            
        
        self.seq_noise_comp_1 = nn.Linear(100,256*8*8)
        
        self.seq_noise_comp_1_1 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
            )
            
        self.seq_noise_comp_2 = nn.Sequential(
            #I think upsample is not needed here
            nn.Conv2d(256,128,3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            #1*128*8*8
            nn.Conv2d(128,64,3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Upsample((16,16)) #I added it
            )
            #1*64*16*16
            
        self.down_1 = nn.Sequential(
            #1*128*16*16
            nn.Conv2d(128,64,3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            #I'm adding a second block
            nn.Conv2d(64,32,3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Upsample((32,32))
            #1*32*32*32
            )
            
        self.upsample_1 = nn.Sequential(
            nn.Conv2d(128,32,1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Upsample((32,32))
            #1*32*32*32
            )
            
        self.down_2 = nn.Sequential(
            nn.Conv2d(64,32,3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32,16,3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Upsample((64,64))
            #1*16,64,64
            )
            
        
        self.upsample_2 = nn.Sequential(
            nn.Conv2d(64,16,1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Upsample((64,64)),
            #1*16*64*64
            )
            
        self.down_3 = nn.Sequential(
            nn.Conv2d(32,16,3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16,8,3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Upsample((128,128))
            #1*8*128*128
            )
            
        self.upsample_3 = nn.Sequential(
            nn.Conv2d(32,8,1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Upsample((128,128))
            #1*8*128*128
            )
            
        self.last_layer = nn.Sequential(
            nn.Conv2d(16,1,1), #same here, we want a one-channel output image
            nn.Tanh()
            )
            
                     
    def forward(self,input_mask, latent_vector):

        b_size = input_mask.size(0)
        
        x1 = self.seq1(input_mask)
        x2 = self.seq2(x1)
        x3 = self.seq3(x2)
        x4 = self.seq4(x3)
            
        vmean = torch.ones(256).to(device)
        vstd = torch.ones(256).to(device)
            
        for i in range(256):
            vmean[i] = x4[:,i,:,:][0].mean()
            vstd[i] = x4[:,i,:,:][0].std()
                
        var_component = torch.ones(b_size,256,8,8).to(device)
            
        for j in range(256):
            var_component[:,j,:,:] = torch.normal(vmean[j].item(), vstd[j].item(), size=(8,8))
            
        n1 = self.seq_noise_comp_1(latent_vector)
        n1 = n1.view(-1,256,8,8)
        n1 = self.seq_noise_comp_1_1(n1)
        n2 = self.seq_noise_comp_2(n1)
            
        v_out = self.variational_out(var_component)
            
        #print("shape v_out :", v_out.shape, "\nshape n2", n2.shape)
        y1 = torch.cat((v_out, n2), dim=1) #dim=1 in order to cat channel dim
            #1*128*16*16
            
        tmp1 = self.down_1(y1)
        u1 = self.upsample_1(x3)
        y2 = torch.cat((u1,tmp1), dim=1)
        #1*64*32*32
            
        tmp2 = self.down_2(y2)
        u2 = self.upsample_2(x2)
        y3 = torch.cat((u2,tmp2), dim=1)
        #1*32*64*64 
            
        tmp3 = self.down_3(y3)
        u3 = self.upsample_3(x1)
        y4 = torch.cat((u3,tmp3), dim=1)
            
        return self.last_layer(y4)
        #1*3*128*128
        
        
        

class Discriminator_2(nn.Module):
    def __init__(self):
        
         #We assume that Imput Mask and Input Image are not concatenated
         
        super(Discriminator_2, self).__init__()
         
        self.main = nn.Sequential(
            #conv(2,16) because we are in gray scale here
            nn.Conv2d(2,16,4, stride=2 ,padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
             
            nn.Conv2d(16,32,4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
             
            nn.Conv2d(32,64,4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
             
            nn.Conv2d(64,128,4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
             
            nn.Conv2d(128,256,4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256,1,4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Sigmoid() #modification here
            )
        
        self.ups = nn.Upsample((128,128))        
             
    def forward(self, input_mask, input_image):
        tmp_mask = self.ups(input_mask)
        tmp_img = self.ups(input_image)
        #print("SIZE MASK :", input_mask.shape)
        #print("SIZE TMP_MASK :", tmp_mask.shape)
        #print("SIZE IMG :",input_image.shape)
        #print("SIZE TMP_IMG", tmp_img.shape)
        x = torch.cat((tmp_mask, tmp_img), dim=1)
        #print("SIZE sortie :", self.main(x).shape)
        return self.main(x)


G2 = Generator_2().to(device) #pour le faire tourner sur le GPU
D2 = Discriminator_2().to(device) #de meme

G2.apply(weights_init)
D2.apply(weights_init)

print(G2)
print(D2)

loss = nn.BCELoss()

#same optimizer and parameters as the first stage GAN
G2_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(beta1, 0.999))
D2_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(beta1, 0.999))

G2_loss_l = []
D2_loss_l = []

################################################################################################################
#          Ici, on charge les masques que l'on vient de créer avec le premier GAN ci-dessus.                   #
#           Ces masques vont permettre de "nourir" notre second générateur, et vont surtout                    #
#  nous permettre de générer la fameuse paire vérité_terrain/image (la vérité terrain étant le masque binaire) #
################################################################################################################

dataset_generated_mask = dset.ImageFolder(root_data_mask_generated,
                           transform=transforms.Compose([
                               transforms.Grayscale(num_output_channels=1),  #on passe en nuances de gris !
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5), (0.5)),  #Attention, on change ici parce qu'on est passé en grayscale
                           ]))

#dataloader_mask_generated contient donc les masques générés par notre premier GAN
dataloader_mask_generated = torch.utils.data.DataLoader(dataset_generated_mask, batch_size=batch_size, shuffle=False, num_workers=workers)

#NOTE : On a fait exprès de généré autant de masque que le nombre d'image qu'on avait en entrée (sinon ça va bloquer ensuite



################################################################################################################################
#************************************************** Entrainement du 2nd GAN ***************************************************#
################################################################################################################################


print("\nStarting 2nd Stage GAN Training...")

compteur = 0

for epoch in range(num_epoch_stage_2):

  iter_img = iter(dataloader_img)

  for i,data_mask in enumerate(dataloader_mask_generated,0):
    
    #########################
    #**** Discriminator ****#
    #########################

    data_img = next(iter_img)
    
    D2.zero_grad()
    real_mask = data_mask[0].to(device)
    real_img = data_img[0].to(device)

    b_size = real_mask.size(0)
    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

    outputD_real = D2(real_mask, real_img).view(-1)

    #print("outputD.shape", outputD_real.shape)
    #print("label shape", label.shape)

    loss_D_real = loss(outputD_real, label)
    loss_D_real.backward()
    D_real = outputD_real.mean().item()


    noise = torch.randn(b_size, nz, 1, 1, device=device).view(b_size,100)


    fake_img = G2(real_mask, noise) 
    label.fill_(fake_label)


    #print("shape real_masque", real_mask.shape, "shape fake_img",fake_img.shape)
    outputD_fake1 = D2(real_mask, fake_img.detach()).view(-1)

    loss_D_fake = loss(outputD_fake1, label)
    loss_D_fake.backward()

    D_fake1 = outputD_fake1.mean().item()

    loss_D2 = loss_D_real + loss_D_fake

    D2_optimizer.step()

    #######################
    #***** Generator *****#
    #######################

    G2.zero_grad()
    
    label.fill_(real_label)
    
    outputD_fake2 = D2(real_mask, fake_img).view(-1)

    loss_G2 = loss(outputD_fake2, label)
    loss_G2.backward()
    D_fake2 = outputD_fake2.mean().item()

    G2_optimizer.step()


    G2_loss_l.append(loss_G2.item())
    D2_loss_l.append(loss_D2.item())


    if i%50==0 :
      print("iteration ", i+1, "/", len(dataloader_mask),"-- epoch", epoch+1, "/", num_epoch_stage_2, "---- Loss_D : ", loss_D2.item(), " Loss_G : ", loss_G2.item(),
            " D(x) : ", D_x, " D(G(noise1)) : ", D_fake1, " D(G(noise2)) : ", D_fake2)


    if epoch%5==0 and i==0:

      plt.figure(figsize=(14,14))
      plt.axis("off")
      plt.title("Training Images a l'époque "+str(epoch)+" et à l'itération "+str(i))
      img_cpu = np.transpose(vutils.make_grid(fake.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0))
      img_to_save = img_cpu.detach().numpy()
      plt.imsave(result_root_stage_2_batch+str(epoch)+".png", img_to_save)
      plt.cla()
      plt.clf()
      plt.close()


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During 2nd Stage GAN Training")
plt.plot(G2_loss_l,label="G")
plt.plot(D2_loss_l,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

    



