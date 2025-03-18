# Ce programme double la taille d'une image en assayant de ne pas trop pixeliser l'image.

from PIL import Image
import os
import numpy as np
from scipy import signal
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD.Dup()
rank = comm.Get_rank()
nbp = comm.Get_size()

# Fonction pour doubler la taille d'une image sans trop la pixeliser
def double_size(image):
    # On charge l'image
    if rank == 0:
        img = Image.open(image)
        print(f"Taille originale {img.size}")
        img = img.convert('HSV')
        # On convertit l'image en tableau numpy
        img = np.array(img, dtype=np.double)
        # On double sa taille et on normalise
        img = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)/255.
        print(f"Nouvelle taille : {img.shape}")
        height, width, _ = img.shape
    else:
        img_array = None
        height = width = 0

    height = comm.bcast(height, root=0)
    width = comm.bcast(width, root=0)

    interval = height // nbp
    start = rank * interval
    end = start + interval if rank < nbp - 1 else height
    # On crée un masque de flou gaussien pour la teinte et la saturation (H et S)
    mask_gaussien = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.

    if rank == 0:
        img_chunks = []
        for r in range(nbp):
            img_chunks.append(img[r * interval : (r+1) * interval if r < nbp - 1 else height])
    else:
        img_array = None
        img_chunks = None
    
    img_loc = comm.scatter(img_chunks, root=0)
    
    # On applique le filtre de flou
    blur_image_loc = np.zeros_like(img_loc, dtype=np.double)
    for i in range(2):
        blur_image_loc[:,:,i] = signal.convolve2d(img_loc[:,:,i], mask_gaussien, mode='same')
    blur_image_loc[:,:,2] = img_loc[:,:,2]
    # On crée un masque de flou 5x5 :
    mask_5_5 = -np.array([[1., 4., 6., 4., 1.], [4., 16., 24., 16., 4.], [6., 24., -476., 24., 6.], [4., 16., 24., 16., 4.], [1., 4., 6., 4., 1.]]) / 256
    # On applique le filtre sur la luminance:
    blur_image_loc[:,:,2] = np.clip(signal.convolve2d(blur_image_loc[:,:,2], mask_5_5, mode='same'), 0., 1.)
    blur_image_loc = (255.*blur_image_loc).astype(np.uint8)
    gathered_chunks = comm.gather(blur_image_loc, root = 0)

    if rank == 0:
        img_final = np.vstack(gathered_chunks)
        # On retourne l'image modifiée
        return Image.fromarray(img_final, 'HSV').convert('RGB')
t1 = time.time()
path = "datas/"
image = path+"paysage.jpg"
doubled_image = double_size(image)
t2 = time.time()
# On sauvegarde l'image modifiée
if rank == 0:
    doubled_image.save("sorties/paysage_double_2.jpg")
    print("Image sauvegardée")
    print(t2 - t1)
