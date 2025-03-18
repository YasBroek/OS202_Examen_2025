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
        # Convertir la représentation RGB en HSV :
        img = img.convert('HSV')
        # On convertit l'image en tableau numpy
        img = np.array(img, dtype=np.double)
        # On double sa taille et on la normalise
        img = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)/255.
        print(f"Nouvelle taille : {img.shape}")
        height, width, _ = img.shape
    else:
        img = None
        height = width = 0
    
    height = comm.bcast(height, root=0)
    width = comm.bcast(width, root=0)
    
    interval = height // nbp
    start = rank * interval
    end = start + interval if rank < nbp - 1 else height
    
    if rank == 0:
        img_chunks = []
        for r in range(nbp):
            img_chunks.append(img[r * interval:(r+1) * interval if r < nbp - 1 else height])
    else:
        img_chunks = None
    
    img_loc = comm.scatter(img_chunks, root=0)
    
    # On crée un masque de flou gaussien
    mask_gaussien = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
    # On crée un masque de netteté
    mask_net = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    
    # On applique le filtre de flou
    blur_image_loc = np.zeros_like(img_loc, dtype=np.double)
    for i in range(3):
        blur_image_loc[:,:,i] = signal.convolve2d(img_loc[:,:,i], mask_gaussien, mode='same')
    
    # On applique le filtre de netteté uniquement sur la luminance :
    sharpen_image_loc = np.zeros_like(img_loc, dtype=np.double)
    sharpen_image_loc[:,:,:2] = blur_image_loc[:,:,:2]
    sharpen_image_loc[:,:,2] = np.clip(signal.convolve2d(blur_image_loc[:,:,2], mask_net, mode='same'), 0., 1.)
    
    sharpen_image_loc = (255.*sharpen_image_loc).astype(np.uint8)
    
    gathered_chunks = comm.gather(sharpen_image_loc, root=0)
    
    # Reconstruire l'image complète
    if rank == 0:
        sharpen_image = np.vstack(gathered_chunks)
        return Image.fromarray(sharpen_image, 'HSV').convert('RGB')
    return None


t1 = time.time()
path = "datas/"
image = path+"paysage.jpg"
doubled_image = double_size(image)
t2 = time.time()
    
# Sauvegarder l'image modifiée (seulement sur le processus principal)
if rank == 0 and doubled_image is not None:
    doubled_image.save("sorties/paysage_double.jpg")
    print("Image sauvegardée")
    print(t2-t1)