import random
from numba import cuda
import numba
import numba.cuda
import numpy as np
import cv2
import matplotlib.pyplot as plt
import cupy as cp
import time





def init_centres(k):
    
    centres = np.zeros((k, 3), dtype=int)  

    for i in range(k):
        x_random = random.randint(0, 255)  
        y_random = random.randint(0, 255)
        z_random = random.randint(0, 255)   

        centres[i] = (x_random, y_random,z_random) 

    return centres




@cuda.jit
def kernel_calcul_distances_affectation(image,matrice,centres):
    idx = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    idy = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    longeur = image.shape[0]
    largeur = image.shape[1]
    
    if idx < largeur and idy < longeur:

        min_dist = float('inf')
        cluster_index = -1

        for i in range(len(centres)):
            cx , cy ,cz = centres[i]
            distance = abs(image[idy][idx][0] - cx) + abs(image[idy][idx][1] - cy) + abs(image[idy][idx][2] - cz)
            if distance < min_dist:
                min_dist = distance
                cluster_index = i
    
         
        matrice[idy][idx] = cluster_index
        
         


@cuda.jit
def kernel_calcul_nouveaux_centres(centres_prev,centres_nouv,image,matrice):
    idx = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    longeur = image.shape[0]
    largeur = image.shape[1]

    sum_r=0
    sum_g=0
    sum_b=0
    count=0
    

    if idx < len(centres_prev):
        for i in range(longeur):
            for j in range(largeur):
                if matrice[i][j] == idx:
                    sum_r += image[i][j][0]
                    sum_g += image[i][j][1]
                    sum_b += image[i][j][2]
                    count+=1

        
                    
        centres_nouv[idx] = (sum_r // count, sum_g // count, sum_b // count)





def main():

    start_time = time.time()
    image = cv2.imread("image.jpg",cv2.IMREAD_COLOR)

    threads_per_block = (16,16) 
    bloc_per_grid = ((image.shape[1] + threads_per_block[0] - 1) // threads_per_block[0],
                      (image.shape[0] + threads_per_block[1] - 1) // threads_per_block[1] )

    k=80 ## nombres de clusters ( couleurs )
    tolerance = 5


    centres = init_centres(k)
    matrice = np.zeros((image.shape[0],image.shape[1]))
    image_final = image.copy()
    centres_nouv = np.zeros((k, 3), dtype=int) 
    tab = np.zeros((k,2),dtype=int)

    d_image = cuda.to_device(image)
    d_matrice = cuda.to_device(matrice)
    d_centres = cuda.to_device(centres)
    d_centres_nouv = cuda.to_device(centres_nouv)
    

    
    kernel_calcul_distances_affectation[bloc_per_grid,threads_per_block](d_image,d_matrice,d_centres)
    cuda.synchronize()
    new_matrice = d_matrice.copy_to_host()

    
    
    kernel_calcul_nouveaux_centres[bloc_per_grid,threads_per_block](d_centres,d_centres_nouv,d_image,new_matrice)
    cuda.synchronize()
    new_centres = d_centres_nouv.copy_to_host()

    d_matrice = cuda.to_device(new_matrice)
    d_centres = cuda.to_device(new_centres)

    difference = np.linalg.norm(centres - new_centres)

    while difference>tolerance:
        
        #calcul distances
        kernel_calcul_distances_affectation[bloc_per_grid,threads_per_block](d_image,d_matrice,d_centres_nouv)
        
        new_matrice = d_matrice.copy_to_host()
        d_matrice = cuda.to_device(new_matrice)
        
        #calcul nouveaux centres
        previous_centres = new_centres.copy()
        kernel_calcul_nouveaux_centres[bloc_per_grid,threads_per_block](d_centres,d_centres_nouv,d_image,d_matrice)
        
        new_centres = d_centres_nouv.copy_to_host()

        d_centres = cuda.to_device(new_centres)

        difference = np.linalg.norm( previous_centres - new_centres)
    
    

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            index = int(new_matrice[i][j])
            image_final[i][j]=new_centres[index]
    

    end_time = time.time()

    print("Temps d'execution avec k=",k," : ",end_time-start_time)
    
    cv2.imshow('image depart',image)
    cv2.imshow('image fin',image_final)
    cv2.waitKey(0)


    

main()