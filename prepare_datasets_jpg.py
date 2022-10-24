import os
import cv2
import numpy as np
import sys
from pathlib import Path
from photutils.datasets import make_noise_image

if __name__ == "__main__":

    folder = sys.argv[1]
    sigma = int(sys.argv[2])
    im_type =  str(sys.argv[3])
    clean_destination = sys.argv[4]
    noisy_destination = sys.argv[5]
    y=0
    x=0
    h=600
    w=1024
    width = 1024
    height = 768
    dim = (width, height)
    mask_file_list = [f for f in os.listdir(folder+'/')]
    outfolder = noisy_destination+'/'+str(clean_destination)#+''/'#+str(sigma)
    Path(outfolder).mkdir(exist_ok=True)
    #Path(clean_destination).mkdir(exist_ok=True)
    for v in range(len(mask_file_list)):
        if im_type == 'jpg' and clean_destination == 'gaussian':
            file_name =  mask_file_list[v]
            #os.rename(file_name, file_name)
            img = cv2.imread(folder + '/' + file_name)
            crop_img_clean = img[y:y+h, x:x+w] #Crop images

            # resize image
            resized = cv2.resize(crop_img_clean, dim, interpolation = cv2.INTER_CUBIC)

            #cv2.imwrite(clean_destination + '/' + file_name, resized)
            noise = make_noise_image(resized.shape, distribution='gaussian',
                                   mean=0, stddev=sigma)
            #noise = np.random.normal(0, sigma, resized.shape)
            resized_noisy_img = resized + noise
            #crop_img_noisy = img[y:y+h, x:x+w]
            # resize image
            #resized = cv2.resize(crop_img_noisy, dim, interpolation = cv2.INTER_CUBIC)

            cv2.imwrite(outfolder + '/' + os.path.splitext(file_name)[0]+'_'+str(sigma)+'.jpg', resized_noisy_img)

        elif im_type == 'jpg' and clean_destination == 'poisson':
            file_name =  mask_file_list[v]
            img = cv2.imread(folder + '/' + file_name)
            crop_img_clean = img[y:y+h, x:x+w] #Crop images

            # resize image
            resized = cv2.resize(crop_img_clean, dim, interpolation = cv2.INTER_CUBIC)
            noise = make_noise_image(resized.shape, distribution='poisson',
                                   mean=sigma)
            #cv2.imwrite(clean_destination + '/' + file_name, resized)

            #noise = np.random.normal(0, sigma, resized.shape)
            resized_noisy_img = resized + noise
            #crop_img_noisy = img[y:y+h, x:x+w]
            # resize image
            #resized = cv2.resize(crop_img_noisy, dim, interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(outfolder + '/' + os.path.splitext(file_name)[0]+'_'+str(sigma)+'.jpg', resized_noisy_img)
        elif im_type == 'jpg' and clean_destination == 'hybrid':
            file_name =  mask_file_list[v]
            img = cv2.imread(folder + '/' + file_name)
            crop_img_clean = img[y:y+h, x:x+w] #Crop images

            # resize image
            resized = cv2.resize(crop_img_clean, dim, interpolation = cv2.INTER_CUBIC)
            noise1 = make_noise_image(resized.shape, distribution='gaussian',
                                   mean=0, stddev=sigma)
            noise2 = make_noise_image(resized.shape, distribution='poisson',
                                   mean=sigma)
            #cv2.imwrite(clean_destination + '/' + file_name, resized)

            #noise = np.random.normal(0, sigma, resized.shape)
            resized_noisy_img = resized + noise1 +noise2
            #crop_img_noisy = img[y:y+h, x:x+w]
            # resize image
            #resized = cv2.resize(crop_img_noisy, dim, interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(outfolder + '/' + os.path.splitext(file_name)[0]+'_'+str(sigma)+'.jpg', resized_noisy_img)
        else:
            print("Error type")

    return mean, sigma, residu
