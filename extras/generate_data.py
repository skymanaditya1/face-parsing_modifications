''' Code used to generate paired data by introducing perturbations 
to the images in the Celebahq dataset 
An image translation model learns to generate the reconstructed 
image from the imperfectly blended face image
'''
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from face_segmentation import generate_seg_mask

import torch
import torchvision.transforms as transforms

import os.path as osp
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Perturbation functions - 
# 1. Translation along the horizontal direction 
# 2. Translation along the vertical direction 
# 3. Clockwise and anti-clockwise rotation 
# 4. Resize (zoom-in and zoom-out)

# Translates the image in the horizontal direction 
def translate_horizontal(x, image):
    M = np.float32([
        [1, 0, x],
        [0, 1, 0]
    ])
    
    # Apply the translation on the image
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

# Translates the image in the vertical direction
def translate_vertical(y, image):
    M = np.float32([
        [1, 0, 0],
        [0, 1, y]
    ])
    
    # Apply the translation to the image 
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    return shifted

# Rotate the image in the clockwise or anti-clockwise direction by the specified degrees of rotation
def rotate_image(rotation, image):
    # Rotate the image about the center point 
    h, w = image.shape[:2]
    cX, cY = (w//2, h//2)
    
    M = cv2.getRotationMatrix2D((cX, cY), rotation, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    return rotated

# Resize the image
# If the image is zoomed out, then add padding to match the dimension of the image
# If the image is zoomed in, then crop the image to match dimension of the image 
def resize_image(magnification, image):
    res = cv2.resize(image_with_face, None, fx=magnification, fy=magnification, interpolation=cv2.INTER_CUBIC)
    h, w = image.shape[:2]
    
    if magnification >= 1:
        cX, cY = res.shape[1] // 2, res.shape[0] // 2
        left_index = cX - w // 2
        upper_index = cY - h // 2
        modified_image = res[upper_index : upper_index + h, left_index : left_index + w]
    else:
        modified_image = np.zeros((image.shape), dtype=np.uint)
        hs, ws = res.shape[:2]
        difference_h = h - hs
        difference_w = w - ws
        left_index = difference_w // 2
        upper_index = difference_h // 2
        modified_image[upper_index : upper_index + hs, left_index : left_index + ws] = res
        
    return modified_image

# Method used to blend the perturbed_image and the face_masked image 
# generate_mask flag can be used to mask the region that will be occupied by the perturbation 
def combine_images(face_mask, perturbed_image, generate_mask=True):
    image_masked = face_mask.copy()
    if generate_mask:
        mask = perturbed_image[..., 0] != 0
        image_masked[mask] = 0
    
    combined_image = image_masked + perturbed_image
    
    return combined_image

# The perturb image function randomly selects a perturbation and the amount to perturb the face_image 
# The perturbed image is then combined with the face_mask to produce the final image
# Potentially multiple perturbation functions can be combined to generate more complex perturbations
def perturb_image(face_image, face_mask):
    perturbation_functions = [
        translate_horizontal,
        translate_vertical,
        rotate_image,
        resize_image
    ]

    perturbation_function_map = {
        translate_horizontal : [-20, 20, 1],
        translat_vertical : [-20, 20, 1],
        rotate_image : [-25, 25, 1],
        resize_image : [90, 110, 100]
    }

    random_perturbation_index = random.randint(0, len(perturbation_functions)-1)
    perturbation_function = perturbation_functions[random_perturbation_index]
    perturbation_map = perturbation_function_map[perturbation_function]
    perturbation_value = random.randint(perturbation_map[0], perturbation_map[1])/perturbation_map[2]
    intermediate_perturbed_image = perturbation_function(perturbation_value, face_image)
    perturbed_image = combine_images(face_mask, intermediate_perturbed_image)

    return perturbed_image

# This function segments the face using the face segmentation information
def generate_segmented_face(segmented_image, original_image):
    original_copy = np.asarray(original_image.copy())
    original_copy = np.transpose(original_copy, (2, 0, 1))

    # 3D mask needed for masking face (segmented background)
    field3d_face_mask = np.broadcast_to((parsing == 1) | (parsing == 2) | (parsing == 3) | 
                                        (parsing == 4) | (parsing == 5) | (parsing == 6) |
                                        (parsing == 7) | (parsing == 8) | (parsing == 9) | 
                                        (parsing == 10) | (parsing == 11) | (parsing == 12) | 
                                        (parsing == 13), original_copy.shape) 

    # 3D mask needed for masking background (segmenting face)
    field3d_background_mask = np.broadcast_to((parsing == 0) | (parsing > 13), original_copy.shape)

    background_image = original_copy.copy()
    face_image = original_copy.copy()

    background_image[field3d_face_mask] = 0
    face_image[field3d_background_mask] = 0

    background_image = np.transpose(background_image, (1, 2, 0))
    face_image = np.transpose(face_image, (1, 2, 0))

    plt.imsave('/home2/aditya1/cvit/content_sync/face-parsing.PyTorch/extras/background_image.png', background_image)
    plt.imsave('/home2/aditya1/cvit/content_sync/face-parsing.PyTorch/extras/segmented_face.png', face_image)

    return face_image, background_image

if __name__ == '__main__':
    input_image = '../makeup/116_ori.png'
    cp_path = '../res/cp/79999_iter.pth'
    parsing, image = generate_seg_mask(input_image, cp_path)
    face_image, background_image = generate_segmented_face(parsing, image)
