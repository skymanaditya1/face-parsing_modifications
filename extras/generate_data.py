''' Code used to generate paired data by introducing perturbations 
to the images in the Celebahq dataset 
An image translation model learns to generate the reconstructed 
image from the imperfectly blended face image
'''
import sys
import os
import random
from glob import glob
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Process
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(1, os.path.join(sys.path[0], '..'))

# from face_segmentation import generate_seg_mask

import torch
import torchvision.transforms as transforms
from model import BiSeNet

import os.path as osp
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

perturbed_image_dir = '/ssd_scratch/cvit/aditya1/CelebAPerturbed/'
os.makedirs(perturbed_image_dir, exist_ok=True)
PERTURBATIONS_PER_IDENTITY = 5 # indicates the number of perturbations per identity

ngpus = torch.cuda.device_count()
n_classes = 19
nets = [BiSeNet(n_classes=n_classes).to(device='cuda:{}'.format(id)) for id in range(ngpus)]
# load the pretrained checkpoint
cp_path = '../res/cp/79999_iter.pth'
[net.load_state_dict(torch.load(cp_path)) for net in nets]
# set the model into evaluation mode
[net.eval() for net in nets]

to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

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
    res = cv2.resize(image, None, fx=magnification, fy=magnification, interpolation=cv2.INTER_CUBIC)
    h, w = image.shape[:2]
    
    if magnification >= 1:
        cX, cY = res.shape[1] // 2, res.shape[0] // 2
        left_index = cX - w // 2
        upper_index = cY - h // 2
        modified_image = res[upper_index : upper_index + h, left_index : left_index + w]
    else:
        modified_image = np.zeros((image.shape), dtype=np.uint8)
        hs, ws = res.shape[:2]
        difference_h = h - hs
        difference_w = w - ws
        left_index = difference_w // 2
        upper_index = difference_h // 2
        modified_image[upper_index : upper_index + hs, left_index : left_index + ws] = res
        
    return modified_image

# Applies shear transformation to the image - applies the same share on both the axes
def shear_image(shear, image):
    shear_x, shear_y = shear, shear
    M = np.float32([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ])

    sheared = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return sheared

# Method used to blend the perturbed_image and the face_masked image 
# generate_mask flag can be used to mask the region that will be occupied by the perturbation 
def combine_images(face_mask, perturbed_image, generate_mask=True):
    image_masked = face_mask.copy()
    if generate_mask:
        mask = perturbed_image[..., 0] != 0
        image_masked[mask] = 0
    
    combined_image = image_masked + perturbed_image
    
    return combined_image

# Applies a composite perturbation to a single image
# Generate composite perturbations - generate the number of perturbations to apply randomly
def perturb_image_composite(face_image, face_mask):
    perturbation_functions = [
        translate_horizontal,
        translate_vertical,
        rotate_image,
        resize_image,
        shear_image
    ]

    perturbation_function_map = {
        translate_horizontal : [-20, 20, 1],
        translate_vertical : [-20, 20, 1],
        rotate_image : [-25, 25, 1],
        resize_image : [90, 110, 100],
        shear_image : [-10, 10, 100]
    }

    # indicates the number of perturbations required in the composite perturbation 
    # applies multiple distinct perturbations to the same image
    # composite_perturbations = random.randint(0, len(perturbation_functions)-1)
    composite_perturbations = list()
    # ensures atleast one perturbation is produced
    while len(composite_perturbations) == 0:
        for i, perturbation_function in enumerate(perturbation_functions):
            if random.randint(0, 1):
                composite_perturbations.append(perturbation_function)

    print(f'Perturbations applied : {composite_perturbations}', flush=True)

    for perturbation_function in composite_perturbations:
        perturbation_map = perturbation_function_map[perturbation_function]
        perturbation_value = random.randint(perturbation_map[0], perturbation_map[1])/perturbation_map[2]
        face_image = perturbation_function(perturbation_value, face_image)

    perturbed_image = combine_images(face_mask, face_image)

    return perturbed_image

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
        translate_vertical : [-20, 20, 1],
        rotate_image : [-25, 25, 1],
        resize_image : [90, 110, 100]
    }

    random_perturbation_index = random.randint(0, len(perturbation_functions)-1)
    # random_perturbation_index = 0 # used for debugging
    perturbation_function = perturbation_functions[random_perturbation_index]
    perturbation_map = perturbation_function_map[perturbation_function]
    perturbation_value = random.randint(perturbation_map[0], perturbation_map[1])/perturbation_map[2]
    print(f'Using perturbation : {random_perturbation_index}, with value : {perturbation_value}', flush=True)
    intermediate_perturbed_image = perturbation_function(perturbation_value, face_image)
    perturbed_image = combine_images(face_mask, intermediate_perturbed_image)

    return perturbed_image

# This function segments the face using the face segmentation information
def generate_segmented_face(segmented_image, original_image):
    original_copy = np.asarray(original_image.copy())
    original_copy = np.transpose(original_copy, (2, 0, 1))

    # 3D mask needed for masking face (segmented background)
    field3d_face_mask = np.broadcast_to((segmented_image == 1) | (segmented_image == 2) | (segmented_image == 3) | 
                                        (segmented_image == 4) | (segmented_image == 5) | (segmented_image == 6) |
                                        (segmented_image == 7) | (segmented_image == 8) | (segmented_image == 9) | 
                                        (segmented_image == 10) | (segmented_image == 11) | (segmented_image == 12) | 
                                        (segmented_image == 13), original_copy.shape) 

    # 3D mask needed for masking background (segmenting face)
    field3d_background_mask = np.broadcast_to((segmented_image == 0) | (segmented_image > 13), original_copy.shape)

    background_image = original_copy.copy()
    face_image = original_copy.copy()

    background_image[field3d_face_mask] = 0
    face_image[field3d_background_mask] = 0

    background_image = np.transpose(background_image, (1, 2, 0))
    face_image = np.transpose(face_image, (1, 2, 0))

    # plt.imsave('/home2/aditya1/cvit/content_sync/face-parsing.PyTorch/extras/background_image.png', background_image)
    # plt.imsave('/home2/aditya1/cvit/content_sync/face-parsing.PyTorch/extras/segmented_face.png', face_image)

    return face_image, background_image

def get_random_name(random_chars='abcdefghijklmnopqrstuvwxyz01234566789', random_len=5):
    random_list = list()
    for i in range(random_len):
        random_list.append(random_chars[random.randint(0, len(random_chars)-1)])
    
    return ''.join(random_list)

def save_image(image_path, image):
    plt.imsave(image_path, image)


def generate_segmentation(file, gpu_id):
    with torch.no_grad():
        img = Image.open(file)
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        device = torch.device('cuda:{}'.format(gpu_id))
        img = img.to(device)
        out = nets[gpu_id](img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

    return parsing, image


def test_sample():
    file = '/ssd_scratch/cvit/aditya1/CelebA-HQ-img/13842.jpg'
    gpu_id = 0
    parsing, image = generate_segmentation(file, gpu_id)
    for i in range(PERTURBATIONS_PER_IDENTITY):
        face_image, background_image = generate_segmented_face(parsing, image)
        perturbed_image = perturb_image_composite(face_image, background_image)
        perturbed_filename, extension = osp.basename(file).split('.')
        perturbed_image_path = osp.join(perturbed_image_dir, perturbed_filename + '_' + str(i) + '.' + extension)
        save_image(perturbed_image_path, perturbed_image)


def data_gen(split_gpu):
    split, gpu_id = split_gpu
    for file in split:
        print(f'Processing {file} with GPU {gpu_id}', flush=True)
        parsing, image = generate_segmentation(file, gpu_id)
        for i in range(PERTURBATIONS_PER_IDENTITY):
            face_image, background_image = generate_segmented_face(parsing, image)
            perturbed_image = perturb_image_composite(face_image, background_image)
            perturbed_filename, extension = osp.basename(file).split('.')
            perturbed_image_path = osp.join(perturbed_image_dir, perturbed_filename + '_' + str(i) + '.' + extension)
            save_image(perturbed_image_path, perturbed_image)

if __name__ == '__main__':
    # input_image = '114.jpg'
    # cp_path = '../res/cp/79999_iter.pth'
    # parsing, image = generate_seg_mask(input_image, cp_path)
    # face_image, background_image = generate_segmented_face(parsing, image)
    # perturbed_image = perturb_image(face_image, background_image)

    # perturbed_image_name = get_random_name()
    # print(f'Perturbed image name : {perturbed_image_name}')
    # dir_path = '/home2/aditya1/cvit/content_sync/face-parsing.PyTorch/extras'
    # perturbed_image_path = osp.join(dir_path, perturbed_image_name + '.png')
    # plt.imsave(perturbed_image_path, perturbed_image)

    # Generate the perturbations for all the images in the dataset

    IMAGE_DIR = '/ssd_scratch/cvit/aditya1/CelebA-HQ-img'
    images = glob(IMAGE_DIR + '/*.jpg')
    print(f'Total number of images to process : {len(images)}', flush=True)

    p = ThreadPoolExecutor(ngpus)
    
    # Split the files into the number of GPUs available 
    splits = np.array_split(images, ngpus)

    jobs = [(split, gpu_id) for gpu_id, split in enumerate(splits)]
    futures = [p.submit(data_gen, job) for job in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

    # test_sample()

    # for i, split in enumerate(splits):
    #     generate_segmentation(i, split, perturbed_image_dir)