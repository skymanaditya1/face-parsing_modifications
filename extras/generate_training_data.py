'''
Code for generating paired training data - (perturbed_image, original_image)
'''
from glob import glob
import os
import os.path as osp
import cv2
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Process

perturbed_image_dir = '/ssd_scratch/cvit/aditya1/CelebAPerturbed/'
original_image_dir = '/ssd_scratch/cvit/aditya1/CelebA-HQ-img/'
paired_data_dir = '/ssd_scratch/cvit/aditya1/CelebAPaired/'

os.makedirs(paired_data_dir, exist_ok=True)

# Save image to disk
def save_image(image_path, image):
    plt.imsave(image_path, image)

# generate paired data - (perturbation, original)
def generate_paired_data(perturbed_image_thread_id):
    perturbed_image_path, thread_id = perturbed_image_thread_id
    print(f'Generating {osp.basename(perturbed_image_path)} using thread : {thread_id}', flush=True)

    perturbed_image = cv2.cvtColor(cv2.imread(perturbed_image_path), cv2.COLOR_BGR2RGB)
    original_filename, extension = osp.basename(perturbed_image_path).split('_')[0], osp.basename(perturbed_image_path).split('.')[1]
    original_image_path = osp.join(original_image_dir, original_filename + '.' + extension)
    original_image = cv2.cvtColor(cv2.imread(original_image_path), cv2.COLOR_BGR2RGB)

    perturbed_resized = cv2.resize(perturbed_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    original_resized = res = cv2.resize(original_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

    paired_image = cv2.hconcat([perturbed_resized, original_resized])
    paired_image_path = osp.join(paired_data_dir, osp.basename(perturbed_image_path))

    save_image(paired_image_path, paired_image)


if __name__ == '__main__':
    n_threads = 5
    p = ThreadPoolExecutor(n_threads)

    perturbed_images = glob(perturbed_image_dir + '/*.jpg')

    jobs = [(perturbed_image, job_id%n_threads) for job_id, perturbed_image in enumerate(perturbed_images)]
    futures = [p.submit(generate_paired_data, job) for job in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]


    # generate_paired_data(perturbed_images[0])