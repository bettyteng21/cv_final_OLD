import cv2
import numpy as np
import os

def is_odd(num):
    return num % 2 == 1

def select_blocks(blocks, reference_blocks, num_blocks):
    scores = np.zeros(len(blocks))
    for idx, (block,(y,x)) in enumerate(blocks):
        (ref_block,(a,b)) = reference_blocks[idx]
        mse = np.mean((block - ref_block) ** 2)
        scores[idx] = mse
    sorted_indices = np.argsort(scores)
    output_list = [0] * len(blocks)
    for i in sorted_indices[:num_blocks]:
        output_list[i] = 1
    return output_list

def divide_into_blocks(image, block_size):
    height, width = image.shape[:2]
    blocks = [(image[y:min(y+block_size, height), x:min(x+block_size, width)], (y,x))
            for y in range(0, height, block_size)
            for x in range(0, width, block_size)]
    return blocks

def canny(so_img_paths):
    canny_img = [cv2.Canny(cv2.imread(img, cv2.IMREAD_GRAYSCALE), threshold1=20, threshold2=20)
                for img in so_img_paths]
    return canny_img

def gen_mat(original_image, edges_image):
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(original_image, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(edges_image, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M

def gen_motion_model(canny_img, _list):
    Mat = np.zeros((3,3))
    for (y,x) in _list:
        M = gen_mat(canny_img[y], canny_img[x])
        Mat += M
    Mat /= len(_list)        
    return Mat 

def gen_compensated_img(ref0_img, M):
    height, width = ref0_img.shape
    affine_image = cv2.warpPerspective(ref0_img, M, (width, height))
    return affine_image
        
def gen_mmap(input_dir, output_dir):
    num_lines = 32400
    file_list = os.listdir(input_dir)
    file_list = [f for f in file_list if f.endswith('.png')]
    file_numbers = set(int(os.path.splitext(f)[0]) for f in file_list if f[:-4].isdigit())
    for number in file_numbers:
        filename = f'm_{number:03d}.txt'
        content = "1\n" if number % 2 == 0 else "2\n"
        with open(os.path.join(output_dir, filename), 'w') as file:
            file.writelines([content] * num_lines)