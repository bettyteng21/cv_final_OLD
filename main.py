import cv2
import numpy as np
import os

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
    for img in so_img_paths:
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(image, threshold1=20, threshold2=20)
        cv2.imwrite(f'./frame_canny/{img[-7:-4]}.png', edges)

def gen_mat(original_image, edges_image):
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(original_image, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(edges_image, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.estimateAffine2D(src_pts, dst_pts)
    return M

def gen_motion_model(_list):
    Mat = np.zeros((2,3))
    for (y,x) in _list:
        original_image_path = f'./frame_canny/{y:03}.png'
        edges_image_path = f'./frame_canny/{x:03}.png'
        original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        edges_image = cv2.imread(edges_image_path, cv2.IMREAD_GRAYSCALE)
        M = gen_mat(original_image, edges_image)
        Mat += M
    Mat /= len(_list)        
    return Mat 

def gen_img(_list, M):
    for (y,x) in _list:
        idx = (y+x) // 2
        original_image = cv2.imread(f'./frames/{y:03}.png', cv2.IMREAD_GRAYSCALE)
        edges_image = cv2.imread(f'./frames/{x:03}.png', cv2.IMREAD_GRAYSCALE)
        target_img = cv2.imread(f'./frames/{idx:03}.png', cv2.IMREAD_GRAYSCALE)
        height, width = edges_image.shape
        affine_image = cv2.warpAffine(original_image, M, (width, height))
        cv2.imwrite(f'./output/{idx:03}.png', affine_image)
        
        compensated_blocks = divide_into_blocks(affine_image, 16)
        original_blocks = divide_into_blocks(target_img, 16)
        selected_blocks = select_blocks(compensated_blocks, original_blocks, 13000)

        output_path = f'./output/s_{idx:03}.txt'
        smap_file = open(output_path,'w')
        for s in selected_blocks:
            smap_file.write(str(s)+'\n')
        smap_file.close()
        
        
os.makedirs('frame_canny', exist_ok=True)
os.makedirs('output', exist_ok=True)
image_name = ['%03d.png' % i for i in range(129)]
gt_img_paths = [os.path.join('frames', name) for name in image_name]

mylist1 = [(i, i+32) for i in range(0, 128, 32)]
mylist2 = [(i, i+16) for i in range(0, 113, 16)]
mylist3 = [(i, i+8) for i in range(0, 121, 8)]
mylist4 = [(i, i+4) for i in range(0, 125, 4)]
mylist5 = [(i, i+2) for i in range(0, 127, 2)]

canny(gt_img_paths)

M1 = gen_motion_model(mylist1)
M2 = gen_motion_model(mylist2)
M3 = gen_motion_model(mylist3)
M4 = gen_motion_model(mylist4)
M5 = gen_motion_model(mylist5)

gen_img(mylist1, M1)
gen_img(mylist2, M2)
gen_img(mylist3, M3)
gen_img(mylist4, M4)
gen_img(mylist5, M5)
