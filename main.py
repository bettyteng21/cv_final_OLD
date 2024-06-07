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

def gen_img(_list, M, path):
    for (y,x) in _list:
        med = (y+x) // 2
        original_image = cv2.imread(path[y], cv2.IMREAD_GRAYSCALE)
        edges_image = cv2.imread(path[x], cv2.IMREAD_GRAYSCALE)
        target_img = cv2.imread(path[med], cv2.IMREAD_GRAYSCALE)
        height, width = edges_image.shape
        affine_image = original_image
        affine_image = cv2.warpPerspective(affine_image, M, (width, height))
        cv2.imwrite(f'./output/{med:03}.png', affine_image)
        
        compensated_blocks = divide_into_blocks(affine_image, 16)
        original_blocks = divide_into_blocks(target_img, 16)
        selected_blocks = select_blocks(compensated_blocks, original_blocks, 13000)

        output_path = f'./output/s_{med:03}.txt'
        smap_file = open(output_path,'w')
        for s in selected_blocks:
            smap_file.write(str(s)+'\n')
        smap_file.close()

os.makedirs('output', exist_ok=True)
image_name = ['%03d.png' % i for i in range(129)]
gt_img_paths = [os.path.join('frames', name) for name in image_name]

canny_img = canny(gt_img_paths)
frame_img = [cv2.imread(img, cv2.IMREAD_GRAYSCALE)
             for img in gt_img_paths]

mylist0 = [(i, i+1) for i in range(128)]
mylist1 = [(i, i+32) for i in range(0, 128, 32)]
mylist2 = [(i, i+16) for i in range(0, 113, 16)]
mylist3 = [(i, i+8) for i in range(0, 121, 8)]
mylist4 = [(i, i+4) for i in range(0, 125, 4)]
mylist5 = [(i, i+2) for i in range(0, 127, 2)]

# M1 = gen_motion_model(canny_img, mylist1)
# M2 = gen_motion_model(canny_img, mylist2)
# M3 = gen_motion_model(canny_img, mylist3)
# M4 = gen_motion_model(canny_img, mylist4)
M5 = gen_motion_model(canny_img, mylist5)

# gen_img(mylist1, M1, gt_img_paths)
# gen_img(mylist2, M2, gt_img_paths)
# gen_img(mylist3, M3, gt_img_paths)
# gen_img(mylist4, M4, gt_img_paths)
# gen_img(mylist5, M5, gt_img_paths)

M = np.eye(3)
gen_img(mylist1, M, gt_img_paths)
gen_img(mylist2, M, gt_img_paths)
gen_img(mylist3, M, gt_img_paths)
gen_img(mylist4, M, gt_img_paths)
gen_img(mylist5, M5, gt_img_paths)
