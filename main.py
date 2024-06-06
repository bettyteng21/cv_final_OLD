import numpy as np
import cv2
import os
import argparse
import glob
import pandas as pd

from detect_object import load_yolo_model, divide_into_object_based_blocks, detect_objects, find_corresponding_blocks, retrieve_bounding_box_image
from model import estimate_motion, cal_model, warp_blocks
from select_block import select_blocks

# Divide an image into non-overlapping blocks of the specified size.
def divide_into_blocks(image, block_size):
    height, width = image.shape[:2]
    blocks = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y+block_size, x:x+block_size]
            blocks.append((block, (y,x)))
    return blocks

# Paste the blocks back to an image
def reconstruct_image_from_blocks(blocks, image_shape):
    compensated_image = np.full(image_shape, 50, dtype=np.uint8)
    for block, (y, x) in blocks:
        block_h, block_w = block.shape[:2]
        # Ensure the block fits within the image dimensions
        if y + block_h > image_shape[0] or x + block_w > image_shape[1]:
            block = cv2.resize(block, (min(block_w, image_shape[1]-x), min(block_h, image_shape[0]-y)))
        compensated_image[y:y+block_h, x:x+block_w] = block

    return compensated_image

# Identify blocks containing objects
def identify_object_blocks(blocks, block_size, direction, full_image):
    object_blocks = []
    full_image_height, full_image_width = len(full_image) // block_size, len(full_image[0]) // block_size

    for idx, bb in enumerate(blocks):
        if idx not in direction:  # Only consider blocks present in the direction list
            continue   

        obj_block, (y, x) = bb
        obj_height, obj_width = obj_block.shape

        # Calculate the range of blocks the object intersects with
        start_block_y = y // block_size
        end_block_y = (y + obj_height - 1) // block_size
        start_block_x = x // block_size
        end_block_x = (x + obj_width - 1) // block_size

        # Collect all intersecting blocks
        for by in range(start_block_y, end_block_y + 1):
            for bx in range(start_block_x, end_block_x + 1):
                if 0 <= by < full_image_height and 0 <= bx < full_image_width:
                    block_value = full_image[by * block_size:(by + 1) * block_size, bx * block_size:(bx + 1) * block_size]
                    object_blocks.append((block_value,(by* block_size, bx* block_size)))
    return object_blocks


def main():
    parser = argparse.ArgumentParser(description='main function of GMC')
    parser.add_argument('--input_path', default='./frames/', help='path to read input frames')
    parser.add_argument('--output_path', default='./output/', help='path to put output files')
    parser.add_argument('--csv_file', default='./processing_order.csv', help='processing order CSV file')
    args = parser.parse_args()

    image_files = glob.glob(os.path.join(args.input_path, '[0-9][0-9][0-9].png'))
    if not image_files:
        print("Cannot find image files from given link.")
        return

    block_size = 16
    num_blocks = 13000
    psnr_list = []

    df = pd.read_csv(args.csv_file)

    net, output_layers = load_yolo_model()

    for idx, row in df.iterrows():

        target = row['Target Picture']
        ref0 = row['Reference Pic0']
        ref1 = row['Reference Pic1']

        if '(X)' in str(target):
            # skip the ones that are labeled (X)
            continue
        
        # Load img
        target, ref0, ref1 = int(target), int(ref0), int(ref1)
        print('\nCurrent processing order '+str(idx)+', target frame '+str(target))
        # if target in [1]:
        #     pass
        # else:
        #     continue

        target_img_path = os.path.join(args.input_path, f'{target:03}.png')
        ref0_img_path = os.path.join(args.input_path, f'{ref0:03}.png')
        ref1_img_path = os.path.join(args.input_path, f'{ref1:03}.png')

        target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
        ref0_img = cv2.imread(ref0_img_path, cv2.IMREAD_GRAYSCALE)
        ref1_img = cv2.imread(ref1_img_path, cv2.IMREAD_GRAYSCALE)

        # motion model calculation start ##################################################################
        ref0_boxes = detect_objects(ref0_img, net, output_layers)
        target_boxes = detect_objects(target_img, net, output_layers)

        target_blocks, target_mask = divide_into_object_based_blocks(target_img, target_boxes)
        ref0_blocks, ref0_mask = divide_into_object_based_blocks(ref0_img, ref0_boxes)

        target_boxes_img = []
        ref0_boxes_img = []

        for i in range(len(target_blocks)):
            img = retrieve_bounding_box_image(target_blocks[i])
            if img.size != 0:
                target_boxes_img.append(img)
                # cv2.imwrite('./temp_output/target/'+str(i)+'.png', img)

        for i in range(len(ref0_blocks)):
            img = retrieve_bounding_box_image(ref0_blocks[i])
            if img.size != 0:
                ref0_boxes_img.append(img)

                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                img = cv2.filter2D(img, -1, kernel)    
                # cv2.imwrite('./temp_output/ref0/'+str(i)+'.png', img)


        # 找出target obj對應的ref0 obj & ref1 obj，並記錄他們對應的index
        ref0_mapping = find_corresponding_blocks(target_boxes_img, ref0_boxes_img, threshold=10)

        # Output the mapping
        with open(f"temp_output/mapping.txt", "w") as f:
            for i, r0, in enumerate(ref0_mapping):
                f.write(f"Target object {i}: ref0 block {r0}\n")

        toward, same_speed = [], []
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        for idx, ((target_block, (y,x)),idx_ref0) in enumerate(zip(target_blocks, ref0_mapping)):
            (ref0_block, (y0,x0)) = ref0_blocks[idx_ref0]
            if idx_ref0 == -1 :
                continue
            
            if y0+ref0_block.shape[0] > ref0_img.shape[0] or x0+ref0_block.shape[1] > ref0_img.shape[1]:
                # 當超出圖片範圍時，resize成可以fit進去圖片的大小
                ref0_block = cv2.resize(ref0_block, (min(ref0_block.shape[0], ref0_img.shape[1]-x0), min(ref0_block.shape[1], ref0_img.shape[0]-y0)), interpolation=cv2.INTER_CUBIC)

            # Find corresponding blocks in reference images
            ref0_block = ref0_img[y0:y0+ref0_block.shape[0], x0:x0+ref0_block.shape[1]]

            # Make the image look clearer
            target_block = cv2.filter2D(target_block, -1, kernel) 
            ref0_block = cv2.filter2D(ref0_block, -1, kernel)       

            # Apply feature matching for motion compensation
            direction, magnitude = None, None
            result = estimate_motion(target_block, ref0_block, (y,x), (y0,x0))
            if result:
                direction, magnitude = result
                # print(f"The object {idx} is moving {direction} with a magnitude of {magnitude}")
            # else:
                # print(f"The object {idx} is One of the descriptor sets is None")

            # 篩選level
            if (target % 32) in [16]: # level 1
                if result and direction=='towards the camera' and magnitude>150:
                    # 向我移動, model 1
                    M1 = cal_model(target_block, ref0_block, (y,x), (y0,x0))
                    toward.append(idx)
                elif result and magnitude < 50: # 與我同速, model 2
                    M2 = cal_model(target_block, ref0_block, (y,x), (y0,x0))
                    same_speed.append(idx)
                else: # 靜止, model 3
                    M3 = cal_model(target_block, ref0_block, (y,x), (y0,x0))


            elif (target % 32) in [8,24]:
                if result and magnitude < 30: # 與我同速, model 4
                    M4 = cal_model(target_block, ref0_block, (y,x), (y0,x0))
                    same_speed.append(idx)
                else: # 靜止, model 5
                    M5 = cal_model(target_block, ref0_block, (y,x), (y0,x0))

            elif (target % 32) in [4,12,20,28]:
                if result and magnitude < 15: # 與我同速, model 6
                    M6 = cal_model(target_block, ref0_block, (y,x), (y0,x0))
                    same_speed.append(idx)
                else: # 靜止, model 7
                    M7 = cal_model(target_block, ref0_block, (y,x), (y0,x0))
            
            elif (target % 32) in [2,6,10,14,18,22,26,30]:
                if result and magnitude < 7: # 與我同速, model 6
                    M8 = cal_model(target_block, ref0_block, (y,x), (y0,x0))
                    same_speed.append(idx)
                else: # 靜止, model 7
                    M9 = cal_model(target_block, ref0_block, (y,x), (y0,x0))

            else:
                if result and magnitude < 3: # 與我同速, model 6
                    M10 = cal_model(target_block, ref0_block, (y,x), (y0,x0))
                    same_speed.append(idx)
                else: # 靜止, model 7
                    M11 = cal_model(target_block, ref0_block, (y,x), (y0,x0))

                    

        # motion model calculation ends ##################################################################
        
        # reconstruct target image start ##################################################################

        # Combine the compensated images into a single image
        compensated_image = np.full(target_img.shape, 50, dtype=np.uint8)
        ref0_small_block = divide_into_blocks(ref0_img, block_size)

        # 以16*16 block為單位，找出object所在的blocks
        toward_object_blocks = identify_object_blocks(ref0_blocks, block_size, toward, ref0_img)
        same_speed_object_blocks = identify_object_blocks(ref0_blocks, block_size, same_speed, ref0_img)
        
        if (target % 32) in [16]:
            warped_blocks_M1 = warp_blocks(toward_object_blocks, M1, (block_size,block_size))
            warped_blocks_M2 = warp_blocks(same_speed_object_blocks, M2, (block_size,block_size))
            warped_blocks_M3 = warp_blocks(ref0_small_block, M3, (block_size,block_size))
            
            for block, (y, x) in warped_blocks_M3:
                block_h, block_w = block.shape[:2]
                compensated_image[y:y+block_h, x:x+block_w] = block

            for block, (y, x) in warped_blocks_M1:
                block_h, block_w = block.shape[:2]
                compensated_image[y:y+block_h, x:x+block_w] = block
            
            for block, (y, x) in warped_blocks_M2:
                block_h, block_w = block.shape[:2]
                compensated_image[y:y+block_h, x:x+block_w] = block

        elif (target % 32) in [8,24]:
            warped_blocks_M4 = warp_blocks(same_speed_object_blocks, M4, (block_size,block_size))
            warped_blocks_M5 = warp_blocks(ref0_small_block, M5, (block_size,block_size))
            
            for block, (y, x) in warped_blocks_M4:
                block_h, block_w = block.shape[:2]
                compensated_image[y:y+block_h, x:x+block_w] = block

            for block, (y, x) in warped_blocks_M5:
                block_h, block_w = block.shape[:2]
                compensated_image[y:y+block_h, x:x+block_w] = block
        
        elif (target % 32) in [4,12,20,28]:
            warped_blocks_M6 = warp_blocks(same_speed_object_blocks, M6, (block_size,block_size))
            warped_blocks_M7 = warp_blocks(ref0_small_block, M7, (block_size,block_size))
            
            for block, (y, x) in warped_blocks_M6:
                block_h, block_w = block.shape[:2]
                compensated_image[y:y+block_h, x:x+block_w] = block

            for block, (y, x) in warped_blocks_M7:
                block_h, block_w = block.shape[:2]
                compensated_image[y:y+block_h, x:x+block_w] = block

        elif (target % 32) in [2,6,10,14,18,22,26,30]:
            warped_blocks_M8 = warp_blocks(same_speed_object_blocks, M8, (block_size,block_size))
            warped_blocks_M9 = warp_blocks(ref0_small_block, M9, (block_size,block_size))
            
            for block, (y, x) in warped_blocks_M8:
                block_h, block_w = block.shape[:2]
                compensated_image[y:y+block_h, x:x+block_w] = block

            for block, (y, x) in warped_blocks_M9:
                block_h, block_w = block.shape[:2]
                compensated_image[y:y+block_h, x:x+block_w] = block

        else:
            warped_blocks_M10 = warp_blocks(same_speed_object_blocks, M10, (block_size,block_size))
            warped_blocks_M11 = warp_blocks(ref0_small_block, M11, (block_size,block_size))
            
            for block, (y, x) in warped_blocks_M10:
                block_h, block_w = block.shape[:2]
                compensated_image[y:y+block_h, x:x+block_w] = block

            for block, (y, x) in warped_blocks_M11:
                block_h, block_w = block.shape[:2]
                compensated_image[y:y+block_h, x:x+block_w] = block


        cv2.imwrite(os.path.join(args.output_path, f'{target:03}.png'), compensated_image)

        target_small_block = divide_into_blocks(target_img, block_size)
        compensated_blocks = divide_into_blocks(compensated_image, block_size)
        selected_blocks = select_blocks(compensated_blocks, target_small_block, num_blocks)

        # Eval for printing current psnr
        mask = np.array(selected_blocks).astype(bool)
        assert np.sum(mask) == 13000, 'The number of selection blocks should be 13000'
        s = compensated_image.reshape(2160//16, 16, 3840//16, 16).swapaxes(1, 2).reshape(-1, 16, 16).astype(int)
        g = target_img.reshape(2160//16, 16, 3840//16, 16).swapaxes(1, 2).reshape(-1, 16, 16).astype(int)
        s = s[mask]
        g = g[mask]
        assert not (s == g).all(), "The prediction should not be the same as the ground truth"
        mse = np.sum((s-g)**2)/s.size
        psnr_curr = 10*np.log10((255**2)/mse)
        psnr_list.append(psnr_curr)
        print('Current psnr= '+str(psnr_curr))

        # output selection map: s_xxx.txt
        output_path = os.path.join(args.output_path, f's_{target:03}.txt')
        smap_file = open(output_path,'w')
        for s in selected_blocks:
            smap_file.write(str(s)+'\n')
        smap_file.close()

    psnr_list = np.array(psnr_list)
    avg_psnr = np.mean(psnr_list)
    print('Avg psnr: '+str(avg_psnr))


if __name__ == '__main__':
    main()