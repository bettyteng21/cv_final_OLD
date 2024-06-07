import numpy as np
import cv2
import os
import argparse
import pandas as pd
from utils import *


def main():
    parser = argparse.ArgumentParser(description='main function of GMC')
    parser.add_argument('--input_path', default='./frames/', help='path to read input frames')
    parser.add_argument('--output_path', default='./output/', help='path to put output files')
    parser.add_argument('--csv_file', default='./processing_order.csv', help='processing order CSV file')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs('model_map', exist_ok=True)
    image_name = ['%03d.png' % i for i in range(129)]
    gt_img_paths = [os.path.join('frames', name) for name in image_name]
    canny_img = canny(gt_img_paths)
    Hierarchical5 = [(i, i+2) for i in range(0, 127, 2)]
    M = gen_motion_model(canny_img, Hierarchical5)
    num_blocks = 13000
    df = pd.read_csv(args.csv_file)

    for idx, row in df.iterrows():
        target = row['Target Picture']
        ref0 = row['Reference Pic0']
        ref1 = row['Reference Pic1']

        if '(X)' in str(target):
            continue

        target, ref0, ref1 = int(target), int(ref0), int(ref1)
        print('\nCurrent processing order '+str(idx)+', target frame '+str(target))

        model_map_path = os.path.join('model_map', f'm_{target:03}.txt')
        if os.path.isfile(model_map_path):
            open(model_map_path, 'w').close()
        
        target_img_path = os.path.join(args.input_path, f'{target:03}.png')
        ref0_img_path = os.path.join(args.input_path, f'{ref0:03}.png')
        ref1_img_path = os.path.join(args.input_path, f'{ref1:03}.png')

        target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
        ref0_img = cv2.imread(ref0_img_path, cv2.IMREAD_GRAYSCALE)
        ref1_img = cv2.imread(ref1_img_path, cv2.IMREAD_GRAYSCALE)
        
        compensated_image = gen_compensated_img(ref0_img, M) if is_odd(target) else ref0_img
        

        cv2.imwrite(os.path.join(args.output_path, f'{target:03}.png'), compensated_image)
        
        compensated_blocks = divide_into_blocks(compensated_image, 16)
        original_blocks = divide_into_blocks(target_img, 16)
        selected_blocks = select_blocks(compensated_blocks, original_blocks, num_blocks)

        output_path = os.path.join(args.output_path, f's_{target:03}.txt')
        smap_file = open(output_path,'w')
        for s in selected_blocks:
            smap_file.write(str(s)+'\n')
        smap_file.close()
    
    gen_mmap(args.output_path, './model_map')
        
if __name__ == '__main__':
    main()