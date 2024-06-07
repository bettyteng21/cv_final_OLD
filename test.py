import cv2
import os

left_folder = "./left_frames/"
right_folder = "./right_frames/"
output_folder = "./merged_frames/"


left_images = os.listdir(left_folder)
right_images = os.listdir(right_folder)


# 對每一對左右圖片進行合併
for left_img, right_img in zip(left_images, right_images):
    # 讀取左右兩邊的圖片
    left_image = cv2.imread(os.path.join(left_folder, left_img))
    right_image = cv2.imread(os.path.join(right_folder, right_img))
    
    # 合併左右兩邊的圖片
    merged_image = cv2.hconcat([left_image, right_image])
    
    # 獲取文件名
    filename = os.path.splitext(left_img)[0] + ".png"
    
    # 保存合併後的圖片到輸出資料夾中
    # cv2.imwrite(os.path.join(output_folder, filename), merged_image)

