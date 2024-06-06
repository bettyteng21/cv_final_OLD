import cv2
import numpy as np

def estimate_motion(blk_target, blk_ref0, pos_target, pos_ref0):
    orb = cv2.ORB_create()
    kp_target, des_target = orb.detectAndCompute(blk_target, None)
    kp_ref0, des_ref0 = orb.detectAndCompute(blk_ref0, None)

    if des_target is None or des_ref0 is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_target, des_ref0)

    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:10]  # Limit to the top 10 matches for robustness

    if len(good_matches) > 3:
        dst_pts = np.float32([kp_target[m.queryIdx].pt for m in good_matches])
        src_pts = np.float32([kp_ref0[m.trainIdx].pt for m in good_matches])

        dst_pts += pos_target
        src_pts += pos_ref0

        # Calculate centroid of the object in both frames
        cen_target = np.mean(dst_pts, axis=0)
        cen_ref0 = np.mean(src_pts, axis=0)

        movement_vector = cen_target - cen_ref0

        # Determine the direction of movement
        direction = "towards the camera" if movement_vector[1] < 0 else "away from the camera"

        # Calculate the magnitude of the movement
        magnitude = np.linalg.norm(movement_vector)

        return direction, magnitude
    else:
        # print("Not enough matches are found")
        return None

def cal_model(blk_target, blk_ref0, pos_target, pos_ref0):
    (y,x), (y0,x0) = pos_target, pos_ref0
    src_pts = np.array([
        [x0, y0],
        [x0 + blk_ref0.shape[1], y0],
        [x0 + blk_ref0.shape[1], y0 + blk_ref0.shape[0]],
        [x0, y0 + blk_ref0.shape[0]]
    ], dtype=np.float32)

    dst_pts = np.array([
        [x, y],
        [x + blk_target.shape[1], y],
        [x + blk_target.shape[1], y + blk_target.shape[0]],
        [x, y + blk_target.shape[0]]
    ], dtype=np.float32)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) 

    return M


# Apply perspective transform using the homography matrix
def warp_blocks(blocks, M, image_shape):
    warped_blocks = []
    for block, (y, x) in blocks:
        block_h, block_w = block.shape[:2]

        # Warp the block using the homography matrix
        transformed_block = cv2.warpPerspective(block, M, image_shape, borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_CUBIC)

        # Add the transformed block and its new position to the list
        warped_blocks.append((transformed_block, (y, x)))

    return warped_blocks

