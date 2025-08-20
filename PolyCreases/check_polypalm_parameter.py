'''
From the saved polynomial paramters, it try to reconstruct the polypalm and
estimate the similarity with the original constructed polypalm image
'''
import os
import cv2
import numpy as np

def show(curve_set, secondary_line, thickness=25):
    
    scale_factor = 3000
    image = np.ones((scale_factor, scale_factor, 3), dtype=np.uint8) * 255

    for curve in curve_set:
        for line in curve.Lines:
            # opencv
            points = []
            for point in line.certain_points:
                scaled_x = int(point.position[0] * scale_factor) #+ 400
                scaled_y = int(image.shape[0] - point.position[1] * scale_factor) #- 200
                points.append([scaled_x, scaled_y])
                
            cv2.polylines(image, [np.array(points)], False, (0, 0, 0), thickness=thickness)
   
    image_size = 512
    image = cv2.resize(image, [image_size,]*2)

    for secondary in secondary_line:
        cv2.line(image, (int(secondary[0]), int(secondary[1])), (int(secondary[2]), int(secondary[3])), (0, 0, 0), secondary[4])
    
    return image


polypalm_path = "/home/ra1/Project/ZZY/Diff-Palm/PolyCreases/coeff_test/polynominal"
polypalm_list = [f for f in os.listdir(polypalm_path) if f.endswith('.npz')]
image_path = "/home/ra1/Project/ZZY/Diff-Palm/PolyCreases/coeff_test/image"
image_list = [f for f in os.listdir(image_path) if f.endswith('.png')]
reconstruct_path = "/home/ra1/Project/ZZY/Diff-Palm/PolyCreases/coeff_test/reconstruct"
polypalm_list.sort()
image_list.sort()
for polypalm in polypalm_list:
    npz = np.load(os.path.join(polypalm_path, polypalm), allow_pickle=True)
    reconstruct_img = show(curve_set=npz['curve_set'],
                           secondary_line=npz['secondary_line'])
    img = cv2.imread(os.path.join(image_path, polypalm.split('.')[0]+'.png'))
    diff = np.sum(reconstruct_img - img)
    print(f"The difference between reconstructed image and source image is {diff}")
    cv2.imwrite(os.path.join(reconstruct_path, polypalm.split('.')[0]+'_reconstruct.png'), reconstruct_img)

