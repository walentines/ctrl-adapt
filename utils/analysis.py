

import os
import cv2
import numpy as np
from CityscapesDataset import CityscapesDataset

segmentation_path= "/bigdata/userhome/andrei.tarca/shared/MIRPR-proiectCONTROL-NET/dataset_cityscapes_full/gtFine/train"
depth_path ="/bigdata/userhome/andrei.tarca/shared/MIRPR-proiectCONTROL-NET/dataset_cityscapes_full/crestereo_depth/train"
image_path = "/bigdata/userhome/andrei.tarca/shared/MIRPR-proiectCONTROL-NET/dataset_cityscapes_full/leftImg8bit/train"
# depths = sorted(os.listdir(depths_path))

dataset = CityscapesDataset(segmentation_path, depth_path, image_path)

print(len(dataset))

print(dataset[0])
# dataset[0]



# import os
# import cv2
# import numpy as np

# segmentation_path= "/bigdata/userhome/andrei.tarca/shared/MIRPR-proiectCONTROL-NET/dataset_cityscapes_full/gtFine/train"
# depth_path ="/bigdata/userhome/andrei.tarca/shared/MIRPR-proiectCONTROL-NET/dataset_cityscapes_full/crestereo_depth/train"
# image_path = "/bigdata/userhome/andrei.tarca/shared/MIRPR-proiectCONTROL-NET/dataset_cityscapes_full/leftImg8bit/train"
# # depths = sorted(os.listdir(depths_path))

# depths = []

# for dirpath, _, filenames in os.walk(segmentation_path):
#     for file in filenames:
#         if file.endswith("_gtFine_labelIds.png"):
#             depths.append(os.path.join(dirpath, file))

# depths = sorted(depths)

# maxim_gl = 0
# minim_gl = 999

# for masca in depths:
#     # masca_depth = np.load(masca)
#     # print(masca)
#     imagine = cv2.imread(masca, -1)
#     maxim = np.max(imagine)
#     minim = np.min(imagine)
#     if minim < minim_gl:
#         minim_gl = minim
#     if maxim > maxim_gl:
#         maxim_gl = maxim

# print(f"Maxim: {maxim_gl}")
# print(f"Minim: {minim_gl}")
