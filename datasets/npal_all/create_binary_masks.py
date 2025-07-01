import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

folder_name = 'val_labels'
img_list = os.listdir(folder_name)

for img_name in img_list:
    img_path = os.path.join(folder_name, img_name)
    img = cv2.imread(img_path, -1)
    new_img = img.copy()
    new_img[np.where(img==1)] = 0
    new_img[np.where(img!=1)] = 1
    new_img[np.where(new_img>0)] = 1
    #plt.imshow(new_img); plt.show()
    cv2.imwrite(img_path, new_img.astype(np.uint8))
