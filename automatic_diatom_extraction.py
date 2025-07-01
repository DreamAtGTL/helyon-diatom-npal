import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import math

image_path = 'images/'
seg_mask_path = 'labels/'
rbb_labels_path = 'rbb_labels/'
original_rbb_labels = 'NPAL_t28_Gly_Temoin_repG_11-12-20/VOC/labels'

thumbnail_save_path = 'extracted_thumbnails'
os.makedirs(thumbnail_save_path, exist_ok=True)

img_list = sorted(os.listdir(image_path))

for img_ in img_list:
    img_name = os.path.splitext(img_)[0]
    img = cv2.imread(os.path.join(image_path, img_), -1)
    img = cv2.resize(img,(2040, 1536))

    #Segmentation
    mask_ = img_.replace('image', 'pred')
    mask = cv2.imread(os.path.join(seg_mask_path, mask_), -1)
    mask = cv2.resize(mask, (2040,1536))
    new_img = img #* cv2.merge((mask,mask,mask))
    

    #RBB Label
    rbb_label_file = open(os.path.join(rbb_labels_path, img_name+'.txt'), 'r')
    lines = rbb_label_file.readlines()
    count = 0
    for line in lines:
        _, cx, cy, w, h, cos_th, sin_th = line.split(' ')
        cx = float(cx) * img.shape[1]
        cy = float(cy) * img.shape[0]
        w = float(w) * img.shape[1]
        h = float(h) * img.shape[0]
        cos_th = float(cos_th) #* img.shape[0]
        sin_th = float(sin_th) #* img.shape[1]


        theta = np.arctan2(cos_th, sin_th) * 180 / np.pi
        rect = (cx,cy), (h,w), 180 - theta #+ 90
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        #width = round(rect[1][0])
        #height = round(rect[1][1])
        width = round(h)
        height = round(w)
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_img = cv2.warpPerspective(new_img, M, (width, height))
        warped_mask = cv2.warpPerspective(mask, M, (width, height))



        '''
        rotated = img.copy()
        rotated = ndimage.rotate(img, -theta)
        result = rotated.copy()
 
        plt.imshow(img)
        plt.plot(
            np.append(box[:,0],box[0,0]),
            np.append(box[:,1],box[0,1])
        )
        plt.show()
        '''


        #Some morphology based filtering
        contours, hierarchy = cv2.findContours(image=warped_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]  
            M = cv2.moments(cnt,True)
            if M['m00'] == 0.0:
                break

            cx = M['m10']/M['m00']
            cy = M['m01']/M['m00']
            cov_xx = M['mu20']/M['m00']
            cov_xy = M['mu11']/M['m00']
            cov_yy = M['mu02']/M['m00']
            T = cov_xx + cov_yy;
            D = cov_xx*cov_yy - cov_xy*cov_xy;
            delta = T*T/4 - D;
            assert (delta > -1e-8);
            if abs(delta) <= 1e-8:
                delta = 0;
            lambda1 = T/2 + math.sqrt(delta);
            lambda2 = T/2 - math.sqrt(delta);
            if lambda1>1e-12 and lambda2>1e-12:
                length = 4*math.sqrt(lambda1);
                thickness = 4*math.sqrt(lambda2);
                angle = 0.5 * math.atan2(2*cov_xy,(cov_xx-cov_yy));
                r1=2*math.sqrt(lambda1)
                r2=2*math.sqrt(lambda2)
                area=math.pi*r1*r2
                ellipsicity=M['m00']/area
            else:
                ellipsicity = 0

            if ellipsicity > 0.99:
                thumbnail_name = img_name + '_' + str(count) +'.png'
                if warped_img.shape[1] > warped_img.shape[0]:
                    warped_img = cv2.rotate(warped_img, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(os.path.join(thumbnail_save_path, thumbnail_name), warped_img)
                count += 1

    
   
