import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import math
import torch
from xml.etree import ElementTree
import shapely
import shapely.geometry





def polygon_inter_union_cpu(boxes1, boxes2):
    """
        Reference: https://github.com/ming71/yolov3-polygon/blob/master/utils/utils.py ;
        iou computation (polygon) with cpu;
        Boxes have shape nx8 and Anchors have mx8;
        Return intersection and union of boxes[i, :] and anchors[j, :] with shape of (n, m).
    """
    
    boxes1 = torch.from_numpy(boxes1)
    boxes2 = torch.from_numpy(boxes2)
    n, m = boxes1.shape[0], boxes2.shape[0]
    inter = torch.zeros((n, m))
    union = torch.zeros((n, m))
    for i in range(n):
        polygon1 = shapely.geometry.Polygon(boxes1[i, :].view(4,2)).convex_hull
        for j in range(m):
            polygon2 = shapely.geometry.Polygon(boxes2[j, :].view(4,2)).convex_hull
            if polygon1.intersects(polygon2):
                try:
                    inter[i, j] = polygon1.intersection(polygon2).area
                    union[i, j] = polygon1.union(polygon2).area
                except shapely.geos.TopologicalError:
                    print('shapely.geos.TopologicalError occured')
    return inter, union

def rotate_box_iou(boxes1, boxes2, GIoU=False, DIoU=False, CIoU=False, eps=1e-7, device="cpu"):
    """
        Compute iou of rotated boxes via cpu or cuda;
        For cuda code, please refer to files in ./iou_cuda
        Returns the IoU of shape (n, m) between boxes1 and boxes2. boxes1 is nx6, boxes2 is mx6
    """

    boxes1_xyxyxyxy = xywhrm2xyxyxyxy(boxes1)
    boxes2_xyxyxyxy = xywhrm2xyxyxyxy(boxes2)
    return polygon_box_iou(boxes1_xyxyxyxy, boxes2_xyxyxyxy, GIoU, DIoU, CIoU, eps, device)  # IoU

def xywhrm2xyxyxyxy(xywhrm):
    """
        xywhrm : shape (N, 6)
        Transform x,y,w,h,re,im to x1,y1,x2,y2,x3,y3,x4,y4
        Suitable for both pixel-level and normalized
    """
    is_array = isinstance(xywhrm, np.ndarray)
    if is_array:
        xywhrm = torch.from_numpy(xywhrm)
        
    x0, x1, y0, y1 = -xywhrm[:, 2:3]/2, xywhrm[:, 2:3]/2, -xywhrm[:, 3:4]/2, xywhrm[:, 3:4]/2
    xyxyxyxy = torch.cat((x0, y0, x1, y0, x1, y1, x0, y1), dim=-1).view(-1, 4, 2).contiguous()
    R = torch.zeros((xyxyxyxy.shape[0], 2, 2), dtype=xyxyxyxy.dtype, device=xyxyxyxy.device)
    R[:, 0, 0], R[:, 1, 1] = xywhrm[:, 4], xywhrm[:, 4]
    R[:, 0, 1], R[:, 1, 0] = xywhrm[:, 5], -xywhrm[:, 5]
    
    xyxyxyxy = torch.matmul(xyxyxyxy, R).view(-1, 8).contiguous()+xywhrm[:, [0, 1, 0, 1, 0, 1, 0, 1]]
    return xyxyxyxy.cpu().numpy() if is_array else xyxyxyxy


def get_roundness(area, perimeter):
    return 4 * np.pi * area / perimeter**2



image_path = 'images/'
seg_mask_path = 'labels/'
rbb_labels_path = 'rbb_labels'
original_rbb_labels = 'NPAL_t28_Gly_Temoin_repG_11-12-20/VOC/labels'
original_seg_labels = 'npal_all/val_labels'

thumbnail_save_path = 'extracted_thumbnails'
os.makedirs(thumbnail_save_path, exist_ok=True)

img_list = sorted(os.listdir(image_path))

h_error = []
w_error = []
h_mean = []
w_mean = []
area_error = []
perim_error = []
roundness_error = []
area_mean = []
perim_mean = []
roundness_mean = []

for img_ in img_list:
    img_name = os.path.splitext(img_)[0]
    img = cv2.imread(os.path.join(image_path, img_), -1)
    img = cv2.resize(img,(2040, 1536))

    #Segmentation
    mask_ = img_.replace('image', 'pred')
    mask = cv2.imread(os.path.join(seg_mask_path, mask_), -1)
    mask = cv2.resize(mask, (2040,1536))
    new_img = img * cv2.merge((mask,mask,mask))

    #Hand labelled segmentation
    file_name = img_name[:-6]
    hand_labelled_mask_path = os.path.join(original_seg_labels, file_name+'.jpg')
    gt_mask = cv2.imread(hand_labelled_mask_path, -1)
    gt_mask = cv2.resize(gt_mask, (2040,1536))
    gt_mask[np.where(gt_mask>1)] = 0
    gt_mask = 1-gt_mask
    #gt_mask = cv2.merge((gt_mask,gt_mask,gt_mask))

    #Hand labelled RBB
    hand_label_file_path = os.path.join(original_rbb_labels, file_name+'.xml')
    rbb_label = open(hand_label_file_path)
    tree = ElementTree.parse(rbb_label)
    root = tree.getroot()
    cx = list()
    cy = list()
    hei = list()
    wid = list()
    angle = list()

    for box in root.findall('object'):
        if box.find('name').text == "Diatom":
            for rbox in box.findall('robndbox'):
                cx.append(int(float(rbox.find('cx').text)))
                cy.append(int(float(rbox.find('cy').text)))
                hei.append(int(float(rbox.find('h').text)))
                wid.append(int(float(rbox.find('w').text)))
                angle.append(float(rbox.find('angle').text))
    
    gt_boxes = np.zeros((len(cx), 8))
    gt_height = np.zeros((len(cx), 1))
    gt_width = np.zeros((len(cx), 1))
    gt_area = np.zeros((len(cx), 1))
    gt_perim = np.zeros((len(cx), 1))
    gt_roundness = np.zeros((len(cx), 1))

    for i in range(len(cx)):

        x,y,w,h,a = cx[i], cy[i], hei[i], wid[i], int(angle[i] * 180/np.pi)

        if a > 90:
            a = a-180
        rect = (x,y), (w,h), a+90
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        height = round(h)
        width = round(w)
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_img = cv2.warpPerspective(new_img, M, (width, height))
        warped_mask = cv2.warpPerspective(gt_mask, M, (width, height))

        

        #Hand label morph params
        contours, hierarchy = cv2.findContours(image=warped_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]  
            M = cv2.moments(cnt,True)
            if M['m00'] == 0.0:
                break

            P = cv2.arcLength(cnt,True)
            cnt_area = cv2.contourArea(cnt)
            roundness = get_roundness(cnt_area, P)

            #cx = M['m10']/M['m00']
            #cy = M['m01']/M['m00']
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
                angle_ = 0.5 * math.atan2(2*cov_xy,(cov_xx-cov_yy));
                r1=2*math.sqrt(lambda1)
                r2=2*math.sqrt(lambda2)
                area=math.pi*r1*r2
                ellipsicity=M['m00']/area
            else:
                ellipsicity = 0




        
            height = max(length, thickness)
            width = min(length, thickness)
            gt_boxes[i,:] = box.ravel()
            gt_height[i,:] = height
            gt_width[i,:] = width
            gt_area[i,:] = cnt_area
            gt_perim[i,:] = P
            gt_roundness[i,:] = roundness


    #Deep RBB Label 
    rbb_label_file = open(os.path.join(rbb_labels_path, img_name+'.txt'), 'r')
    lines = rbb_label_file.readlines()
    dl_boxes = np.zeros((0, 8))
    dl_height = np.zeros((0, 1))
    dl_width = np.zeros((0, 1))
    dl_area = np.zeros((0,1))
    dl_perimeter = np.zeros((0,1))
    dl_roundness = np.zeros((0,1))
    #xywhrm2xyxyxyxy(xywhrm)
    count = 0
    for idx, line in enumerate(lines):
        _, cx, cy, w, h, cos_th, sin_th = line.split(' ')
        cx = float(cx) * img.shape[1]
        cy = float(cy) * img.shape[0]
        w = float(w) * img.shape[1]
        h = float(h) * img.shape[0]
        cos_th = float(cos_th) #* img.shape[0]
        sin_th = float(sin_th) #* img.shape[1]
        #dl_boxes[idx, :] = cx, cy, w, h, cos_th, sin_th


        theta = np.arctan2(cos_th, sin_th) * 180 / np.pi
        rect = (cx,cy), (h,w), 180 - theta #+ 90
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        '''
        plt.imshow(img)
        plt.plot(
            np.append(box[:,0],box[0,0]),
            np.append(box[:,1],box[0,1])
        )
        plt.show()
        '''

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
        #plt.imshow(warped_mask); plt.show()

        #Some morphology based filtering
        contours, hierarchy = cv2.findContours(image=warped_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]  
            M = cv2.moments(cnt,True)
            if M['m00'] == 0.0:
                break
            P = cv2.arcLength(cnt,True)
            cnt_area = cv2.contourArea(cnt)
            roundness = get_roundness(cnt_area, P)

            #cx = M['m10']/M['m00']
            #cy = M['m01']/M['m00']
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
                angle_ = 0.5 * math.atan2(2*cov_xy,(cov_xx-cov_yy));
                r1=2*math.sqrt(lambda1)
                r2=2*math.sqrt(lambda2)
                area=math.pi*r1*r2
                ellipsicity=M['m00']/area
            else:
                ellipsicity = 0

            if ellipsicity > 0.99:
                height = max(length,thickness)
                width = min(length,thickness)

                dl_boxes = np.vstack((dl_boxes, box.ravel()))
                dl_height = np.vstack((dl_height, height))
                dl_width = np.vstack((dl_width, width))
                dl_area = np.vstack((dl_area, cnt_area))
                dl_perimeter = np.vstack((dl_perimeter, P))
                dl_roundness = np.vstack((dl_roundness, roundness))
                thumbnail_name = img_name + '_' + str(count) +'.png'
                if warped_img.shape[1] > warped_img.shape[0]:
                    warped_img = cv2.rotate(warped_img, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(os.path.join(thumbnail_save_path, thumbnail_name), warped_img)
                count += 1

    
    #Calculate the iou
    inter, union = polygon_inter_union_cpu(dl_boxes, gt_boxes)
    union += 1e-20
    iou = inter/union
    iou = np.array(iou)
    #Get the gt box number corresponding to the dl box
    if iou.shape[1] != 0 and iou.shape[0]!= 0:
        pred_box = np.argmax(iou, axis=1)
        #Reorder the gt box such that it matches the dl box order
        rearranged_gt_boxes = gt_boxes[pred_box]
        rearranged_gt_h = gt_height[pred_box]
        rearranged_gt_w = gt_width[pred_box]
        rearranged_gt_area = gt_area[pred_box]
        rearranged_gt_perim = gt_perim[pred_box]
        rearranged_gt_roundness = gt_roundness[pred_box]

        #Check for the boxes whose iou > threshold
        is_match = iou[np.arange(dl_boxes.shape[0]),pred_box] > 0.50
        #Select the dl and gt boxes whose iou > threshold
        selected_dl_boxes = dl_boxes[is_match, :]
        selected_gt_boxes = rearranged_gt_boxes[is_match, :]

        #Height and width
        selected_dl_h = dl_height[is_match, :]
        selected_dl_w = dl_width[is_match, :]
        selected_gt_h = rearranged_gt_h[is_match, :]
        selected_gt_w = rearranged_gt_w[is_match, :]  

        #Area
        selected_dl_area = dl_area[is_match, :]
        selected_gt_area = rearranged_gt_area[is_match, :]

        #Perimeter
        selected_dl_perim = dl_perimeter[is_match, :]
        selected_gt_perim = rearranged_gt_perim[is_match, :]

        #Roundness
        selected_dl_roundness = dl_roundness[is_match, :]
        selected_gt_roundness = rearranged_gt_roundness[is_match, :]       

        error_height = abs(selected_dl_h - selected_gt_h)
        error_width = abs(selected_dl_w - selected_gt_w)
        error_area = abs(selected_dl_area - selected_gt_area)
        error_perim = abs(selected_dl_perim - selected_gt_perim)
        error_roundness = abs(selected_dl_roundness - selected_gt_roundness)

        h_error.extend(error_height)
        w_error.extend(error_width)
        h_mean.extend(selected_gt_h)
        w_mean.extend(selected_gt_w)
        area_error.extend(error_area)
        perim_error.extend(error_perim)
        roundness_error.extend(error_roundness)

        area_mean.extend(selected_gt_area)
        perim_mean.extend(selected_gt_perim)
        roundness_mean.extend(selected_gt_roundness)


print("Error for height and width")
print(np.mean(h_error))
print(np.mean(w_error))

print("Height and width")
print(np.mean(h_mean))
print(np.std(h_mean))
print(np.mean(w_mean))
print(np.std(w_mean))

print("Area error")
print(np.mean(area_error))
print("Area")
print(np.mean(area_mean))
print(np.std(area_mean))

print("Perimeter error")
print(np.mean(perim_error))
print("Perimeter")
print(np.mean(perim_mean))
print(np.std(perim_mean))

print("Roundness error")
print(np.mean(roundness_error))
print("Roundness")
print(np.mean(roundness_mean))
print(np.std(roundness_mean))




#my_data = {'Length': np.array(h_error).flatten(), 'Width': np.array(w_error).flatten()}
my_data = {'Length': np.array(h_error).flatten()}
fig, ax = plt.subplots()
ax.boxplot(my_data.values(), showfliers=False)
ax.set_xticklabels(my_data.keys())
plt.title("Mean: %f, Std. deviation: %f, Mean error: %.2f%%" %(np.mean(h_mean), np.std(h_mean), np.mean(h_error)/np.mean(h_mean)*100), fontsize=10)
plt.show()

my_data = {'Width': np.array(w_error).flatten()}
fig, ax = plt.subplots()
ax.boxplot(my_data.values(), showfliers=False)
ax.set_xticklabels(my_data.keys())
plt.title("Mean: %f, Std. deviation: %f, Mean error: %.2f%%" %(np.mean(w_mean), np.std(w_mean),np.mean(w_error)/np.mean(w_mean)*100), fontsize=10)
plt.show()
        

my_data = {'Area': np.array(area_error).flatten()}
fig, ax = plt.subplots()
ax.boxplot(my_data.values(), showfliers=False)
ax.set_xticklabels(my_data.keys())
plt.title("Mean: %f, Std. deviation: %f, Mean error: %.2f%%" %(np.mean(area_mean), np.std(area_mean), np.mean(area_error)/np.mean(area_mean)*100), fontsize=10)
plt.show()
        
my_data = {'Perimeter': np.array(perim_error).flatten()}
fig, ax = plt.subplots()
ax.boxplot(my_data.values(), showfliers=False)
ax.set_xticklabels(my_data.keys())
plt.title("Mean: %f, Std. deviation: %f, Mean error: %.2f%%" %(np.mean(perim_mean), np.std(perim_mean), np.mean(perim_error)/np.mean(perim_mean)*100), fontsize=10)
plt.show()

my_data = {'Roundness': np.array(roundness_error).flatten()}
fig, ax = plt.subplots()
ax.boxplot(my_data.values(), showfliers=False)
ax.set_xticklabels(my_data.keys())
plt.title("Mean: %f, Std. deviation: %f, Mean error: %.2f%%" %(np.mean(roundness_mean), np.std(roundness_mean), np.mean(roundness_error)/np.mean(roundness_mean)*100), fontsize=10)
plt.show()
        
    

    

    
   
