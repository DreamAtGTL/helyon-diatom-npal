import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random as rng
import scipy as sp
import scipy.ndimage

rng.seed(12345)

A = os.listdir('./image')
def flood_fill(test_array,h_max=255):
    input_array = np.copy(test_array) 
    el = sp.ndimage.generate_binary_structure(2,2).astype(np.int)
    inside_mask = sp.ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask]=h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)   
    el = sp.ndimage.generate_binary_structure(2,1).astype(np.int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,sp.ndimage.grey_erosion(output_array, size=(3,3), footprint=el))
    return output_array

print(len(A))

img = cv2.imread('./image/29_image.png',0)
pred = cv2.imread('./mask/29_pred.png', 0)
edge = cv2.Canny(img,0,100)
mask = pred.copy()
diatoms = img * mask
diatoms_edge = edge * mask
contours, hierarchy = cv2.findContours(diatoms_edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
out = np.zeros_like(img)
cv2.drawContours(out, contours, -1, 255, 3)

kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))
out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel2)
out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel2)
final_img = out*mask
#plt.imshow(out*mask,cmap='jet'); plt.show()



#Watershed algo
kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]], dtype=np.float32)
hist = list(cv2.calcHist([img],[0],None,[256],[0,256]))
max_occurence = max(hist)
dominant_pixel = (np.argwhere(hist==max_occurence))[0][0]
sub_value = np.abs(img.astype(np.int)-dominant_pixel)
mask1 = np.where(sub_value<15,0,1)
mask1 = (mask1 * 255).astype(np.uint8)
mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel2)
mask1 = mask/255
img_masked = img * mask1
img_masked = diatoms
img_masked = img_masked.astype(np.uint8)
#plt.imshow(img_masked, cmap='jet'); plt.show()
imgLaplacian = cv2.filter2D(img_masked,-1, kernel)
imgResult = img_masked - imgLaplacian
_,bw = cv2.threshold(imgResult,40,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
dist = cv2.distanceTransform(pred*255,cv2.DIST_L2,3)
cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
_, dist = cv2.threshold(dist, 0.25,1.0, cv2.THRESH_BINARY)

dist = (dist*255).astype(np.uint8)
#dist = cv2.morphologyEx(dist, cv2.MORPH_CLOSE, kernel2)
#dist = flood_fill(dist)
kernel1 = np.ones((3,3), dtype=np.uint8)
dist = cv2.dilate(dist,kernel1)
contours, _ = cv2.findContours(dist, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
markers = np.zeros(dist.shape,dtype=np.int32)
for i in range(len(contours)):
    cv2.drawContours(markers, contours, i, (i+1), -1)
#cv2.circle(markers, (5,5), 3, (255,255,255), -1)
imgResult = np.stack((imgResult,)*3, axis=-1)
markers=cv2.watershed(imgResult, markers)
mark = markers.astype(np.uint8)
mark = cv2.bitwise_not(mark)
colors = []
bboxes = []
for i,contour in enumerate(contours):
    bboxes.append(cv2.minAreaRect(contours[i]))
    colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))
dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)

for i in range(markers.shape[0]):
    for j in range(markers.shape[1]):
        index = markers[i,j]
        if index > 0 and index <= len(contours):
            dst[i,j,:] = colors[index-1]

rect2 = img.copy()
rect2 = np.stack((rect2,)*3, axis=-1)
for i in range(len(bboxes)):
    (x,y),(w,h),a = bboxes[i]
    box = cv2.boxPoints(bboxes[i])
    box = np.int0(box)
    cv2.drawContours(rect2,[box],0,(rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)),3)
cv2.imshow('img', rect2); cv2.waitKey(0)
plt.imshow(rect2); plt.show()
