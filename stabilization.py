import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
import init
import Steger

avi_file_path = "D:/BaiduNetdiskDownload/2023LPS鼠/大鼠4/2024010104-1h/20240101144520.avi"
frame_list, fps, total_frames = init.get_frames_by_aviFilePath(avi_file_path)
# BGR通道 转化为 RGB通道
imgs = init.bgr2rgb(frame_list)
# 预处理
processed_imgs = init.my_PreProc(imgs)
segmented_imgs = []
for i in tqdm(range(len(processed_imgs))):
    segmented_imgs.append(init.seg_by_threshold(processed_imgs[i][0], isGaussian=True, isFiltered=True, min_area_threshold=30))

stable_imgs = init.find_smoothest_frames(segmented_imgs)

tmp = segmented_imgs[0].copy()
for j in range(1, 10):
    tmp += segmented_imgs[j]

thresh = 5
accu_imgs = tmp.copy()
accu_imgs[tmp >= thresh] = 1
accu_imgs[tmp < thresh] = 0

fig, ax = plt.subplots()
plt.imshow(accu_imgs, cmap='gray')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
plt.axis('off')
plt.show()

tmp = stable_imgs[0].copy()
for j in range(1, 10):
    tmp += stable_imgs[j]

thresh = 5
accu_imgs = tmp.copy()
accu_imgs[tmp >= thresh] = 1
accu_imgs[tmp < thresh] = 0

fig, ax = plt.subplots()
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
plt.imshow(accu_imgs, cmap='gray')
plt.axis('off')
plt.show()

