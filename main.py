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

avi_file_path = "./20231118190703.avi"

frame_list, fps, total_frames = init.get_frames_by_aviFilePath(avi_file_path)
# BGR通道 转化为 RGB通道
imgs = init.bgr2rgb(frame_list)
# 预处理
processed_imgs = init.my_PreProc(imgs)
ans_img = init.seg_by_threshold(processed_imgs[0][0], isGaussian=False, isFiltered=False)

segmented_imgs = []
for i in tqdm(range(len(processed_imgs))):
    segmented_imgs.append(init.seg_by_threshold(processed_imgs[i][0], isGaussian=False, isFiltered=False))

test_imgs = segmented_imgs[15 : 25].copy()
tmp = test_imgs[0].copy()

for j in range(1, 10):
    tmp += test_imgs[j]

thresh = 2
accu_imgs = tmp.copy()
accu_imgs[tmp >= thresh] = 1
accu_imgs[tmp < thresh] = 0

print(accu_imgs.shape)

G_kernel_size = (5, 5)
G_thresh = 0.3
smoothed_image = cv2.GaussianBlur(accu_imgs, G_kernel_size, 0)
smoothed_image[smoothed_image >= G_thresh] = 1
smoothed_image[smoothed_image < G_thresh] = 0
bina_img = smoothed_image

min_area_threshold = 100
# 连通分量标记
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bina_img.astype(np.uint8))
# 根据面积筛选连通分量
filtered_labels = [label for label, stat in enumerate(stats) if stat[4] < min_area_threshold]
# 创建输出图像，仅保留符合条件的连通分量
filtered_image = bina_img.copy()
for label in filtered_labels:
    filtered_image[labels == label] = 0
bina_img = filtered_image

plt.imshow(bina_img, cmap="gray")
plt.axis('off')
plt.show()
cv2.imwrite("binary_image.png", bina_img)


# init.draw_img(result_image)
# base = 1E13
#
# vessel_speeds = init.calculate_speed_for_all_vessels(test_imgs, result_image)
# velocity = []
#
# for vessel_id, speed in vessel_speeds.items():
#     velocity.append(speed * base)
#
# plt.plot(velocity)
# plt.title("Blood flow")
# plt.show()