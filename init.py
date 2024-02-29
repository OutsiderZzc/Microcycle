import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim

# pre processing (use for both training and testing!)
def my_PreProc(data):
    assert(len(data.shape) == 4)
    assert (data.shape[1] == 3)  #Use the original images
    #black-white conversion
    train_imgs = rgb2gray(data)
    #my preprocessing:
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs / 255.  #reduce to 0-1 range
    return train_imgs


#convert RGB image in black and white
def rgb2gray(rgb):
    assert (len(rgb.shape) == 4)  #4D arrays
    assert (rgb.shape[1] == 3)
    bn_imgs = rgb[:,0,:,:] * 0.299 + rgb[:,1,:,:] * 0.587 + rgb[:,2,:,:] * 0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]))
    return bn_imgs

#==== histogram equalization
def histo_equalized(imgs):
    assert (len(imgs.shape) == 4)  #4D arrays
    assert (imgs.shape[1] == 1)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i, 0], dtype = np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
    assert (len(imgs.shape) == 4)  #4D arrays
    assert (imgs.shape[1] == 1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, 0] = clahe.apply(np.array(imgs[i, 0], dtype = np.uint8))
    return imgs_equalized


# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape) == 4)  #4D arrays
    assert (imgs.shape[1] == 1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean) / imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


def adjust_gamma(imgs, gamma = 1.0):
    assert (len(imgs.shape) == 4)  #4D arrays
    assert (imgs.shape[1] == 1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs


def get_frames_by_aviFilePath(avi_file_path):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(avi_file_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    frame_list = []
    # Loop through each frame in the video
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            break  # Break the loop if the end of the video is reached

        frame_list.append(frame)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Release the VideoCapture object and close any open windows
    cap.release()
    cv2.destroyAllWindows()

    return frame_list, fps, total_frames


def bgr2rgb(frame_list):
    num_imgs = len(frame_list)

    height, width, channels = frame_list[0].shape

    imgs = np.empty((num_imgs, channels, height, width))

    for i in range(num_imgs):
        frame_rgb = cv2.cvtColor(frame_list[i], cv2.COLOR_BGR2RGB).transpose([2, 0, 1])
        imgs[i] = frame_rgb

    return imgs


def binarize_by_threshold(gray_img, low_threshold=0.5, high_threshold=0.6):
    '''
    二值化灰度图, 灰度位于low_threshold~high_threshold之间像素被置为1, 其余为0
    '''
    bina_img = gray_img.copy()
    bina_img[(gray_img <= high_threshold).astype(bool) & (gray_img >= low_threshold).astype(bool)] = 1

    bina_img[gray_img < low_threshold] = 0
    bina_img[gray_img > high_threshold] = 0

    # bina_img = cv2.dilate(bina_img, np.ones((1,1), np.uint8), iterations=1)
    return bina_img


def seg_by_threshold(img, low_thresh=0.35, high_thresh=0.6, GaussianBlur_kernel_size=(5, 5), GaussianBlur_thresh=0.3,
                     isGaussian=True, isFiltered=True, min_area_threshold=100):
    '''
    该函数包含三个操作, 二值化, 高斯平滑, 小连通分量过滤
    isGaussian, isFiltered 分别为是否执行高斯平滑和小连通分量过滤的Flag
    '''
    # 区间阈值滤出
    bina_img = binarize_by_threshold(img, low_threshold=low_thresh, high_threshold=high_thresh)

    # 高斯平滑
    if isGaussian:
        smoothed_image = cv2.GaussianBlur(bina_img, GaussianBlur_kernel_size, 0)

        smoothed_image[smoothed_image >= GaussianBlur_thresh] = 1
        smoothed_image[smoothed_image < GaussianBlur_thresh] = 0

        bina_img = smoothed_image

    # 去除小连通分量
    if isFiltered:
        # 连通分量标记
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bina_img.astype(np.uint8))

        # 根据面积筛选连通分量
        filtered_labels = [label for label, stat in enumerate(stats) if stat[4] < min_area_threshold]

        # 创建输出图像，仅保留符合条件的连通分量
        filtered_image = bina_img.copy()
        for label in filtered_labels:
            filtered_image[labels == label] = 0

        bina_img = filtered_image

    return bina_img


def find_similar_frames(image_sequence, reference_frame_index, threshold=0.42):
    reference_frame = image_sequence[reference_frame_index]

    similar_frames = []
    similar_frames.append(image_sequence[reference_frame_index])
    for i in range(len(image_sequence)):
        if i != reference_frame_index:
            # 计算SSIM
            ssim_index = ssim(reference_frame, image_sequence[i])
            # 如果相似性超过阈值，则认为是相似帧
            if ssim_index > threshold:
                similar_frames.append(image_sequence[i])

    return similar_frames

def ssim(img1, img2):
    # 计算SSIM
    ssim_index, _ = compare_ssim(img1, img2, full=True)
    return ssim_index

def calculate_frame_difference(frame1, frame2):
    # 结构相似性指标(SSIM)返回值范围为[-1,1]，值越大表示两帧越相似
    ssim_diff = 1 - ssim(frame1, frame2)
    return ssim_diff

def calculate_video_shakiness(frames):
    shakiness = []

    # 计算每对相邻帧的差异
    for i in range(len(frames) - 1):
        diff = calculate_frame_difference(frames[i], frames[i + 1])
        shakiness.append(diff)

    return shakiness

def find_smoothest_frames(frames):
    min_shakiness = float('inf')
    smoothest_frames_index = None

    shakiness = calculate_video_shakiness(frames)
    # 寻找抖动最小的连续十帧
    for i in range(len(shakiness) - 9):
        shakiness_sum = sum(shakiness[i : i + 10])
        if shakiness_sum < min_shakiness:
            min_shakiness = shakiness_sum
            smoothest_frames_index = i

    print(smoothest_frames_index)
    return frames[smoothest_frames_index : smoothest_frames_index + 10]

def calculate_flow_in_vessel(prev_frame, current_frame, vessel_mask):
    """
    计算并返回给定血管掩模区域内的平均光流速度。
    """
    # 应用血管掩模
    prev_vessel = cv2.bitwise_and(prev_frame, prev_frame, mask=vessel_mask)
    current_vessel = cv2.bitwise_and(current_frame, current_frame, mask=vessel_mask)

    # 计算光流
    flow = cv2.calcOpticalFlowFarneback(prev_vessel, current_vessel, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 计算平均流速
    avg_speed = np.mean(mag)
    return avg_speed


def calculate_speed_for_all_vessels(binary_frames, vessel_mask):
    """
    对二值化图像序列中的所有血管计算血液流速。
    """
    # 假设第一个帧用于血管分割
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(vessel_mask))

    # 对于每个血管标签，计算流速
    vessel_speeds = {}
    for i in range(1, num_labels):  # 标签0是背景
        vessel_mask = np.uint8(labels == i)
        speeds = []
        for j in range(1, len(binary_frames)):
            prev_frame = binary_frames[j - 1].astype(np.float32)
            current_frame = binary_frames[j].astype(np.float32)
            speed = calculate_flow_in_vessel(prev_frame, current_frame, vessel_mask)
            speeds.append(speed)
        vessel_speeds[i] = np.mean(speeds)

    return vessel_speeds

def draw_img(binary_image, isLabeled=False):
    # 获取连通分量的标签、统计信息和质心
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(binary_image))

    # 创建一个彩色图像以便可视化
    output_image = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), np.uint8)

    # 遍历所有的连通分量（跳过背景）
    for i in range(1, num_labels):
        # 获取连通分量的质心
        mask = labels == i
        output_image[mask] = np.random.randint(0, 255, size=(3,), dtype=np.uint8)

        if isLabeled:
            centroid = centroids[i]
            # 在质心位置上绘制标号
            text = str(i)
            position = (int(centroid[0]), int(centroid[1]))
            cv2.putText(output_image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 使用matplotlib显示图像
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), interpolation="nearest")
    plt.axis('off')
    plt.title("Connected Components with Labels" if isLabeled == True else "Connected Components without Labels")
    plt.savefig('t.png', bbox_inches='tight', pad_inches=0.0)
    plt.show()

def remove_circular_components(binary_image, circularity_threshold=0.33):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(binary_image))
    result_image = np.zeros_like(binary_image)

    for label in range(1, num_labels):  # 忽略背景
        component_mask = (labels == label).astype("uint8") * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue  # 如果没有找到轮廓，跳过此连通分量

        area = cv2.contourArea(contours[0])
        perimeter = cv2.arcLength(contours[0], True)

        if perimeter == 0:
            continue  # 防止除以零

        circularity = 4 * np.pi * area / (perimeter ** 2)

        if circularity <= circularity_threshold:
            # 如果连通分量不是近似圆形，将其保留在结果图像中
            result_image += component_mask

    return result_image