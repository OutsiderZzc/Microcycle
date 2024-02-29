import cv2
import numpy as np
from matplotlib import pyplot as plt
import Steger
import Skeleton

def process_as_components(bin_img):
    # 获取连通分量的标签、统计信息和质心
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(bin_img))

    # 创建一个彩色图像以便可视化
    output_image = np.zeros((bin_img.shape[0], bin_img.shape[1], 3), np.uint8)

    # 遍历所有的连通分量（跳过背景）
    print(num_labels)
    for i in range(1, num_labels):
        # 获取连通分量的质心
        mask = labels == i
        output_image[mask] = np.random.randint(0, 255, size=(3,), dtype=np.uint8)

        cur_img = np.zeros(bin_img.shape, np.uint8)
        cur_img[mask] = 1

        #steger or skeleton
        if i < 5:
            Skeleton.skeleton(cur_img)

    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), interpolation="nearest")
    plt.axis('off')
    plt.savefig('t.png', bbox_inches='tight', pad_inches=0.0)
    plt.show()

file_path = "./binary_image.png"
bin_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

#process_as_components(bin_img)
#res = Skeleton.skeleton(bin_img)
res = Steger.steger(bin_img)
res[res > 0] = 1
Skeleton.morph_find(bin_img)