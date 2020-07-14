from builtins import *
import cv2                      #导入 Opencv
import os
import numpy as np
import matplotlib.pyplot as plt


output_dir = 'output2'	        #设置输出文件夹，若不存在则创建
if not os.path.exists(output_dir):
	os.mkdir(output_dir)

img_file = 'C:/Users/25787/Desktop/tupian/left_59.jpg'     #读取图片
img = cv2.imread(img_file)

type(img)                       #读入图片后得到ndarray 对象
img.shape                       #ndarray的三个维度分别是图片的：高，宽，通道

# pyplot.imgshow 在显示图片时是按照RGB通道顺序显示，cv2则相反
# 需要通过 np.flip(img,axis = 2) 调整3个通道的顺序（若不调整图片颜色失真）
plt.imshow(np.flip(img,axis = 2))
plt.axis('off')
plt.show()                      #图1

plt.imshow(img)
plt.axis('off')                 #不显示坐标
plt.show()                      #图2

#输出并保存图片
output_image = os.path.join(output_dir,'image.png')
cv2.imwrite(output_image,img)

import cv2
import numpy as np
import matplotlib.pyplot as plt


def calcAndDrawHist(image, color):
	hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
	minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
	histImg = np.zeros([256, 256, 3], np.uint8)
	hpt = int(0.9 * 256);

	for h in range(256):
		intensity = int(hist[h] * hpt / maxVal)
		cv2.line(histImg, (h, 256), (h, 256 - intensity), color)
	return histImg


if __name__ == '__main__':
	original_img = cv2.imread(img_file)
	img = cv2.resize(original_img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_CUBIC)
	b, g, r = cv2.split(img)

	histImgB = calcAndDrawHist(b, [255, 0, 0])
	histImgG = calcAndDrawHist(g, [0, 255, 0])
	histImgR = calcAndDrawHist(r, [0, 0, 255])

	cv2.imshow("histImgB", histImgB)
	cv2.imshow("histImgG", histImgG)
	cv2.imshow("histImgR", histImgR)
	cv2.imshow("Img", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



pic_file = 'C:/Users/25787/Desktop/tupian/left_59.jpg'

img_bgr = cv2.imread(pic_file, cv2.IMREAD_COLOR) #OpenCV读取颜色顺序：BRG
img_b = img_bgr[..., 0]
img_g = img_bgr[..., 1]
img_r = img_bgr[..., 2]
fig = plt.gcf()                                  #图片详细信息


fig = plt.gcf()                                  #分通道显示图片
fig.set_size_inches(10, 15)

plt.subplot(221)
plt.imshow(np.flip(img_bgr, axis=2))             #展平图像数组并显示
plt.axis('off')
plt.title('Image')

plt.subplot(222)
plt.imshow(img_r, cmap='gray')
plt.axis('off')
plt.title('R')

plt.subplot(223)
plt.imshow(img_g, cmap='gray')
plt.axis('off')
plt.title('G')

plt.subplot(224)
plt.imshow(img_b, cmap='gray')
plt.axis('off')
plt.title('B')

plt.show()

#HSV颜色空间
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_h = img_hsv[..., 0]
img_s = img_hsv[..., 1]
img_v = img_hsv[..., 2]

fig = plt.gcf()                      # 分通道显示图片
fig.set_size_inches(10, 15)

plt.subplot(221)
plt.imshow(img_hsv)
plt.axis('off')
plt.title('HSV')

plt.subplot(222)
plt.imshow(img_h, cmap='gray')
plt.axis('off')
plt.title('H')

plt.subplot(223)
plt.imshow(img_s, cmap='gray')
plt.axis('off')
plt.title('S')

plt.subplot(224)
plt.imshow(img_v, cmap='gray')
plt.axis('off')
plt.title('V')

plt.show()

img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
fig = plt.gcf()
fig.set_size_inches(5, 7.5)

plt.imshow(img_gray, cmap='gray')
plt.axis('off')
plt.title('Gray')
plt.show()

"""
	cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) → hist
	参数说明
	images:   图片列表
	channels: 需要计算直方图的通道。[0]表示计算通道0的直方图，[0,1,2]表示计算通道0,1,2所表示颜色的直方图
	mask:     蒙版，只计算值>0的位置上像素的颜色直方图，取None表示无蒙版
	histSize: 每个维度上直方图的大小，[8]表示把通道0的颜色取值等分为8份后计算直方图
	ranges:   每个维度的取值范围，[lower0, upper0, lower1, upper1, ...]，lower可以取到，upper无法取到
	hist:     保存结果的ndarray对象
	accumulate: 是否累积，如果设置了这个值，hist不会被清零，直方图结果直接累积到hist中
"""

img_gray_hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])

plt.plot(img_gray_hist)
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')
plt.show()

# 按R、G、B三个通道分别计算颜色直方图
b_hist = cv2.calcHist([img_bgr], [0], None, [256], [0, 256])
g_hist = cv2.calcHist([img_bgr], [1], None, [256], [0, 256])
r_hist = cv2.calcHist([img_bgr], [2], None, [256], [0, 256])

# 显示3个通道的颜色直方图
plt.plot(b_hist, label='B', color='blue')
plt.plot(g_hist, label='G', color='green')
plt.plot(r_hist, label='R', color='red')
plt.legend(loc='best')
plt.xlim([0, 256])
plt.show()

# 显示3个通道的颜色直方图
plt.plot(b_hist, label='B', color='blue')
plt.plot(g_hist, label='G', color='green')
plt.plot(r_hist, label='R', color='red')
plt.legend(loc='best')
plt.xlim([0, 256])
plt.show()
