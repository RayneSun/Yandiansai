import cv2
from Circle.circle import imgProcess,detectCircle,drawCircle,circleFilter



def cameraInit():
    left_cam = cv2.VideoCapture(2)
    right_cam=cv2.VideoCapture(1)
    if left_cam.isOpened() and right_cam.isOpened():
        print("Init success!")
        return left_cam,right_cam
    elif not left_cam.isOpened():
        print('Cam2 not opened!')
        return -1
    else:
        print('Cam2 not opened!')
        return -1

def process(left_cam,right_cam,j,delay=5):
    _, left_img = left_cam.read()
    __, right_img = right_cam.read()
    left = imgProcess(left_img)
    right = imgProcess(right_img)
    left_circle_batch = detectCircle(left)
    right_circle_batch = detectCircle(right)
    if j % 20 == 19:
        print("过滤前: left_circle_batch={}, right_circle_batch={}".format(left_circle_batch, right_circle_batch))
    left_circle_batch,right_circle_batch =circleFilter(left_circle_batch,right_circle_batch)
    if j % 20 == 19:
        print("过滤后: left_circle_batch={}, right_circle_batch={}".format(left_circle_batch, right_circle_batch))
    else:
        disparity=1
    drawCircle(img=left_img, circle_batch=left_circle_batch, name='Left')
    drawCircle(img=right_img, circle_batch=right_circle_batch, name='Right')
    cv2.waitKey(delay)

def main():
    left_cam,right_cam=cameraInit()
    j=0
    while 1:
        process(left_cam,right_cam,j=j,delay=10)
        j+=1

main()


def test():
    left_img=cv2.imread('./Sample/left_1.jpg')
    right_img=cv2.imread('./Sample/right_1.jpg')
    left = imgProcess(left_img)
    right = imgProcess(right_img)
    left_circle_batch = detectCircle(left)
    right_circle_batch = detectCircle(right)
    left_circle_batch, right_circle_batch = circleFilter(left_circle_batch, right_circle_batch)
    print("过滤前: left_circle_batch={}, right_circle_batch={}".format(left_circle_batch, right_circle_batch))
    drawCircle(img=left_img, circle_batch=left_circle_batch, name='Left')
    drawCircle(img=right_img, circle_batch=right_circle_batch, name='Right')
    cv2.waitKey()

# test()