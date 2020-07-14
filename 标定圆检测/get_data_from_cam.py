import cv2

conf={'save_load':'./Sample/'}

left = cv2.VideoCapture(2)
right=cv2.VideoCapture(1)
i=1
while 1:
    __ret,left_img = left.read()
    _ret,right_img = right.read()
    if __ret and _ret:


        cv2.imshow('leftImage', left_img)
        cv2.imshow('ringtImage', right_img)
        if i == 1:
            cv2.waitKey(3000)
            print('开始采集')
        cv2.waitKey(100)
        cv2.imwrite(filename=conf['save_load']+'left_'+str(i)+'.jpg',img=left_img)
        cv2.imwrite(filename=conf['save_load']+'right_'+str(i)+'.jpg',img=right_img)
        i += 1
    else:
        if not __ret:
            print('left camera not found!')
        if not _ret:
            print('right camera not found!')
        cv2.waitKey(100)