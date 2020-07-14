import cv2

conf={'save_load':'./test/'}

cap = cv2.VideoCapture(1)
i=105
while 1:
    ret,img = cap.read()
    if ret:
        cv2.imshow('Image', img)
        if cv2.waitKey(1)==ord(' '):
            img=cv2.resize(img,(540,384))
            cv2.imwrite(filename=conf['save_load']+str(i)+'.jpg',img=img)
            i += 1
    else:
        print('camera not found!')
        cv2.waitKey(1000)