import cv2




def imgProcess(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 7)
    return img

def circleFilter(left_circle_batch,right_circle_batch):#两个都不为空
    if left_circle_batch==0 or right_circle_batch==0:
        return left_circle_batch,right_circle_batch
    score=[]
    for left_index,left_circle in enumerate(left_circle_batch):
        for right_index,right_circle in enumerate(right_circle_batch):
            score.append((left_index,right_index,abs(left_circle[1]-right_circle[1])+2*(left_circle[2]-right_circle[2])))
    best=min(score, key=lambda x: x[2])
    left_circle_batch=[left_circle_batch[best[0]]]
    right_circle_batch=[right_circle_batch[best[1]]]
    return left_circle_batch,right_circle_batch


def drawCircle(img,circle_batch,name):
    if circle_batch ==0:
        cv2.putText(img,"No circle!", (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)
        cv2.imshow(name, img)
    else:
        for i,(x,y,r) in enumerate(circle_batch):
            disp=abs(6400.0/disp)
            img = cv2.circle(img, (int(x), int(y)), int(r), (0, 0, 255), 1, 8, 0)
            cv2.putText(img, "x={}, y={}".format(x,y), (20, 20*(i+1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)
        cv2.imshow(name,img)



def detectCircle(img):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 200, param1=300, param2=25, minRadius=0, maxRadius=50)
    if circles is None:
        print('Cannot find circle! ')
        return 0
    else:
        circle_batch=[]
        for circle in circles[0]:
            x,y,r=float(circle[0]),float(circle[1]),float(circle[2])
            circle_batch.append((x,y,r))
            # print('x={}, y={}, r={}'.format(str(x),str(y),str(r)))
        return circle_batch