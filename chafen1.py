import os

import cv2
import numpy as np

video_path = 'ele_97a3d855ae65bee568373e84555ac61e.ts'

# ele_d9a990c1c407d83442232971b58fab5d 先忽略

# cap = cv2.VideoCapture(os.path.join(r'D:\aliyun\video', video_path))
cap = cv2.VideoCapture(os.path.join(r'/data/gulingrui/code/mclz/data/video', video_path))

# 获取三个连续的帧
frame1 = cap.read()[1][:, :, 0]
frame2 = cap.read()[1][:, :, 0]
frame3 = cap.read()[1][:, :, 0]

kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))

while True:
    # 计算帧间差异
    diff1 = cv2.absdiff(frame1, frame2)
    diff2 = cv2.absdiff(frame2, frame3)

    # 对差异进行二值化处理
    diff = cv2.bitwise_and(diff1, diff2)
    _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # 对二值化后的图像进行膨胀, 并且使用原型的kernel
    diff = cv2.dilate(diff, kernel)

    # 将diff转成unit8
    diff = diff.astype(np.uint8)
    print(diff.shape)

    # 获取到diff中的连通域
    contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    show_img = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)

    # 过滤掉面积小于100的连通域
    length = 0
    for c in contours:

        if cv2.contourArea(c) < 150:
            continue
        else:
            print(cv2.contourArea(c))
            # 计算连通域的外接矩形框
            (x, y, w, h) = cv2.boundingRect(c)
            # 在原始图像中绘制出外接矩形框

            cv2.rectangle(show_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            length += 1
    print('len', length)


    # 将show_img resize到长边为512，以进行展示
    w, h = show_img.shape[:2]
    resize_w = 512
    resize_h = int(512 * (h / w))

    show_img = cv2.resize(show_img, (resize_h, resize_w))

    # 显示图像
    cv2.imshow("Moving Object Detection", show_img)

    # 准备下一轮迭代，更新帧
    frame1 = frame2
    frame2 = frame3
    ret, frame3 = cap.read()
    # 判断 frame3 是不是None
    if not ret:
        break
    frame3 = frame3[:,:, 0]

    # 如果按下q键，停止循环
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()