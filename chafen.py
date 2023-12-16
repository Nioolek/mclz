import cv2
import numpy as np

cap = cv2.VideoCapture(r'D:\aliyun\video\ele_dfa10ee47b7705c786947aa90479cbb9.ts')  # read the video file

# Initialization
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while(1):
    ret, frame2 = cap.read()
    if not ret:
        break
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # compute magnitude of 2D vectors
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    print('!!!!')
    print(mag.max(), mag.min())

    # normalize magnitude to range 0-255
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    print(mag.max(), mag.min())


    # display the tracking result
    cv2.imshow('frame2', mag)
    # cv2.imshow('frame', frame2)

    # hit 'q' on the keyboard to stop the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # now update the previous frame
    prvs = next

# close all windows
cap.release()
cv2.destroyAllWindows()