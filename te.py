import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('C:\ML PROJECTS\InfraScan-Sentinel\OIP.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

h,w = gray.shape
mid_y,mid_x = h//3,w//3
(b,g,r) = image[mid_y,mid_x]
print(b,g,r)
roi = gray[h//2,0:w]
brightness = roi[::2]
print(np.mean(brightness))

plt.plot(gray[h//2,0:w])
plt.xlabel('Pixel Position')
plt.ylabel('Brightness')
plt.show()