import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform



image = cv2.imread('C:\ML PROJECTS\InfraScan-Sentinel\OIP.jpg')
RGB = cv2.cvtColor(image,cv2.COLOR_BGRA2RGB)

input_batch = transform(RGB).to(device)

with torch.no_grad():
    prediction = midas(input_batch)
    b = prediction
    prediction=prediction.squeeze()


depth_map = prediction.cpu().numpy()
print(depth_map.shape[0])
h = depth_map.shape[0]
middle_row = depth_map[h//2]  # which row is the middle?
plt.plot(middle_row)  # what do you plot?
plt.xlabel('Pixel Position')
plt.ylabel('Depth')
plt.show()

x_scale = image.shape[0]/depth_map.shape[0]
y_scale = image.shape[1]/depth_map.shape[1]

depth_gradient = np.abs(np.gradient(middle_row))

threshold = np.percentile(depth_gradient, 90)  # top 10% strongest edges
edge_pixels_depth = np.where(depth_gradient > threshold)[0]
edge_pixels_original = edge_pixels_depth * x_scale

print("Depth edges in depth coordinates:", edge_pixels_depth)
print("Depth edges in original coordinates:", edge_pixels_original)
print("Before squeeze:", b.shape)

print("After squeeze:", prediction.shape)