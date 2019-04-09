import cv2
import matplotlib.pyplot as plt
 
im = cv2.imread('./pytorch-CycleGAN-and-pix2pix/photos/resized/2.jpg')
edges = cv2.Canny(im,256,256,L2gradient=False)
plt.imshow(edges,cmap='gray')
plt.show()
