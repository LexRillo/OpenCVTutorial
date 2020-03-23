import cv2
import toolbox
import numpy as np
from matplotlib import pyplot as plt

# Translation
image = cv2.imread('Lenna.png')
cv2.imshow("Original", image)
shifted = toolbox.translate(image, 0, 100)
cv2.imshow("Shifted Down", shifted)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Rotation
rotated = toolbox.rotate(image, 180)
cv2.imshow("Rotated by 180 Degrees", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Scale
# r = 50.0 / image.shape[0]
# dim = (int(image.shape[1] * r), 50)
# resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
resized = toolbox.resize(image, width = 100)
cv2.imshow("Resized (Height)", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Flipping image
cv2.imshow("Original", image)
flipped = cv2.flip(image, 1)
cv2.imshow("Flipped Horizontally", flipped)
flipped = cv2.flip(image, 0)
cv2.imshow("Flipped Vertically", flipped)
flipped = cv2.flip(image, -1)
cv2.imshow("Flipped Horizontally & Vertically", flipped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Masking
cv2.imshow("Original", image)
mask = np.zeros(image.shape[:2], dtype = "uint8")
(cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
cv2.rectangle(mask, (cX - 75, cY - 75), (cX + 75 , cY + 75), 255,-1)
cv2.imshow("Mask", mask)
masked = cv2.bitwise_and(image, image, mask = mask)
cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)
mask = np.zeros(image.shape[:2], dtype = "uint8")
cv2.circle(mask, (cX, cY), 100, 255, -1)
masked = cv2.bitwise_and(image, image, mask = mask)
cv2.imshow("Mask", mask)
cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Splitting channels
(B, G, R) = cv2.split(image)
cv2.imshow("Red", R)
cv2.imshow("Green", G)
cv2.imshow("Blue", B)
cv2.waitKey(0)

zeros = np.zeros(image.shape[:2], dtype = "uint8")
cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))
cv2.waitKey(0)

merged = cv2.merge([B, G, R])
cv2.imshow("Merged", merged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Colorspaces
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv)

lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow("L*a*b*", lab)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Grayscale Histograms
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()

# Colour Histograms
chans = cv2.split(image)
colors = ("b", "g", "r")
plt.figure()
plt.title("’Flattened’ Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color = color)
    plt.xlim([0, 256])
plt.show()

# 2D Colour Histograms
fig = plt.figure()
ax = fig.add_subplot(131)
hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None,[32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation = "nearest")
ax.set_title("2D Color Histogram for G and B")
plt.colorbar(p)
ax = fig.add_subplot(132)
hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None,[32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation = "nearest")
ax.set_title("2D Color Histogram for G and R")
plt.colorbar(p)
ax = fig.add_subplot(133)
hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None,[32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation = "nearest")
ax.set_title("2D Color Histogram for B and R")
plt.colorbar(p)
plt.show()
print("2D histogram shape: {}, with {} values".format(hist.shape, hist.flatten().shape[0]))

# 3D histogram
hist = cv2.calcHist([image], [0, 1, 2],None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
print("3D histogram shape: {}, with {} values".format(hist.shape, hist.flatten().shape[0]))
plt.show()

# Histogram equalization
eq = cv2.equalizeHist(gray)
cv2.imshow("Histogram Equalization", np.hstack([gray, eq]))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Average Blurring
blurred = np.hstack([cv2.blur(image, (3, 3)),cv2.blur(image, (5, 5)),cv2.blur(image, (7, 7))])
cv2.imshow("Averaged", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Gaussian Blurring
blurred = np.hstack([cv2.GaussianBlur(image, (3, 3), 0),cv2.GaussianBlur(image, (5, 5), 0),cv2.GaussianBlur(image, (7, 7), 0)])
# the las parameter is the sigma of the kernel. If it's 0 we make cv2 calculate for the image kernel automatically
cv2.imshow("Gaussian", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Median Blurring
blurred = np.hstack([cv2.medianBlur(image, 3), cv2.medianBlur(image, 5), cv2.medianBlur(image, 7)])
cv2.imshow("Median", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Bilateral filter (Blurring)
blurred = np.hstack([cv2.bilateralFilter(image, 5, 21, 21), cv2.bilateralFilter(image, 7, 31, 31), cv2.bilateralFilter(image, 9, 41, 41)])
cv2.imshow("Bilateral", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Thresholding
gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
(T, thresh) = cv2.threshold(gray_blurred, 155, 255, cv2.THRESH_BINARY)
#The threshold is 155 and whatever is higher than that is set to 255
cv2.imshow("Threshold Binary", thresh)

(T, threshInv) = cv2.threshold(gray_blurred, 155, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Threshold Binary Inverse", threshInv)
cv2.imshow("Coins", cv2.bitwise_and(image, image, mask =threshInv))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Adaptive threshold
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("Image", image)
thresh = cv2.adaptiveThreshold(gray_blurred, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
cv2.imshow("Mean Thresh", thresh)
thresh = cv2.adaptiveThreshold(gray_blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
cv2.imshow("Gaussian Thresh", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Otsu thresholding
import mahotas
T = mahotas.thresholding.otsu(gray_blurred)
print("Otsu’s threshold: {}".format(T))
thresh = image.copy()
thresh[thresh > T] = 255
thresh[thresh < 255] = 0
thresh = cv2.bitwise_not(thresh)
cv2.imshow("Otsu", thresh)

T = mahotas.thresholding.rc(blurred)
print("Riddler-Calvard: {}".format(T))
thresh = image.copy()
thresh[thresh > T] = 255
thresh[thresh < 255] = 0
thresh = cv2.bitwise_not(thresh)
cv2.imshow("Riddler-Calvard", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()