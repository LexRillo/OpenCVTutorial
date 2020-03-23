import numpy as np
import cv2
from packaging import version

image = cv2.imread('Lenna.png')

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)
lap = cv2.Laplacian(image, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
cv2.imshow("Laplacian", lap)
lapint = cv2.Laplacian(image, cv2.CV_8U)
cv2.imshow("Laplacianint", lapint)
cv2.waitKey(0)
cv2.destroyAllWindows()

sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)

sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

sobelCombined = cv2.bitwise_or(sobelX, sobelY)

cv2.imshow("Sobel X", sobelX)
cv2.imshow("Sobel Y", sobelY)
cv2.imshow("Sobel Combined", sobelCombined)
cv2.waitKey(0)
cv2.destroyAllWindows()

image = cv2.imread('Lenna.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("Blurred", gray_blurred)

canny = cv2.Canny(gray_blurred, 30, 150)
cv2.imshow("Canny", canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Contours
if version.parse(cv2.__version__) < version.parse("2.5"):
    (cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
else:
    (_, cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print("I count {} coins in this image".format(len(cnts)))
coins = image.copy()
cv2.drawContours(coins, cnts, -1, (0, 255, 0), 2)
cv2.imshow("Contours", coins)
cv2.waitKey(0)
cv2.destroyAllWindows()