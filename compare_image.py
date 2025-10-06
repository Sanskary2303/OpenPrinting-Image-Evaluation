import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np

original = cv2.imread('sample.png')
processed = cv2.imread('sample.png')

gray1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

(score, diff) = ssim(gray1, gray2, full=True)
print(f"SSIM Score: {score}")

diff = (diff * 255).astype("uint8")

cv2.imshow("Difference", diff)
cv2.waitKey(0)
cv2.destroyAllWindows()

