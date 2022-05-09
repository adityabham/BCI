import numpy as np
from cross_validation import CV

# Lambdas
params = [1000, 1, 0.01, 1e-4]

# Img
img_1 = np.genfromtxt('feaSubEImg_1.csv', delimiter=',').T
img_2 = np.genfromtxt('feaSubEImg_2.csv', delimiter=',').T
X_img = np.vstack((img_1, img_2))
y_img = np.r_[np.zeros(120), np.ones(120)]

CV(X_img, y_img, params)

# Overt
overt_1 = np.genfromtxt('feaSubEOvert_1.csv', delimiter=',').T
overt_2 = np.genfromtxt('feaSubEOvert_2.csv', delimiter=',').T
X_overt = np.vstack((overt_1, overt_2))
y_overt = np.r_[np.zeros(120), np.ones(120)]

# CV(X_overt, y_overt, params)
