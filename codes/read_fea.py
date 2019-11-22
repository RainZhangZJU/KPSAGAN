import scipy.io as io
import matplotlib.pyplot as plt
import numpy as np

data = io.loadmat('1956en.mat')
#print(data)
data = data['re']
print(data.shape)
img = data.reshape(21760,21760)
#img = img*100000
img = img[0,0:21760]
img = img.reshape(128,170)
m = np.max(img)
print(m)
#img = img.reshape(170,128)
#img = np.matrix([(4.63608885,4.62597091,4.61224154,4.62117678,4.62109565,4.62128737),(4.62175594,4.62153112,4.62096032,4.68637263,4.68521612,4.68601902),(4.68611652,4.68558028,4.68532526,4.66034035,4.66002894,4.65971971),(4.58959839,4.59393996,4.59533985,4.57774913,4.57808055,4.58192226),(4.58257382,4.57788119,4.58178947,4.62211101,4.61723903,4.61576637),(4.63245233,4.64054683,4.63740653,4.64758814,4.63086362,4.59216644)])
#img=img.astype(np.float64)
print(img)
plt.axis('off')
plt.imshow(img,plt.cm.jet)
plt.savefig("en.png")
