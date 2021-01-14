from calculators import disparitycalculator as dispcal
from matplotlib import pyplot as plt

disparity = dispcal.compute('./resources/Left.png', './resources/Right.png')
plt.imshow(disparity,'gray')
plt.show()
