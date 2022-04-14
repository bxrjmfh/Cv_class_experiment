import numpy as np
import matplotlib.pyplot as plt

def f(x,y):
    return (x+y)*np.exp(-5.0*(x**2+y**2))

x,y = np.mgrid[-1:1:100j, -1:1:100j]

z = f(x,y)

plt.imshow(z,origin='lower')
plt.colorbar()

i,j = np.unravel_index(z.argmin(), z.shape)

plt.scatter(i,j,color='r')

plt.xlim(0,100)
plt.ylim(0,100)
plt.title('Draw a point on an image with matplotlib (2/2)')

plt.savefig("draw_on_image_01.png")
plt.show()
plt.close()