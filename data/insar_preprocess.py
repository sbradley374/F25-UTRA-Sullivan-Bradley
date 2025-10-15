from PIL import Image
import numpy as np

for i in range(1, 1001):
    img1 = Image.open('input/%d.png' % i)
    img2 = Image.open('output_scaled_up/%d.png' % i)
    img1 = img1.resize((384, 384))
    img2 = img2.resize((384, 384))
    buf1 = np.asarray(img1)
    buf2 = np.asarray(img2)
    output = Image.fromarray(np.minimum(255., (buf1 + buf2 * .5)).astype(np.uint8))
    output.save('input_scaled_resize/%d.png' % i)
    img2.save('output_scaled_resize/%d.png' % i)
