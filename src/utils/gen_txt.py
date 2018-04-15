"""Generate image labels."""

import os
from numpy import genfromtxt
import numpy as np

image_data_path = '/Users/macbookpro/Downloads/nolabels'

list_imname = []

for imname in os.listdir(image_data_path):
    im_num = imname.split('.')[0] + ',0\n'
    list_imname.append(im_num)

output_dir = '../output.txt'

with open("output.txt", "w") as f:
    for imname in list_imname:
        f.write(imname)

# find . -name '*.png' \
# | awk 'BEGIN{ a=13234 }{ printf "mv \"%s\" %d.jpg\n", $0, a++ }' \
# | bash

txtdir = '/Users/macbookpro/Desktop/total.csv'

arr = genfromtxt(txtdir, delimiter=',')

print(arr.shape[0])
np.random.shuffle(arr)

np.savetxt('train.txt', arr[0:14523, :], delimiter=',', fmt='%d')
np.savetxt('val.txt', arr[14523:19364, :], delimiter=',', fmt='%d')
np.savetxt('test.txt', arr[19364:, :], delimiter=',', fmt='%d')
