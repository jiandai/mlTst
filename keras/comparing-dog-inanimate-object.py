"""
Use pretrained Deep Learning model to classify images on https://www.awesomeinventions.com/comparing-dog-inanimate-object/
# Version history:
    -20180222 by Jian: first pass
    -20180620 by Jian: download jpg directly, plot the prediction
# Ref: 
https://stackoverflow.com/questions/43462896/how-do-i-download-images-with-an-https-url-in-python-3?noredirect=1&lq=1
https://github.com/jiandai/mlTst/blob/master/keras/DL-features.ipynb
"""

import io
import requests
from PIL import Image
import numpy as np

from scipy.misc import imresize
from keras.applications.xception import Xception
from keras.applications.imagenet_utils import decode_predictions

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Image URLs and collage grid structure
img_info = [("https://www.awesomeinventions.com/wp-content/uploads/2016/03/dog-or-food-towel.jpg",3,4),
("https://www.awesomeinventions.com/wp-content/uploads/2016/03/dog-or-food-muffin.jpg",4,4),
("https://www.awesomeinventions.com/wp-content/uploads/2016/03/dog-or-food-mop.jpg",4,4),
("https://www.awesomeinventions.com/wp-content/uploads/2016/03/dog-or-food-loaf.jpg",3,4),
("https://www.awesomeinventions.com/wp-content/uploads/2016/03/dog-or-food-doodle.jpg",4,4),
("https://www.awesomeinventions.com/wp-content/uploads/2016/03/dog-or-food-croissant.jpg",3,4),
("https://www.awesomeinventions.com/wp-content/uploads/2016/03/dog-or-food-bear.jpg",3,4),
("https://www.awesomeinventions.com/wp-content/uploads/2016/03/dog-or-food-bagel.jpg",4,4)]

model = Xception()
output_file = open('output.csv','w')
output_file.write("image_url,row_in_grid,column_in_grid,top5_prediction\n")

for i, rec in enumerate(img_info):
  data = requests.get(rec[0]).content
  img = Image.open(io.BytesIO(data)) 
  img =np.array(img)
  print(rec[0], img.shape)
  X,Y,C = img.shape
  x,y = X//rec[1], Y//rec[2]
  # initiate plotting
  fig, ax = plt.subplots(rec[1], rec[2])
  plt.tight_layout()
  for r in range(rec[1]):
    output_line = ''
    for c in range(rec[2]):
      patch = img[r*x:(r+1)*x, c*y:(c+1)*y] 
      ax[r,c].imshow(patch)
      # prepare predicting
      patch = imresize(patch, [299,299])
      patch = np.expand_dims(patch, axis=0)
      # Change type
      patch = patch.astype(np.float32)
      patch /= 255.
      preds = model.predict(patch)
      decoded_preds = decode_predictions(preds)
      output_string = rec[0] + ','+str(r) +','+ str(c) + ',"'+ str(decoded_preds[0]) + '"\n'
      _a, _b, _c = decoded_preds[0][0]
      line = "{0} ({1:.2f}) ".format(_b, _c)
      ax[r,c].set_title(line, fontsize=10)
      ax[r,c].axis('off')
      output_line += line
      output_file.write(output_string)
    print(output_line)
    fn = rec[0][(rec[0].rfind("/")+1):]
  plt.savefig(fn)
  plt.close()
  plt.gcf().clear()


output_file.close()


