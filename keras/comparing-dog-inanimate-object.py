"""
Use pretrained Deep Learning model to classify images on https://www.awesomeinventions.com/comparing-dog-inanimate-object/
ref: https://github.com/jiandai/mlTst/blob/master/keras/DL-features.ipynb
#version history:
    -20180222 by Jian: first pass
"""

from scipy.misc import imread, imresize
from keras.applications.xception import Xception
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.xception import preprocess_input
import numpy as np

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
local_path = 'C:/Users/daij12/Downloads/'
output_file = open('output.csv','w')
output_file.write("image_url,row_in_grid,column_in_grid,top5_prediction\n")
for rec in img_info[:]:
    img = imread(local_path + rec[0].split("/")[-1])
    X,Y,C = img.shape
    x,y = X//rec[1], Y//rec[2]
    for r in range(rec[1]):
        for c in range(rec[2]):
            patch = img[r*x:(r+1)*x, c*y:(c+1)*y] 
            patch = imresize(patch, [299,299])
            patch = np.expand_dims(patch, axis=0)
            # Change type
            patch = patch.astype(np.float32)
            patch /= 255.
            preds = model.predict(patch)
            decoded_preds = decode_predictions(preds)
            output_string = rec[0] + ','+str(r) +','+ str(c) + ',"'+ str(decoded_preds[0]) + '"\n'
            output_file.write(output_string)

output_file.close()

