from segment_anything import SamPredictor, sam_model_registry
from skimage.io import imread, imsave
import numpy as np

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")

predictor = SamPredictor(sam)
im_np = imread('/home/abe/Dropbox/tmp/cup.jpg')

predictor.set_image(im_np)

im_width = im_np.shape[1]
im_height = im_np.shape[0]

masks, _, _ = predictor.predict(
    # make a prediction where we say the central point is foreground.
    point_coords=np.array([np.array([im_width // 2, im_height // 2])]),
    point_labels=np.array([1]) # foreground
)

print('len masks', len(masks))
for i, m in enumerate(masks):
    out_path = f'/home/abe/Dropbox/tmp/cup_seg_{str(i).zfill(3)}.png'
    print('saving', out_path)
    imsave(out_path, m)
