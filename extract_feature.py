import glob
import os
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications import resnet50
from keras.preprocessing import image
from keras.models import Model
import json
from tqdm import tqdm

## EGOCENTRIC_IMAGE_FOLDER = 'abc'
DATA_PATH = 'data'  # Where to store temporary files
PUBLIC_PATH = 'public'  # Where to show graphs and diagnostics
EGOCENTRIC_IMAGE_FOLDER = 'egocentric_images'  # Where egocentric images are stored
TMP_PATH = 'tmp'  # Temporary directly. Deleted after running
FEATURE_RESNET50_PATH = 'resnet50_features_original_size'

frame_dict_filename = os.path.join(DATA_PATH, 'frame_dict_resnet50_original.json')

image_features_filename = os.path.join(DATA_PATH, 'egocentric_image_resnet50_features_original.npy')

img_filename_vec = sorted(glob.glob(os.path.join(EGOCENTRIC_IMAGE_FOLDER, '*.png')))

frame_dict = {os.path.basename(filename).split('.')[0]: index for (index, filename) in enumerate(img_filename_vec)}

b_model = ResNet50(weights='imagenet', input_shape=(800, 1400, 3), include_top=False)
model = Model(inputs=b_model.input, outputs=b_model.layers[-1].output)

with open(frame_dict_filename, 'wb') as f:
    json.dump(frame_dict, f)
print("New Code")
# image_feature_matrix = np.zeros((len(img_filename_vec),3*6*2048))
used_indices = []
for (index, img_filename) in tqdm(enumerate(img_filename_vec), total=len(img_filename_vec)):
    temp_img_filename = "".join(img_filename.split('/')[1:])
    feature_filename = os.path.join(FEATURE_RESNET50_PATH, "".join(temp_img_filename.split('.')[:-1]) + '.npy')
    if os.path.exists(feature_filename):
        print("Existed: ", feature_filename)
        """
        try:
            pass
            #img_feature = np.load(feature_filename)
        except IOError:
            img = image.load_img(img_filename, target_size=(800, 1400))

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = resnet50.preprocess_input(x)
            img_feature = model.predict(x).flatten()

            np.save(feature_filename, img_feature)
        """
        pass
    else:
        print("Load File: ", feature_filename)
        img = image.load_img(img_filename, target_size=(800, 1400))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = resnet50.preprocess_input(x)
        img_feature = model.predict(x).flatten()

        np.save(feature_filename, img_feature)
    used_indices.append(index)
    # image_feature_matrix[index] = img_feature

assert used_indices == range(len(img_filename_vec))
# assert np.all(np.mean(image_feature_matrix != 0, axis=1) > 0)
# print("Feature shape: ", image_feature_matrix.shape)
print("Frame dict: ", frame_dict)

# np.save(image_features_filename, image_feature_matrix)
