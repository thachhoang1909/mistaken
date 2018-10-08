import numpy as np

print("Reading file egocentric_image_features.npy...")
egocentric_image_features = np.load('data/egocentric_image_features.npy')
print("Save test_egocentric_image_features.npy...")
test_egocentric = egocentric_image_features[:100]
np.save('test_egocentric_image_features.npy', test_egocentric)
print("Done!")
