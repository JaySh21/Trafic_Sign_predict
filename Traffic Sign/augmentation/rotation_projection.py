from skimage.transform import rotate, warp, ProjectiveTransform
import numpy as np
import random

def rotate_images(X, intensity):
    for i in range(X.shape[0]):
        delta = 30. * intensity
        X[i] = rotate(X[i], random.uniform(-delta, delta), mode='edge')
    return X

def apply_projection_transform(X, intensity):
    image_size = X.shape[1]
    d = image_size * 0.3 * intensity
    for i in range(X.shape[0]):
        tl_top = random.uniform(-d, d)
        tl_left = random.uniform(-d, d)
        bl_bottom = random.uniform(-d, d)
        bl_left = random.uniform(-d, d)
        tr_top = random.uniform(-d, d)
        tr_right = random.uniform(-d, d)
        br_bottom = random.uniform(-d, d)
        br_right = random.uniform(-d, d)

        transform = ProjectiveTransform()
        transform.estimate(np.array((
                (tl_left, tl_top),
                (bl_left, image_size - bl_bottom),
                (image_size - br_right, image_size - br_bottom),
                (image_size - tr_right, tr_top)
            )), np.array((
                (0, 0),
                (0, image_size),
                (image_size, image_size),
                (image_size, 0)
            )))
        X[i] = warp(X[i], transform, output_shape=(image_size, image_size), order=1, mode='edge')
    
    return X
