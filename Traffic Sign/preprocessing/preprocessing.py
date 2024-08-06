import numpy as np
from sklearn.utils import shuffle
from skimage import exposure

def preprocess_dataset(X, y=None):
    X = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]
    X = (X / 255.).astype(np.float32)
    
    for i in range(X.shape[0]):
        X[i] = exposure.equalize_adapthist(X[i])
    
    if y is not None:  
        y = np.eye(43)[y]
        X, y = shuffle(X, y)

    X = X.reshape(X.shape + (1,))
    return X, y
