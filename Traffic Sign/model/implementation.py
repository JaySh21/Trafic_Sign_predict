import tensorflow as tf

class ModelParams:
    def __init__(self):
        self.conv1_k = 5
        self.conv1_d = 32
        self.conv1_p = 0.5
        self.conv2_k = 5
        self.conv2_d = 64
        self.conv2_p = 0.5
        self.conv3_k = 5
        self.conv3_d = 128
        self.conv3_p = 0.5
        self.fc4_size = 1024
        self.fc4_p = 0.5

def get_placeholders():
    images = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
    labels = tf.placeholder(tf.float32, shape=(None, 43))
    is_training = tf.placeholder(tf.bool)
    return images, labels, is_training
