import tensorflow as tf

def fully_connected(input, size):
    weights = tf.get_variable('weights', shape=[input.get_shape()[1], size], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', shape=[size], initializer=tf.constant_initializer(0.0))
    return tf.matmul(input, weights) + biases

def fully_connected_relu(input, size):
    return tf.nn.relu(fully_connected(input, size))

def conv_relu(input, kernel_size, depth):
    weights = tf.get_variable('weights', shape=[kernel_size, kernel_size, input.get_shape()[3], depth], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', shape=[depth], initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)

def pool(input, size):
    return tf.nn.max_pool(input, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')

def model_pass(input, params, is_training):
    with tf.variable_scope('conv1'):
        conv1 = conv_relu(input, kernel_size=params.conv1_k, depth=params.conv1_d)
        pool1 = pool(conv1, size=2)
        pool1 = tf.cond(is_training, lambda: tf.nn.dropout(pool1, keep_prob=params.conv1_p), lambda: pool1)
    with tf.variable_scope('conv2'):
        conv2 = conv_relu(pool1, kernel_size=params.conv2_k, depth=params.conv2_d)
        pool2 = pool(conv2, size=2)
        pool2 = tf.cond(is_training, lambda: tf.nn.dropout(pool2, keep_prob=params.conv2_p), lambda: pool2)
    with tf.variable_scope('conv3'):
        conv3 = conv_relu(pool2, kernel_size=params.conv3_k, depth=params.conv3_d)
        pool3 = pool(conv3, size=2)
        pool3 = tf.cond(is_training, lambda: tf.nn.dropout(pool3, keep_prob=params.conv3_p), lambda: pool3)
    
    pool1 = pool(pool1, size=4)
    shape = pool1.get_shape().as_list()
    pool1 = tf.reshape(pool1, [-1, shape[1] * shape[2] * shape[3]])
    
    pool2 = pool(pool2, size=2)
    shape = pool2.get_shape().as_list()
    pool2 = tf.reshape(pool2, [-1, shape[1] * shape[2] * shape[3]])
    
    shape = pool3.get_shape().as_list()
    pool3 = tf.reshape(pool3, [-1, shape[1] * shape[2] * shape[3]])
    
    flattened = tf.concat(1, [pool1, pool2, pool3])
    
    with tf.variable_scope('fc4'):
        fc4 = fully_connected_relu(flattened, size=params.fc4_size)
        fc4 = tf.cond(is_training, lambda: tf.nn.dropout(fc4, keep_prob=params.fc4_p), lambda: fc4)
    
    with tf.variable_scope('fc5'):
        fc5 = fully_connected(fc4, size=43)
    
    return fc5
