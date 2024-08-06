import tensorflow as tf
import numpy as np

def train_model(sess, train_X, train_y, valid_X, valid_y, model_pass, num_epochs=10, batch_size=64, learning_rate=1e-3, l2_strength=1e-4):
    images, labels, is_training = get_placeholders()
    logits = model_pass(images, ModelParams(), is_training)
    
    weights_list = [v for v in tf.trainable_variables() if 'weights' in v.name]
    loss = total_loss(logits, labels, weights_list, l2_strength)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(num_epochs):
        num_batches = len(train_X) // batch_size
        for batch in range(num_batches):
            batch_X = train_X[batch * batch_size: (batch + 1) * batch_size]
            batch_y = train_y[batch * batch_size: (batch + 1) * batch_size]
            feed_dict = {images: batch_X, labels: batch_y, is_training: True}
            _, batch_loss = sess.run([optimizer, loss], feed_dict=feed_dict)
        
        feed_dict = {images: valid_X, labels: valid_y, is_training: False}
        valid_accuracy = sess.run(accuracy, feed_dict=feed_dict)
        print(f'Epoch {epoch + 1}, Validation Accuracy: {valid_accuracy}')
