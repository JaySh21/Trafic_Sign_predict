import tensorflow as tf

def l2_regularization(weights_list):
    l2_reg = 0
    for weights in weights_list:
        l2_reg += tf.nn.l2_loss(weights)
    return l2_reg

def total_loss(logits, labels, weights_list, l2_strength):
    ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    l2_reg = l2_regularization(weights_list)
    return ce_loss + l2_strength * l2_reg
