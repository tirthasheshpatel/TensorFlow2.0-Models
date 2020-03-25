import tensorflow as tf

def accuracy(labels, logits):
    return tf.reduce_mean(tf.cast(tf.cast(labels, tf.float32) == tf.cast(logits > 0.5, tf.float32), tf.float32))
