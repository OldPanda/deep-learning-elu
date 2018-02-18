import time

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

learning_rate = 0.01
training_epochs = 300
batch_size = 64
display_step = 1

n_h_1 = 128 
n_h_2 = 128
n_h_3 = 128
n_h_4 = 128
n_h_5 = 128
n_h_6 = 128
n_input = 784 
n_classes = 10 


X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])


weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_h_1])),
    'h2': tf.Variable(tf.random_normal([n_h_1, n_h_2])),
    'h3': tf.Variable(tf.random_normal([n_h_2, n_h_3])),
    'h4': tf.Variable(tf.random_normal([n_h_3, n_h_4])),
    'h5': tf.Variable(tf.random_normal([n_h_4, n_h_5])),
    'h6': tf.Variable(tf.random_normal([n_h_5, n_h_6])),
    'out': tf.Variable(tf.random_normal([n_h_6, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_h_1])),
    'b2': tf.Variable(tf.random_normal([n_h_2])),
    'b3': tf.Variable(tf.random_normal([n_h_3])),
    'b4': tf.Variable(tf.random_normal([n_h_4])),
    'b5': tf.Variable(tf.random_normal([n_h_5])),
    'b6': tf.Variable(tf.random_normal([n_h_6])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}



def build_nn(x):
    
    l1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    l1 = tf.nn.elu(l1)
    
    l2 = tf.add(tf.matmul(l1, weights['h2']), biases['b2'])
    l2 = tf.nn.elu(l2)
    
    l3 = tf.add(tf.matmul(l2, weights['h3']), biases['b3'])
    l3 = tf.nn.elu(l3)
    
    l4 = tf.add(tf.matmul(l3, weights['h4']), biases['b4'])
    l4 = tf.nn.elu(l4)
    
    l5 = tf.add(tf.matmul(l4, weights['h5']), biases['b5'])
    l5 = tf.nn.elu(l5)
    
    l6 = tf.add(tf.matmul(l5, weights['h6']), biases['b6'])
    l6 = tf.nn.elu(l6)
    
    out_layer = tf.matmul(l6, weights['out']) + biases['out']
    return out_layer


logits = build_nn(X)


loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    start_time = time.time()
    
    for epoch in range(training_epochs):
        a_c = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})

            a_c += c / total_batch

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(a_c))
    duration = time.time() - start_time
    print(duration)
    
    pred = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

