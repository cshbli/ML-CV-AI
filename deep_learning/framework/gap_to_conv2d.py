import tensorflow as tf
import numpy as np

channels = 2048

np.random.seed(0)

# Do global average pooling on original input [1, 7, 7, 2048]
x = np.random.rand(1, 7, 7, channels)
y = tf.reduce_mean(x, [1, 2])

sess = tf.Session()
result = sess.run(y)
print(result[0])

# Change the input shape to [1, 7, 7*2048, 1]
x1 = x[0,:,:,0]     # x1 is one 7x7 block
for i in range(channels - 1):
    c = x[0,:,:,i+1]
    x1 = np.concatenate((x1, c), axis = 1)
x1 = np.expand_dims(x1, axis=0)
x1 = np.expand_dims(x1, axis=3)

# Do Conv2D with the kernel 7x7, only 49 weight values
x1.astype(np.float64)
weight = tf.constant(1/49, dtype= tf.float64, shape=[7, 7, 1, 1])
y1 = tf.nn.conv2d(x1, weight, strides=[1, 7, 7, 1], padding="SAME")
result1 = sess.run(y1)
# The output shape will be [1, 1, 2048, 1]
print(result1.shape)
print(result1[0][0])

# Change the input shape to [1, 7*32, 7*64, 1]
h_blocks = 32
w_blocks = 64
for row in range(h_blocks):    
    row_block = x[0,:,:,row*w_blocks]     # initial row_block is 7x7 
    for col in range(w_blocks - 1):
        # block is a 7x7 block        
        block = x[0, :, :, row * w_blocks + col + 1] 
        # each row block will be [7, 7x64] eventually
        row_block = np.concatenate((row_block, block), axis = 1) 

    if row==0:
        # The first row        
        x2 = row_block        
    else:
        x2 = np.concatenate((x2, row_block), axis = 0)
x2 = np.expand_dims(x2, axis=0)
x2 = np.expand_dims(x2, axis=3)

# Do Conv2D with the kernel 7x7, only 49 weight values
x2.astype(np.float64)
weight = tf.constant(1/49, dtype= tf.float64, shape=[7, 7, 1, 1])
y2 = tf.nn.conv2d(x2, weight, strides=[1, 7, 7, 1], padding="SAME")
result2 = sess.run(y2)
# The output shape will be [1, 32, 64, 1]
print(result2.shape)
print(result2[0])
