import numpy as np
import tensorflow as tf;

#c = tf.constant([[0.0, 0.0, 0.0],
#    [0.0, 0.0, 0.0],
#    [0.0, 0.0, 0.0]])
#sess = tf.Session()
#sess.run(result)
def rectify():
    a = np.array([1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9])
    b = a.reshape([3,3,3])
    current = tf.constant(b);
    bit_map = tf.cast(tf.greater(current,5), tf.int64)
    tf_sum = tf.reduce_sum(current,[0,1,2])
    sess = tf.Session()
    sess.run(tf_sum)
    print tf_sum.eval(session=sess)
    print bit_map.eval(session=sess)
    tf_count = tf.reduce_sum(bit_map,[0,1,2])
    tf_avg = tf.divide(tf_sum,tf_count)
    avg_bit_map = tf.multiply(tf.cast(tf.equal(bit_map,0),tf.float64),tf_avg)
    print avg_bit_map.eval(session=sess)

def divideError():
    a = np.array([1,2,3,4,5])
    b  = np.matmul(a.T,a)/(np.float(1)/a.size)
    print b
def rectifyWithConv() :
    a = tf.cast(np.ones([1, 6,6]), tf.float32);
    w = tf.cast(np.ones([3, 3]), tf.float32);
    b = tf.nn.conv1d(a, w, stride =1, padding='SAME',use_cudnn_on_gpu=False)
    sess = tf.Session()
    sess.run(a)
    print a.eval(session=sess)
    print w.eval(session=sess)
    print b.eval(session=sess)



def rectifyEdges(current,current_bitMap,count_bit_map):
    count_bit_map = _conv_layer_sum2d(current_bitMap)
    #count_bit_map = tf.select(tf.equal(count_bit_map,0),
    #        tf.ones(count_bit_map.get_shape()),
    #            count_bit_map)
    sum_current = _conv_layer_sum3d(current)
    #avg_current = tf.divide(sum_current,count_bit_map)

    #avg_current_masked = tf.multiply(tf.cast(tf.equal(current_bitMap,0),tf.float32),avg_current)
    return sum_current

def _conv_layer_sum2d(input):
    shape = (3,3,1,1)
    #weights = tf.ones(shape)
    weights = tf.ones(shape)
    conv = tf.nn.conv2d(input, weights,strides=(1, 1, 1, 1),padding='SAME')
    return conv

def _conv_layer_sum3d(input):
    input_shape = map(int,list(input.get_shape())) + [1]
    input = tf.reshape(input,input_shape)
    print input_shape

    shape = (1,3,3,1,1)
    weights = np.array([[1,1,1],[1,1,1],[1,1,1]]).reshape(shape)
    conv = tf.nn.conv3d(input, weights,
            strides=(1, 1, 1, 1,1),padding='SAME')
    return conv


if __name__ == '__main__':
    a = np.array([1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9])
    b = a.reshape([1,3,3,3])
    current = tf.constant(b,tf.float32);
    bit_map = tf.cast(tf.greater(current,5), tf.float32)
    bit_map = tf.reshape(tf.reduce_prod(bit_map,[1]),[1,3,3,1])
    sess = tf.Session()
    t = rectifyEdges(current,
        bit_map,bit_map)
    sess.run(t)
    print t.eval(session=sess)
    print current.eval(session=sess)
    #print tf.reshape(bit_map,[1,3,3]).eval(session=sess)



