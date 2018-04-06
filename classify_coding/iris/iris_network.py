#coding：utf-8
# 兰鸢化数据集的留一法和神经网络训练融合
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer,normalize
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import  accuracy_score,f1_score
from matplotlib import pyplot as plt
import tensorflow as tf
data = load_iris()
X = data['data']
X = normalize(X)
Y = data['target']
lb = LabelBinarizer()
y = lb.fit_transform(Y)

xs = tf.placeholder(tf.float32,shape=[None,4])
ys = tf.placeholder(tf.float32,shape=[None,3])

def add_layers(data,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]))

    Wx_plus_b = tf.matmul(data,Weights) + biases

    if activation_function:
        outputs = activation_function(Wx_plus_b)
    else:
        outputs = Wx_plus_b
    return outputs

l1 = add_layers(xs,4,10,activation_function=tf.nn.relu)
predict = add_layers(l1,10,3,activation_function=tf.nn.softmax)

loss = tf.nn.softmax_cross_entropy_with_logits(labels=ys,logits=predict)
loss = tf.reduce_mean(loss)

train = tf.train.AdamOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    output_accuracy,out_f1 = [],[]
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        loo = LeaveOneOut()
        lo = loo.split(X, y)
        pred = []
        for train_index, test_index in lo:
            train_X, train_Y = X[train_index], y[train_index]
            test_X, test_Y = X[test_index], y[test_index]
            sess.run(train,feed_dict={xs:train_X,ys:train_Y})
            test_res = tf.argmax(sess.run(predict, feed_dict={xs: test_X}),1)
            res = sess.run(test_res)
            for r in res:
                pred.append(r)
        test_accuracy = accuracy_score(Y,pred)
        print(test_accuracy*100)
        output_accuracy.append(test_accuracy)