from sklearn.datasets import load_diabetes
from sklearn.preprocessing import normalize
from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
import tensorflow as tf
import os,numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


data = load_diabetes()
X = data['data']
y = data['target']
Y = y[:,np.newaxis]
X = normalize(X)

xs = tf.placeholder(tf.float32,shape=[None,10])
ys = tf.placeholder(tf.float32,shape=[None,1])

def add_layers(data,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]))

    Wx_plus_b = tf.matmul(data,Weights) + biases

    if activation_function:
        outputs = activation_function(Wx_plus_b)
    else:
        outputs = Wx_plus_b
    return outputs

l1 = add_layers(xs,10,50,activation_function=tf.nn.sigmoid)
l2 = add_layers(l1,50,100,activation_function=tf.nn.sigmoid)
predict = add_layers(l2,100,1,activation_function=None)
#13-50-100-1(0.1)
loss = tf.reduce_mean(tf.square(ys-predict))
tf.summary.scalar('loss',loss)
train = tf.train.AdamOptimizer(0.1).minimize(loss)

sess = tf.Session()
merge = tf.summary.merge_all()
sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter('diabetes/train')
test_writer = tf.summary.FileWriter('diabetes/test')
save_r = []
for i in range(100):
    pre = []
    for t,(train_index,test_index) in enumerate(LeaveOneOut().split(X,Y)):
        X_train,y_train = X[train_index],Y[train_index]
        X_test,y_test = X[test_index],Y[test_index]
        sess.run(train, feed_dict={xs: X_train, ys: y_train})
        pred = sess.run(predict,feed_dict={xs:X_test})
        for p in pred:
            for P in p:
                pre.append(P)
        if t == 441:
            train_res = sess.run(merge,feed_dict={xs:X_train,ys:y_train})
            test_res = sess.run(merge,feed_dict={xs:X_test,ys:y_test})
            train_writer.add_summary(train_res,i)
            test_writer.add_summary(test_res,i)
    [r,p] = pearsonr(y,pre)
    save_r.append(r)
    print('%s time'%(i),'r= ',r,'p =',p)

for s in save_r:
    print(s)
