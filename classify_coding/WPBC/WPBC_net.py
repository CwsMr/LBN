from scipy.io import loadmat
from sklearn.preprocessing import LabelBinarizer,normalize,scale
from sklearn.model_selection import LeaveOneOut,train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

file = loadmat('wpbc.mat')
data = file['Data']
X = normalize(data)
label = file['Label'][0]
y = LabelBinarizer().fit_transform(label)
y = tf.reshape(tf.one_hot(y,2),[197,2])
print(data.shape,y.shape,y)
xs = tf.placeholder(tf.float32,shape=[None,33])
ys = tf.placeholder(tf.float32,shape=[None,2])
keep_pro = tf.placeholder(tf.float32)

def add_layers(data,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]))

    Wx_plus_b = tf.matmul(data,Weights) + biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_pro)
    if activation_function:
        outputs = activation_function(Wx_plus_b)
    else:
        outputs = Wx_plus_b
    return outputs

l1 = add_layers(xs,33,20,activation_function=tf.nn.relu)
l2 = add_layers(l1,20,10,activation_function=tf.nn.relu)
predict = add_layers(l2,10,2,activation_function=tf.nn.softmax)

loss = tf.nn.softmax_cross_entropy_with_logits(labels=ys,logits=predict)
loss = tf.reduce_mean(loss)

train = tf.train.AdamOptimizer(0.005).minimize(loss)
# train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output_res = []
    for i in range(50):
        loo = LeaveOneOut()
        lo = loo.split(X, y)
        pred = []
        for train_index, test_index in lo:
            train_X, train_Y = X[train_index], sess.run(y)[train_index]
            test_X, test_Y = X[test_index], sess.run(y)[test_index]
            sess.run(train,feed_dict={xs:train_X,ys:train_Y,keep_pro:0.8})
            test_res = tf.argmax(sess.run(predict, feed_dict={xs: test_X,keep_pro:0.8}),1)
            res = sess.run(test_res)
            for r in res:
                pred.append(r)
        test_accuracy = accuracy_score(label,pred)
        # print('----%s times------%f%%'%(i,test_accuracy*100))
        print(test_accuracy * 100)
        output_res.append(test_accuracy)

