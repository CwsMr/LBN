from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import LeaveOneOut
import tensorflow as tf,matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
data = load_breast_cancer()
X = normalize(data['data'])
y = data['target']
print(X.shape)

xs = tf.placeholder(tf.float32,shape=[None,30])
ys = tf.placeholder(tf.float32,shape=[None,2])

def add_layers(data,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]))

    Wx_plus_b = tf.matmul(data,Weights) + biases

    if activation_function:
        outputs = activation_function(Wx_plus_b)
    else:
        outputs = Wx_plus_b
    return outputs

l1 = add_layers(xs,30,10,activation_function=tf.nn.relu)
predict = add_layers(l1,10,2,activation_function=tf.nn.softmax)

loss = tf.nn.softmax_cross_entropy_with_logits(labels=ys,logits=predict)
loss = tf.reduce_mean(loss)
train = tf.train.AdamOptimizer(0.1).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
Y = tf.one_hot(y,depth=2,axis=1,dtype=tf.float32)

output_res,out_f1 = [],[]
for i in range(20):
    pred = []
    for t,(train_index,test_index) in enumerate(LeaveOneOut().split(X)):
        X_train,X_test = X[train_index],X[test_index]
        y_train,y_test = sess.run(Y)[train_index],sess.run(Y)[test_index]
        # print(y_train.shape)
        sess.run(train,feed_dict={xs:X_train,ys:y_train})
        pre = sess.run(tf.argmax(sess.run(predict,feed_dict={xs:X_test}),1))
        for p in pre:
            pred.append(pre)
    accuracy = accuracy_score(y,pred)
    f1 = f1_score(y,pred)
    print('----%s times  %f%%------' % (i,accuracy*100), f1)
    output_res.append(accuracy)
    out_f1.append(f1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([i for i in range(0, 20)], output_res, lw=5, color='red',label='ACCURACY')
ax.plot([i for i in range(0,20)],out_f1,label ='F1_SCORE',lw=5,color='blue')
ax.legend(loc='best')
ax.set_xlabel('iterors')
ax.set_ylabel('accuracy')
ax.set_title('breast_LeaveOutone')
plt.savefig('breast_LeaveOutone')
plt.show()