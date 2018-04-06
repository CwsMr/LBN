import pandas as pd,numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.io import savemat
from sklearn.preprocessing import normalize,scale
from sklearn.svm import SVC,LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data').values
X,Y = [],[]
for d in df:
    X.append(d[:-1])
    Y.append(d[-1])
data = np.asarray(X)
label = np.asarray(Y)
le = LabelEncoder()
label = le.fit_transform(label)
# savemat('ionosphere',{'Data':data,'Label':label})
data = normalize(data)
predict = []
for train,test in LeaveOneOut().split(data,label):
    # clf = LinearSVC()
    # clf = GaussianNB()
    # clf = LogisticRegression()
    # clf = SGDClassifier()
    # clf = KNeighborsClassifier()
    # clf = DecisionTreeClassifier()
    # clf = RandomForestClassifier()
    # clf = ExtraTreesClassifier()
    clf = GradientBoostingClassifier()
    clf.fit(data[train],label[train])
    pre = clf.predict(data[test])
    print(pre,label[test])
    for p in pre:
        predict.append(p)

accuracy = accuracy_score(label,predict)
print('accuracy  is %f%%'%(accuracy*100))


