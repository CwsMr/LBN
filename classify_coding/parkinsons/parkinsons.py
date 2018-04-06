import pandas as pd,numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.io import savemat
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import  GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data').values
X,Y,Label = [],[],[]
for d in df:
    Y.append(d[0])
    X.append(d[1:])
for y in Y:
    Label.append(y[-1])
label = LabelEncoder().fit_transform(Label)
data,label = np.asarray(X),np.asarray(label)

savemat('parkinsons',{'Data':data,'Label':label})
data = normalize(data)
predict = []
for train,test in LeaveOneOut().split(data,label):
    # clf = LinearSVC()
    clf = GaussianNB()
    # clf = LogisticRegression()
    # clf = SGDClassifier()
    # clf = KNeighborsClassifier()
    # clf = DecisionTreeClassifier()
    # clf = RandomForestClassifier()
    # clf = ExtraTreesClassifier()
    # clf = GradientBoostingClassifier()
    clf.fit(data[train],label[train])
    pre = clf.predict(data[test])
    print(pre,label[test])
    for p in pre:
        predict.append(p)

accuracy = accuracy_score(label,predict)
print('accuracy  is %f%%'%(accuracy*100))







