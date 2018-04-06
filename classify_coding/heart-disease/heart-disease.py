import xlrd,numpy as np
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

wk = xlrd.open_workbook('heart-disease.xls').sheet_by_name('Sheet1')
Y = wk.col_values(-1)
X = [wk.row_values(i)[:-1] for i in range(0,wk.nrows)]
data = np.asarray(X)
label = np.asarray(Y)
# print(len(data),data.shape,len(label))
savemat('heart-disease',{'Data':data,'label':label})

# data = normalize(data)
predict = []
for train,test in LeaveOneOut().split(data,label):
    # clf = LinearSVC()
    # clf = GaussianNB()
    # clf = LogisticRegression()
    # clf = SGDClassifier()
    clf = KNeighborsClassifier()
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

