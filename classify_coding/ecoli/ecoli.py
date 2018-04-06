import xlrd,numpy as np
from scipy.io import savemat
from sklearn.preprocessing import LabelEncoder,normalize
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import  GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

wk = xlrd.open_workbook('ecoli.xls')
X = wk.sheet_by_name('data')
Y = wk.sheet_by_name('label')
data = [X.row_values(i) for i in range(X.nrows)]
label = Y.col_values(0)
label = LabelEncoder().fit_transform(label)
data = np.asarray(data)
label = np.asarray(label)
# savemat('ecoli',{'Data':data,'Label':label})
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




