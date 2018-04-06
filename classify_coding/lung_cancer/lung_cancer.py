import xlrd,numpy as np
from sklearn.preprocessing import Imputer,normalize,scale
from sklearn.svm import SVC,LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from scipy.io import savemat

wb = xlrd.open_workbook('lung_data.xls').sheet_by_name('Sheet1')
Y = wb.col_values(0)
label = np.asarray(Y)
X = [wb.row_values(i)[1:]for i in range(0,wb.nrows) ]
X = Imputer().fit_transform(X)
savemat('lung_cancer',{'data':X,'label':label})

# data = normalize(X)
data = scale(X)
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
