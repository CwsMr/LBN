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

wb = xlrd.open_workbook('australian-statlog.xls').sheet_by_name('Sheet1')
label = wb.col_values(-1)
data = [wb.row_values(i)[:-1] for i in range(wb.nrows)]
label,data = np.asarray(label),np.asarray(data)
# print(data.shape,label.shape,data[0])
# savemat('australian-statlog',{'Data':data,'Label':label})
data = normalize(data)

predict = []
for i,(train,test) in enumerate(LeaveOneOut().split(data,label)):
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
    print(i,pre,label[test])
    for p in pre:
        predict.append(p)

accuracy = accuracy_score(label,predict)
print('accuracy  is %f%%'%(accuracy*100))






