from sklearn.datasets import load_breast_cancer
from scipy.io import savemat
file = load_breast_cancer()
data = file['data']
label = file['target']
savemat('breast_cancer',{'Data':data,'Label':label})