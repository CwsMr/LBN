from sklearn.datasets import load_iris
from scipy.io import savemat
file = load_iris()
data = file['data']
label = file['target']
savemat('iris',{'Data':data,'Label':label})