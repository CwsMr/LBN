from sklearn.datasets import load_wine
from scipy.io import savemat
file = load_wine()
data = file['data']
label = file['target']
savemat('Wine',{'Data':data,'Label':label})