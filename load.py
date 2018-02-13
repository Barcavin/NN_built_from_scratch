import numpy as np
import pickle
import gzip

def val_to_onehot(y):
	"""
		transfer label to one hot vector.
	"""
	out = np.zeros((10,len(y)))
	out[y,np.arange(len(y))] = 1
	return out

def load_helper():
	with gzip.open('mnist.pkl.gz','rb') as f:
		u = pickle._Unpickler(f)
		u.encoding = 'latin1'
		ori_train,ori_val,ori_test=  u.load()
	return ori_train,ori_val,ori_test

def data_preprocess(data):
	"""
		Change the shape of the original data to fit in our train method.
	"""
	n = data[0].shape[0]
	x_n = data[0].shape[1]
	y = val_to_onehot(data[1])
	if n==y.shape[1]:
		result = []
		for i in range(n):
			x = data[0][i].reshape(x_n,1)
			y_result = y[:,[i]]
			result.append((x,y_result))
		return result
	else:
		print("The # of train and # of labels are not equal")
		return


def load_data(original=False):
	if original:
		return load_helper()
	else:
		ori_train,ori_val,ori_test = load_helper()
		return data_preprocess(ori_train),data_preprocess(ori_val),data_preprocess(ori_test)
