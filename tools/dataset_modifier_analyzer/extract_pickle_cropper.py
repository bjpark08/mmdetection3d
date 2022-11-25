import pickle

file_name='data/rf2021/rf2021_infos_train'

with open(file_name+'.pkl','rb') as f:
	data=pickle.load(f)

crop_size = 4000
data=data[:crop_size]

with open(file_name+'_size'+str(crop_size)+'.pkl',"wb") as f:
	pickle.dump(data,f)
