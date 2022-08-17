import pickle


with open("rf2021_infos_train_height_make.pkl","rb") as rf:
	data = pickle.load(rf)

data=data[:4000]

with open("rf2021_infos_train_height_make_size4000.pkl","wb") as rf2:
	pickle.dump(data,rf2)
