import pickle
import numpy

car_cnt=0
cyc_cnt=0
ped_cnt=0
car_mini_cnt=0
cyc_mini_cnt=0
ped_mini_cnt=0
car_super_cnt=0
cyc_super_cnt=0
ped_super_cnt=0

file_name='data/rf2021/rf2021_infos_train'

with open(file_name+'.pkl','rb') as f:
	datas=pickle.load(f)

for data in datas:
	for i in range(len(data['annos']['gt_bboxes_3d'])):
		box=data['annos']['gt_bboxes_3d'][i]
		label=data['annos']['gt_names'][i]
		if label=='Car':
			car_cnt+=1
			if box[5]<1:
				car_mini_cnt+=1
				#print("Car :"+str(box))
			if box[5]>4.5:
				car_super_cnt+=1
				#print("Car :"+str(box))
		elif label=='Cyclist':
			cyc_cnt+=1
			if box[5]<1:
				cyc_mini_cnt+=1
			if box[5]>3:
				cyc_super_cnt+=1
		elif label=='Pedestrian':
			ped_cnt+=1
			if box[5]<0.8:
				ped_mini_cnt+=1
				#print("Ped :"+str(box))
			if box[5]>2.5:
				ped_super_cnt+=1
				#print("Ped :"+str(box))
		#else:
			#print(label)

print(car_cnt, cyc_cnt, ped_cnt)
print(car_mini_cnt, cyc_mini_cnt, ped_mini_cnt)
print(car_super_cnt,cyc_super_cnt,ped_super_cnt)
print(car_mini_cnt/car_cnt, cyc_mini_cnt/cyc_cnt, ped_mini_cnt/ped_cnt)
