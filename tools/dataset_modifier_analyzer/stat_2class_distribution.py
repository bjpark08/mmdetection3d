import pickle
import numpy as np
import math

voxel_size=10
point_cloud_range=[200,200]

xcnt=int(point_cloud_range[0]/voxel_size)
ycnt=int(point_cloud_range[1]/voxel_size)

car_dis=[[0]*ycnt for i in range(xcnt)]
ped_dis=[[0]*ycnt for i in range(xcnt)]
car_cnt=0
ped_cnt=0
car_dis2=[0]*12
ped_dis2=[0]*12

file_name='data/rf2021/rf2021_infos_train_height_make'

with open(file_name+'.pkl','rb') as f:
	datas=pickle.load(f)

maxx=0
maxy=0
for data in datas:
	for i in range(len(data['annos']['gt_names'])):
		box=data['annos']['gt_bboxes_3d'][i]
		label=data['annos']['gt_names'][i]
		x=int(box[0]/voxel_size)
		y=int(box[1]/voxel_size)
		z=int(math.sqrt(box[0]*box[0]+box[1]*box[1])/voxel_size)
		if x<-6 or x>6 or y<-10 or y>6:
			continue

		maxx=max(abs(x),maxx)
		maxy=max(abs(y),maxy)
		x+=9
		y+=9

		if label=='Car':
			car_cnt+=1
			car_dis[x][y]+=1
			car_dis2[z]+=1
		elif label=='Pedestrian':
			ped_cnt+=1
			ped_dis[x][y]+=1
			ped_dis2[z]+=1

print(car_cnt, ped_cnt)
for i in range(xcnt):
	for j in range(ycnt):
		print(round(car_dis[i][j]/car_cnt,3),end="\t")
	print()

print()
for i in range(xcnt):
	for j in range(ycnt):
		print(round(ped_dis[i][j]/ped_cnt,3),end="\t")
	print()

print()
for i in range(len(car_dis2)):
	print(round(car_dis2[i]/car_cnt,3),end="\t")

print()
for i in range(len(ped_dis2)):
	print(round(ped_dis2[i]/ped_cnt,3),end="\t")