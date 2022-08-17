import pickle
import numpy as np
import math
import open3d as o3d
from pypcd import pypcd

voxel_size=10
point_cloud_range=[200,200]

xcnt=int(point_cloud_range[0]/voxel_size)
ycnt=int(point_cloud_range[1]/voxel_size)

car_dis=[[0]*ycnt for i in range(xcnt)]
ped_dis=[[0]*ycnt for i in range(xcnt)]
car_cnt=0
ped_cnt=0
car_out_cnt=0
ped_out_cnt=0
car_dis2=[0]*12
ped_dis2=[0]*12
car_dis3=[0]*10
ped_dis3=[0]*10
car_dis4=[0]*12
ped_dis4=[0]*12
car_dis5=[0]*12
ped_dis5=[0]*12
car_dis6=[0]*10
ped_dis6=[0]*10

file_name='rf2021_infos_train'
#file_name='rf2021_infos_train_height_make'

with open(file_name+'.pkl','rb') as f:
	datas=pickle.load(f)

maxx=0
maxy=0
cnt=0
for data in datas:
	cnt+=1
	if cnt%1000==0:
		print(cnt)
	for i in range(len(data['annos']['gt_names'])):
		box=data['annos']['gt_bboxes_3d'][i]
		label=data['annos']['gt_names'][i]
		x=int(box[0]/voxel_size)
		y=int(abs(box[1])/voxel_size)
		z=int(math.sqrt(box[0]*box[0]+box[1]*box[1])/voxel_size)
		if x<-6 or x>6 or y>10:
			continue

		maxx=max(abs(x),maxx)
		maxy=max(abs(y),maxy)
		
		x+=10

		if label=='Car':
			car_cnt+=1
			car_dis4[z]+=1

			center = data['annos']['gt_bboxes_3d'][i, 0:3]
			dim = data['annos']['gt_bboxes_3d'][i, 3:6]
			yaw = np.zeros(3)
			yaw[2] = data['annos']['gt_bboxes_3d'][i, 6]
			rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(yaw)

			box3d = o3d.geometry.OrientedBoundingBox(center, rot_mat, dim)
			pcd = o3d.io.read_point_cloud(data['lidar_points']['lidar_path'])
			indices = box3d.get_point_indices_within_bounding_box(pcd.points)

			w=min(9,int(len(indices)/100))
			car_dis6[w]+=1

			if box[5]<1 or box[5]>4.5:
				car_out_cnt+=1
				car_dis[x][y]+=1
				car_dis2[z]+=1
				car_dis3[w]+=1

				if len(indices)<100:
					car_dis5[z]+=1

				

		elif label=='Pedestrian':
			ped_cnt+=1
			ped_dis4[z]+=1
			center = data['annos']['gt_bboxes_3d'][i, 0:3]
			dim = data['annos']['gt_bboxes_3d'][i, 3:6]
			yaw = np.zeros(3)
			yaw[2] = data['annos']['gt_bboxes_3d'][i, 6]
			rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(yaw)

			box3d = o3d.geometry.OrientedBoundingBox(center, rot_mat, dim)
			pcd = o3d.io.read_point_cloud(data['lidar_points']['lidar_path'])
			indices = box3d.get_point_indices_within_bounding_box(pcd.points)

			w=min(9,int(len(indices)/20))
			ped_dis6[w]+=1

			if box[5]<0.8 or box[5]>2.5:
				ped_out_cnt+=1
				ped_dis[x][y]+=1
				ped_dis2[z]+=1
				ped_dis3[w]+=1

				if len(indices)<10:
					ped_dis5[z]+=1

print(maxx,maxy)

print(car_cnt,ped_cnt)
print(car_out_cnt, ped_out_cnt)
print(car_out_cnt/car_cnt, ped_out_cnt/ped_cnt)

"""
print()
print("==============Outlier Distribution map==================")
for i in range(xcnt):
	for j in range(ycnt):
		print(round(car_dis[i][j]/car_out_cnt,3),end="\t")
	print()

print()
for i in range(xcnt):
	for j in range(ycnt):
		print(round(ped_dis[i][j]/ped_out_cnt,3),end="\t")
	print()
"""

print()
print("============Outlier Distance Distribution===============")
for i in range(len(car_dis2)):
	print(round(car_dis2[i]/(car_out_cnt+1),3),end="\t")

print()
for i in range(len(car_dis2)):
	print(car_dis2[i],end="\t")

print()
print()
for i in range(len(ped_dis2)):
	print(round(ped_dis2[i]/(ped_out_cnt+1),3),end="\t")

print()
for i in range(len(ped_dis2)):
	print(ped_dis2[i],end="\t")


print()
print("==============Distance Outlier Ratio====================")
for i in range(len(car_dis2)):
	print(round(car_dis2[i]/(car_dis4[i]+1),3),end="\t")

print()
for i in range(len(car_dis2)):
	print(car_dis2[i],end="\t")

print()
for i in range(len(car_dis2)):
	print(car_dis4[i],end="\t")

print()
print()
for i in range(len(ped_dis2)):
	print(round(ped_dis2[i]/(ped_dis4[i]+1),3),end="\t")\

print()
for i in range(len(ped_dis2)):
	print(ped_dis2[i],end="\t")

print()
for i in range(len(ped_dis2)):
	print(ped_dis4[i],end="\t")

print()
print("==========Points inside Number Distribution=============")
for i in range(len(car_dis3)):
	print(round(car_dis3[i]/car_out_cnt,3),end="\t")

print()
for i in range(len(ped_dis3)):
	print(round(ped_dis3[i]/ped_out_cnt,3),end="\t")

print()
print("===============Pointless Outlier Ratio==================")
for i in range(len(car_dis5)):
	print(round(car_dis5[i]/(car_dis4[i]+1),3),end="\t")

print()
for i in range(len(car_dis5)):
	print(car_dis5[i],end="\t")

print()
for i in range(len(car_dis4)):
	print(car_dis4[i],end="\t")

print()
print()
for i in range(len(ped_dis5)):
	print(round(ped_dis5[i]/(ped_dis4[i]+1),3),end="\t")

print()
for i in range(len(ped_dis5)):
	print(ped_dis5[i],end="\t")

print()
for i in range(len(ped_dis4)):
	print(ped_dis4[i],end="\t")

print()
print("=====")
