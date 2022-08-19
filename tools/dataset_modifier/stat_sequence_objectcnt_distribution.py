import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

car_all=0
ped_all=0

file_name='../../data/rf2021/rf2021_infos_train'

with open(file_name+'.pkl','rb') as f:
    datas=pickle.load(f)

object_cnt=[[0,0,0,0] for i in range(3000)]
for data in datas:
    fol=int(data['lidar_points']['lidar_path'][23:28])-10002
    object_cnt[fol][0]+=1
    object_cnt[fol][3]=fol+10002
    car_cnt=0
    ped_cnt=0
    for label in data['annos']['gt_names']:
        if label=='Car':
            car_cnt+=1
        elif label=='Pedestrian':
            ped_cnt+=1
    object_cnt[fol][1]+=car_cnt
    object_cnt[fol][2]+=ped_cnt
    car_all+=car_cnt
    ped_all+=ped_cnt

for i in range(3000):
    if object_cnt[i][0]==0:
        object_cnt[i][0]=-1
    object_cnt[i][1]/=object_cnt[i][0]
    object_cnt[i][2]/=object_cnt[i][0]

object_cnt.sort(key=lambda x:x[2])

for i in range(3000):
    if object_cnt[i][0]==-1:
        continue
    print(str(object_cnt[i][3])+"\t"+str(object_cnt[i][0])+"\t"+str(round(object_cnt[i][1],2))+"\t"+str(round(object_cnt[i][2],2)))

object_cnt=np.array(object_cnt)
object_cnt=object_cnt[(object_cnt[:,0]!=-1)]
#sns.histplot(object_cnt[:,2], kde=False, rug=False, norm_hist=False)
plt.hist(object_cnt[:,1], bins=20, density=True, range=(0,100))

print(object_cnt.shape[0])
print(car_all, ped_all)
plt.show()