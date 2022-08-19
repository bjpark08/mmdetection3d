import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

root_path = '../../data/rf2021/'
file_names=[
    '../../data/rf2021/rf2021_infos_train_height_make',
    '../../data/rf2021/rf2021_infos_val_height_make',
    '../../data/rf2021/rf2021_infos_test_height_make',
]

sequence_max = 3000
min_ped = 4

object_cnt=[[0,0,0,0] for i in range(sequence_max)]
car_all=0
ped_all=0

for file_name in file_names:
    with open(file_name+'.pkl','rb') as f:
        datas=pickle.load(f)
    
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

for i in range(sequence_max):
    if object_cnt[i][0]==0 or object_cnt[i][2]<object_cnt[i][0]*min_ped:
        object_cnt[i][0]=-1
        continue
    object_cnt[i][1]/=object_cnt[i][0]
    object_cnt[i][2]/=object_cnt[i][0]

object_cnt.sort(key=lambda x:x[2])

for i in range(sequence_max):
    if object_cnt[i][0]==-1:
        continue
    print(str(object_cnt[i][3])+"\t"+str(object_cnt[i][0])+"\t"+str(round(object_cnt[i][1],2))+"\t"+str(round(object_cnt[i][2],2)))

object_cnt=np.array(object_cnt)
object_cnt=object_cnt[(object_cnt[:,0]!=-1)]
#sns.histplot(object_cnt[:,2], kde=False, rug=False, norm_hist=False)
plt.hist(object_cnt[:,1], bins=20, density=True, range=(0,100))

all_set=[]
train_set=[]
val_set=[]
test_set=[]

for i in range(object_cnt.shape[0]):
    all_set.append(int(object_cnt[i][3]))
    if i%10<=7:
        train_set.append(int(object_cnt[i][3]))
    elif i%10==8:
        val_set.append(int(object_cnt[i][3]))
    else:
        test_set.append(int(object_cnt[i][3]))

all_set.sort()
train_set.sort()
val_set.sort()
test_set.sort()

print(object_cnt.shape[0])
print(str(len(train_set))+" "+str(len(val_set))+" "+str(len(test_set)))
print(train_set[:10])
print(val_set[:10])
print(test_set[:10])
print(car_all, ped_all)
#plt.show()

if not os.path.exists(root_path + 'sequence_set_ped_'+str(min_ped)):
    os.makedirs(root_path + 'sequence_set_ped_'+str(min_ped))

with open(root_path + 'sequence_set_ped_'+str(min_ped)+'/sequence_train_set.pkl','wb') as rf:
	pickle.dump(train_set,rf)

with open(root_path + 'sequence_set_ped_'+str(min_ped)+'/sequence_val_set.pkl','wb') as rf:
	pickle.dump(val_set,rf)

with open(root_path + 'sequence_set_ped_'+str(min_ped)+'/sequence_test_set.pkl','wb') as rf:
	pickle.dump(test_set,rf)