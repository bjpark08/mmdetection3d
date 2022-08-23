import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from os import path as osp
from tqdm import tqdm 

root_path = 'data/rf2021/'

label_dir = osp.join(root_path, "NIA_2021_label", "label")

sequence_max = 3000
min_ped = 0 # 해당 seq의 평균 ped의 최소값. 예를 들어 min_ped가 1이면 ped갯수의 평균이 1미만인 seq는 학습 및 평가 데이터셋에서 제외됨.

# [seq의 scene 갯수, seq의 전체 car 갯수, seq의 전체 ped(+cyclist) 갯수, seq번호]  
object_cnt=[[0,0,0,0] for i in range(sequence_max)]
car_all=0
ped_all=0

folder_list = sorted(os.listdir(label_dir), key=lambda x:int(x))
for fol in tqdm(folder_list):
    veh_label_dir = osp.join(label_dir, fol, "car_label")
    ped_label_dir = osp.join(label_dir, fol, "ped_label")
    fol = int(fol) - 10002
    object_cnt[fol][0] = len(os.listdir(veh_label_dir))
    object_cnt[fol][3] = fol + 10002
    if osp.exists(veh_label_dir):
        for veh_file in sorted(os.listdir(veh_label_dir)):
            veh_label_file_path = osp.join(veh_label_dir, veh_file)
            if osp.exists(veh_label_file_path):
                annot = np.loadtxt(veh_label_file_path, dtype=np.object_).reshape(-1, 8)
            else: continue
            object_cnt[fol][1] += len(annot)
            car_all+=len(annot)
    if osp.exists(ped_label_dir):
        for ped_file in sorted(os.listdir(ped_label_dir)):
            ped_label_file_path = osp.join(ped_label_dir, ped_file)
            if osp.exists(ped_label_file_path):
                annot = np.loadtxt(ped_label_file_path, dtype=np.object_).reshape(-1, 6)
            else: continue
            object_cnt[fol][2] += len(annot)
            ped_all+=len(annot)

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