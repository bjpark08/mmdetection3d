import pickle
import numpy as np
import copy

classnames=['Car','Pedestrian','Dont Care']

filename='../../data/rf2021/rf2021_infos_train'
cnt=0
inclined_cnt=0

#mode='compare'
mode='make'

with open(filename+'.pkl','rb') as f:
	datas=pickle.load(f)
new_datas=[]

for i in range(len(datas)):
    cnt+=1
    if cnt%1000==0:
        print(cnt)

    data=datas[i]
    n=len(data['annos']['gt_bboxes_3d'])
    floors=[]
    maxfloor=-100
    minfloor=100
    for i in range(n):
        box=data['annos']['gt_bboxes_3d'][i]
        label=data['annos']['gt_names'][i]
        if label=='Pedestrian':
            continue
        if 1.5<=box[5]<=3:
            floors.append(box[2]-0.5*box[5])
            maxfloor=max(maxfloor,box[2]-0.5*box[5])
            minfloor=min(minfloor,box[2]-0.5*box[5])

    if maxfloor-minfloor>3:
        inclined_cnt+=1
        new_datas.append(data)
        #print(data['lidar_points']['lidar_path']+"     "+str(round(maxfloor,2))+"    "+str(round(minfloor,2)))

print("Inclined Portion : "+str(round(inclined_cnt/cnt,2)))

with open(filename+'_inclined.pkl','wb') as rf:
    pickle.dump(new_datas,rf,protocol=pickle.HIGHEST_PROTOCOL)