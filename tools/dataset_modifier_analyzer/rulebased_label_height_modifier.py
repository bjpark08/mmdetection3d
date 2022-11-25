import pickle
import numpy as np
import copy

classnames=['Car','Pedestrian','Dont Care']

filename='data/rf2021/rf2021_infos_train'
cnt=0

#mode='compare'
mode='make'

with open(filename+'.pkl','rb') as f:
	datas=pickle.load(f)

change_pcd_cnt=100
change_pcd_cur=0
ped_change_cnt=5
ped_ex=[]

for data in datas:
    cnt+=1
    if cnt%1000==0:
        print(cnt)
    n=len(data['annos']['gt_bboxes_3d'])
    classidx=[0]*n
    abnormal=[False]*n
    normalcnt=0
    floorsum=0
    maxfloor=-100
    minfloor=100
    ped_change_cur=0
    for i in range(n):
        box=data['annos']['gt_bboxes_3d'][i]
        label=data['annos']['gt_names'][i]
        if label=='Pedestrian':
            if 1.5<=box[5]<=2:
                floorsum+=(box[2]-0.5*box[5])
                maxfloor=max(maxfloor,box[2]-0.5*box[5])
                minfloor=min(minfloor,box[2]-0.5*box[5])
                normalcnt+=1 
            elif box[5]<0.8 or box[5]>2.5:
                abnormal[i]=True
                ped_change_cur+=1
            continue
        if label=='Car':
            if 1.5<=box[5]<=3:
                floorsum+=(box[2]-0.5*box[5])
                maxfloor=max(maxfloor,box[2]-0.5*box[5])
                minfloor=min(minfloor,box[2]-0.5*box[5])
                normalcnt+=1 
            elif box[5]<1 or box[5]>4.5:
                abnormal[i]=True

    if normalcnt<5:
        continue
    floor=floorsum/normalcnt
    
    default_h=1.75
    for i in range(n):
        if abnormal[i]:
            box=data['annos']['gt_bboxes_3d'][i]
            if mode=='compare':
                newbox=copy.deepcopy(box)
                if box[2]+box[5]/2.0>floor+default_h:
                    newbox[2]=box[2]+box[5]/2.0-default_h/2.0
                elif box[2]-box[5]/2.0<floor:
                    newbox[2]=box[2]-box[5]/2.0+default_h/2.0
                else:
                    newbox[2]=floor+default_h/2.0
                newbox[5]=default_h
                data['annos']['gt_bboxes_3d']=np.append(data['annos']['gt_bboxes_3d'],newbox.reshape(1,7),axis=0)
                data['annos']['gt_names']=np.append(data['annos']['gt_names'],['Copy'],axis=0)
            else: #make
                if box[2]+box[5]/2.0>floor+default_h:
                    box[2]=box[2]+box[5]/2.0-default_h/2.0
                elif box[2]-box[5]/2.0<floor:
                    box[2]=box[2]-box[5]/2.0+default_h/2.0
                else:
                    box[2]=floor+default_h/2.0
                box[5]=default_h
    if mode=='compare' and ped_change_cur>=ped_change_cnt and change_pcd_cur<change_pcd_cnt:
        ped_ex.append(data)
        change_pcd_cur+=1

with open(filename+'_height_'+mode+'.pkl','wb') as rf:
    pickle.dump(datas,rf,protocol=pickle.HIGHEST_PROTOCOL)

if mode=='compare':
    with open(filename+'_height_compare_example.pkl','wb') as rf:
        pickle.dump(ped_ex,rf,protocol=pickle.HIGHEST_PROTOCOL)
    