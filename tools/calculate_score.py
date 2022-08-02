from curses.ascii import EM
from math import nan
import pickle
from queue import Empty
import numpy as np

car_thres = 0.2
ped_thres = 0.2
sampling = 100

def calculate_scores(datas):
    car_score = []
    ped_score = []
    car_bbox_size = []
    ped_bbox_size = []
    idx = 0
    car_num = 0
    ped_num = 0

    for data in datas:
        if(idx != sampling):
            idx = idx + 1
            continue
        idx = 0
        scores = data['pts_bbox']['scores_3d'].numpy()
        bbox = data['pts_bbox']['boxes_3d'].info.numpy()
        label = data['pts_bbox']['labels_3d'].numpy()
        


        high_car = scores[np.where((scores<car_thres) & (label == 0))]
        high_ped = scores[np.where((scores<ped_thres) & (label == 1))]


        high_car_size = np.absolute(bbox[np.where((scores<car_thres) & (label == 0))])
        high_ped_size = np.absolute(bbox[np.where((scores<ped_thres) & (label == 1))])

        # high_car_size = np.absolute(bbox[np.where((abs(bbox[:, 0]) < 10) & (label == 0))])
        # high_ped_size = np.absolute(bbox[np.where((abs(bbox[:, 0]) < 10) & (label == 1))])

        # car_idx = np.where((scores<car_thres) & (label == 0))
        # ped_idx = np.where((scores<ped_thres) & (label == 1))
        # car_num = car_num + len(car_idx[0])
        # ped_num  = ped_num + len(ped_idx[0])
        
        car_idx = np.where((bbox[:, 0] < 10.0) & (label == 0))
        ped_idx = np.where((bbox[:, 1] < 20.0) & (label == 1))
        car_num = car_num + len(car_idx[0])
        ped_num  = ped_num + len(ped_idx[0])

        if(len(high_car) != 0): car_score.append(high_car.mean())
        if(len(high_ped) != 0): ped_score.append(high_ped.mean())
        if(len(high_car_size) != 0): car_bbox_size.append(high_car_size.mean(axis = 0)) 
        if(len(high_ped_size) != 0): ped_bbox_size.append(high_ped_size.mean(axis = 0)) 


    return car_num, ped_num, car_score, ped_score, car_bbox_size, ped_bbox_size



if __name__ == '__main__':
    filepath = './data/results.pkl'
    
    with open(filepath,'rb') as f:
        datas=pickle.load(f)

    car_num, ped_num, car_score, ped_score, car_bbox_size, ped_bbox_size = calculate_scores(datas)

    car_score = np.array(car_score)
    ped_score = np.array(ped_score)
    car_bbox_size = np.array(car_bbox_size)
    ped_bbox_size = np.array(ped_bbox_size)

    print("car num", car_num)
    print("ped num", ped_num)

    print("car score", car_score.mean())
    print("ped score", ped_score.mean())
    print("car size", np.mean(car_bbox_size, axis = 0))
    print("ped size", np.mean(ped_bbox_size, axis = 0))

    
    # with open('./data/rf2021/rf2021_infos_train_only_ped'+str(ped_cnt)+'.pkl','wb') as rf:
    #     pickle.dump(ped_data,rf,protocol=pickle.HIGHEST_PROTOCOL)
