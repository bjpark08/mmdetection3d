from curses.ascii import EM
from math import nan
import pickle
from queue import Empty
import numpy as np

car_thres = 0.0
ped_thres = 0.7

def calculate_scores(datas):
    car_score = []
    ped_score = []
    car_bbox_size = []
    ped_bbox_size = []
    idx = 0
    car_num = 0
    ped_num = 0
    flag = 0
    for data in datas:
        bbox = data['gt_bboxes_3d']
        label = data['gt_labels_3d']

        high_car_size = np.absolute(bbox[np.where((label == 0))])
        high_ped_size = np.absolute(bbox[np.where((label == 1))])

        if(len(high_car_size) != 0): car_bbox_size.append(high_car_size.mean(axis = 0))
        if(len(high_ped_size) != 0): ped_bbox_size.append(high_ped_size.mean(axis = 0))


    return car_bbox_size, ped_bbox_size



if __name__ == '__main__':
    filepath = './data/sampler.pkl'
    
    with open(filepath,'rb') as f:
        datas=pickle.load(f)

    car_bbox_size, ped_bbox_size = calculate_scores(datas)
    car_bbox_size = np.array(car_bbox_size)
    ped_bbox_size = np.array(ped_bbox_size)

    print("car size", np.mean(car_bbox_size, axis = 0))
    print("ped size", np.mean(ped_bbox_size, axis = 0))
    
    
    # with open('./data/rf2021/rf2021_infos_train_only_ped'+str(ped_cnt)+'.pkl','wb') as rf:
    #     pickle.dump(ped_data,rf,protocol=pickle.HIGHEST_PROTOCOL)
