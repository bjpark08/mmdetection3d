from curses.ascii import EM
from math import nan
import pickle
from queue import Empty
import numpy as np

car_thres = 0.2
ped_thres = 0.2
# sampling = 100

def calculate_scores(datas):
    car_score = []
    ped_score = []
    car_bbox_size = []
    ped_bbox_size = []
    car_num = 0
    ped_num = 0

    for data in datas:
        scores = data['pts_bbox']['scores_3d'].numpy()
        bbox = data['pts_bbox']['boxes_3d'].info.numpy()
        label = data['pts_bbox']['labels_3d'].numpy()
        
        # 크기 위치에 따라 Prediction Score를 분석하는 부분
        big_car = scores[np.where((bbox[:, 4] > 10) & (label == 0))]
        big_car_size = np.absolute(bbox[np.where((abs(bbox[:, 4]) > 10) & (label == 0))])
        big_ped = scores[np.where((bbox[:, 5] > 1.5) & (label == 1))]
        big_ped_size = np.absolute(bbox[np.where((abs(bbox[:, 5]) > 1.5) & (label == 1))])

        # valid_car_size = np.absolute(bbox[np.where((scores<car_thres) & (label == 0))])
        # valid_ped_size = np.absolute(bbox[np.where((scores<ped_thres) & (label == 1))])

        # closed_car_size = np.absolute(bbox[np.where((abs(bbox[:, 0]) < 10) & (label == 0))])
        # closed_ped_size = np.absolute(bbox[np.where((abs(bbox[:, 0]) < 10) & (label == 1))])

        # car_idx = np.where((scores<car_thres) & (label == 0))
        # ped_idx = np.where((scores<ped_thres) & (label == 1))
        # car_num = car_num + len(car_idx[0])
        # ped_num  = ped_num + len(ped_idx[0])
        
        car_idx = np.where((np.absolute(bbox[:, 1]) < 20.0) & (label == 0))
        ped_idx = np.where((np.absolute(bbox[:, 1]) < 20.0) & (label == 1))
        car_num = car_num + len(car_idx[0])
        ped_num  = ped_num + len(ped_idx[0])

        if(len(big_car) != 0): car_score.append(big_car.mean())
        # if(len(high_ped) != 0): ped_score.append(high_ped.mean())
        if(len(big_car_size) != 0): car_bbox_size.append(big_car_size.mean(axis = 0)) 
        # if(len(high_ped_size) != 0): ped_bbox_size.append(high_ped_size.mean(axis = 0)) 


    return car_num, ped_num, car_score, ped_score, car_bbox_size, ped_bbox_size



if __name__ == '__main__':
    filepath = './data/third_results.pkl' # tools/test에서 --format-only option을 활용하여 만든 pkl파일을 사용하면 된다
    
    with open(filepath,'rb') as f:
        datas=pickle.load(f)

    car_num, ped_num, car_score, ped_score, car_bbox_size, ped_bbox_size = calculate_scores(datas)

    car_score = np.array(car_score)
    ped_score = np.array(ped_score)
    car_bbox_size = np.array(car_bbox_size)
    ped_bbox_size = np.array(ped_bbox_size)

    # print("car num", car_num)
    # print("ped num", ped_num)

    # print("car score", car_score.mean())
    # print("ped score", ped_score.mean())

    # print("car size", np.mean(car_bbox_size, axis = 0))
    # print("ped size", np.mean(ped_bbox_size, axis = 0))
