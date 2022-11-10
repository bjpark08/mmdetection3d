# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.utils import print_log
from terminaltables import AsciiTable


def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (np.ndarray): Recalls with shape of (num_scales, num_dets)
            or (num_dets, ).
        precisions (np.ndarray): Precisions with shape of
            (num_scales, num_dets) or (num_dets, ).
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or np.ndarray: Calculated average precision.
    """
    if recalls.ndim == 1:
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]

    assert recalls.shape == precisions.shape
    assert recalls.ndim == 2

    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    return ap


def eval_det_cls(pred, gt, iou_thr=None, ioumode='3d', eval_aos=False, relabeling=False):
    """Generic functions to compute precision/recall for object detection for a
    single class.

    Args:
        pred (dict): Predictions mapping from image id to bounding boxes
            and scores.
        gt (dict): Ground truths mapping from image id to bounding boxes.
        iou_thr (list[float]): A list of iou thresholds.

    Return:
        tuple (np.ndarray, np.ndarray, float): Recalls, precisions and
            average precision.
    """

    # {img_id: {'bbox': box structure, 'det': matched list}}
    class_recs = {}
    npos = 0
    for img_id in gt.keys():
        cur_gt_num = len(gt[img_id])
        if cur_gt_num != 0:
            gt_cur = torch.zeros([cur_gt_num, 7], dtype=torch.float32)
            for i in range(cur_gt_num):
                gt_cur[i] = gt[img_id][i].tensor
            bbox = gt[img_id][0].new_box(gt_cur)
        else:
            bbox = gt[img_id]
        det = [[False] * len(bbox) for i in iou_thr]
        npos += len(bbox)
        class_recs[img_id] = {'bbox': bbox, 'det': det}

    # construct dets
    image_ids = []
    confidence = []
    pred_angle = []
    pred_box = []
    ious = []
    height = []
    for img_id in pred.keys():
        cur_num = len(pred[img_id])
        if cur_num == 0:
            continue
        pred_cur = torch.zeros((cur_num, 7), dtype=torch.float32)
        box_idx = 0
        for box, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            if relabeling:
                pred_box.append(box.tensor[0])
            pred_angle.append(float(box.tensor[0][6]))
            pred_cur[box_idx] = box.tensor
            box_idx += 1
        pred_cur = box.new_box(pred_cur)
        gt_cur = class_recs[img_id]['bbox']
        if len(gt_cur) > 0:
            # calculate iou in each image
            iou_cur = pred_cur.overlaps(pred_cur, gt_cur, ioumode=ioumode)
            #print(iou_cur)
            for i in range(cur_num):
                ious.append(iou_cur[i])
        else:
            for i in range(cur_num):
                ious.append(np.zeros(1))
    
    confidence = np.array(confidence)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    image_ids = [image_ids[x] for x in sorted_ind]
    ious = [ious[x] for x in sorted_ind]
    angle = [pred_angle[x] for x in sorted_ind]
    if relabeling:
        box = [pred_box[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp_thr = [np.zeros(nd) for i in iou_thr]
    fp_thr = [np.zeros(nd) for i in iou_thr]
    sim_thr = [np.zeros(nd) for i in iou_thr]
    if relabeling:
        gt_idx_thr = [-np.ones(nd) for i in iou_thr]
        gt_image_thr = [-np.ones(nd) for i in iou_thr]
        gt_box_thr = [-np.ones((nd, 7)) for i in iou_thr]
    for d in range(nd):
        R = class_recs[image_ids[d]]
        iou_max = -np.inf
        BBGT = R['bbox']
        cur_iou = ious[d]

        if len(BBGT) > 0:
            # compute overlaps
            for j in range(len(BBGT)):
                # iou = get_iou_main(get_iou_func, (bb, BBGT[j,...]))
                iou = cur_iou[j]
                if iou > iou_max:
                    iou_max = iou
                    jmax = j

        for iou_idx, thresh in enumerate(iou_thr):
            if iou_max > thresh:
                if not R['det'][iou_idx][jmax]:
                    tp_thr[iou_idx][d] = 1.
                    R['det'][iou_idx][jmax] = 1
                    if relabeling:
                        gt_idx_thr[iou_idx][d] = jmax
                        gt_image_thr[iou_idx][d] = image_ids[d]
                        gt_box_thr[iou_idx][d] = box[d]
                    if eval_aos:
                        delta=BBGT[jmax].tensor[0][6]-angle[d]
                        sim_thr[iou_idx][d] = (1.0 + np.cos(delta)) / 2.0
                else:
                    fp_thr[iou_idx][d] = 1.
            else:
                fp_thr[iou_idx][d] = 1.

    ret = []
    if relabeling:
        ret.append((gt_image_thr, gt_idx_thr, gt_box_thr))
        return ret

    # print("class:",classname)
    for iou_idx, thresh in enumerate(iou_thr):
        # compute precision recall
        fp = np.cumsum(fp_thr[iou_idx])
        tp = np.cumsum(tp_thr[iou_idx])
        recall = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = average_precision(recall, precision)
        similarity = np.cumsum(sim_thr[iou_idx]) / np.maximum(tp + fp, np.finfo(np.float64).eps)
        aos = None
        if eval_aos:
            aos = average_precision(recall, similarity)
        
        ret.append((recall, precision, ap, tp, fp, aos))

    return ret

def eval_map_recall_relabeling(pred, gt, ovthresh=(0.15,), ioumode='2d', eval_aos=False):
    ret_values = {}
    for classname in gt.keys():
        if classname in pred:
            ret_values[classname] = eval_det_cls(pred[classname],
                                                 gt[classname], ovthresh, ioumode, eval_aos, relabeling=True)
    gt_image_num = {}
    gt_image_idx = {}
    gt_image_box = {}

    for label in gt.keys():
        for iou_idx, thresh in enumerate(ovthresh):
            if label in pred:
                gt_image_num[label], gt_image_idx[label], gt_image_box[label] = ret_values[label][iou_idx]

    return gt_image_num, gt_image_idx, gt_image_box

def eval_map_recall(pred, gt, ovthresh=None, ioumode='3d', eval_aos=False):
    """Evaluate mAP and recall.

    Generic functions to compute precision/recall for object detection
        for multiple classes.

    Args:
        pred (dict): Information of detection results,
            which maps class_id and predictions.
        gt (dict): Information of ground truths, which maps class_id and
            ground truths.
        ovthresh (list[float], optional): iou threshold. Default: None.

    Return:
        tuple[dict]: dict results of recall, AP, and precision for all classes.
    """

    ret_values = {}
    for classname in gt.keys():
        if classname in pred:
            ret_values[classname] = eval_det_cls(pred[classname],
                                                 gt[classname], ovthresh, ioumode, eval_aos)
    recall = [{} for i in ovthresh]
    precision = [{} for i in ovthresh]
    ap = [{} for i in ovthresh]
    tp = [{} for i in ovthresh]
    fp = [{} for i in ovthresh]
    prec_num = [{} for i in ovthresh]
    aos = [{} for i in ovthresh] if eval_aos else None

    for label in gt.keys():
        for iou_idx, thresh in enumerate(ovthresh):
            if label in pred:
                recall[iou_idx][label], precision[iou_idx][label], ap[iou_idx][label], tp[iou_idx][label], \
                    fp[iou_idx][label], aos[iou_idx][label] = ret_values[label][iou_idx]
                gt_num = sum([len(gt[label][i]) for i in pred[label].keys()])
                tp_num = int(tp[iou_idx][label][-1])
                fp_num = int(fp[iou_idx][label][-1])
                prec_num[iou_idx][label] = tp_num/float(tp_num + fp_num)
            else:
                recall[iou_idx][label] = np.zeros(1)
                precision[iou_idx][label] = np.zeros(1)
                ap[iou_idx][label] = np.zeros(1)
                tp[iou_idx][label] = np.zeros(1)
                fp[iou_idx][label] = np.zeros(1)
                prec_num[iou_idx][label] = np.zeros(1)
                aos[iou_idx][label] = np.zeros(1)

    return recall, precision, ap, prec_num, aos

def pickle_change(pkl_path, load_interval, gts_image_num, gts_image_idx, gts_image_box, mode='mid'):
    import matplotlib.pyplot as plt

    car_change=[2,5]
    ped_change=[0,1,2,3,4,5,6]
    car=0
    ped=0
    car_all=0
    ped_all=0

    import pickle
    with open(pkl_path,'rb') as f:
        datas=pickle.load(f)
        datas=datas[::load_interval]

    #Relabeling을 하면서 이동한 거리를 histogram으로 보여줌.
    #단, 메모리 문제 방지를 위해 scene 개수가 10000개 이하일 때만 사용
    datacnt = len(datas)
    histogram = True if datacnt<=10000 else False        

    nonped_cnts=[0]*len(datas)
    for i in range(len(datas)):
        nonped_cnt=0
        for j in range(len(datas[i]['annos']['gt_names'])):
            if datas[i]['annos']['gt_names'][j]=='Pedestrian':
                ped_all+=1
            elif datas[i]['annos']['gt_names'][j]=='Car':
                nonped_cnt+=1
                car_all+=1
            else:
                nonped_cnt+=1
        nonped_cnts[i]=nonped_cnt

    #Relabeling을 하면서 이동한 거리를 histogram으로 보여줌
    if histogram:
        car_z_change=[]
        ped_x_change=[]
        ped_y_change=[]
        ped_xy_change=[]
        ped_z_change=[]

    #car 부분 relabeling. car쪽은 z,h만 수정
    for i in range(len(gts_image_num['2d'][0][0])):
        if gts_image_num['2d'][0][0][i]>=0:
            image_num=int(gts_image_num['2d'][0][0][i])
            image_idx=int(gts_image_idx['2d'][0][0][i])

            change=0
            for boxidx in range(7):
                if boxidx in car_change:
                    former_x=datas[image_num]['annos']['gt_bboxes_3d'][image_idx][boxidx]
                    pred_x=gts_image_box['2d'][0][0][i][boxidx]
                    if boxidx==2:
                        pred_x+=gts_image_box['2d'][0][0][i][5]/2.0

                    if mode=='mid':
                        x=(former_x+pred_x)/2.0
                    elif mode=='pred':
                        x=pred_x
                    datas[image_num]['annos']['gt_bboxes_3d'][image_idx][boxidx]=x
                    if former_x!=x:
                        change=1
                        #print("Caution! "+str(image_num)+"  "+str(datas[image_num]['annos']['gt_bboxes_3d'][image_idx][:3]))
                    
                    if boxidx==2 and histogram:
                        car_z_change.append(x-former_x)
            if change:
                car+=1
            
    #ped 부분 relabeling. ped쪽은 x,y,z,l,w,h,theta 전부 수정
    for i in range(len(gts_image_num['2d'][1][0])):
        if gts_image_num['2d'][1][0][i]>=0:
            image_num=int(gts_image_num['2d'][1][0][i])
            image_idx=int(gts_image_idx['2d'][1][0][i])+nonped_cnts[image_num]
            
            change=0
            for boxidx in range(7):
                if boxidx in ped_change:
                    former_x=datas[image_num]['annos']['gt_bboxes_3d'][image_idx][boxidx]
                    pred_x=gts_image_box['2d'][1][0][i][boxidx]
                    if boxidx==2:
                        pred_x+=gts_image_box['2d'][1][0][i][5]/2.0
                    if mode=='mid':
                        x=(former_x+pred_x)/2.0
                    elif mode=='pred':
                        x=pred_x
                    datas[image_num]['annos']['gt_bboxes_3d'][image_idx][boxidx]=x
                    if former_x!=x:
                        change=1
                    #print("Caution! "+str(image_num)+"  "+str(datas[image_num]['annos']['gt_bboxes_3d'][image_idx][:3]))

                    if histogram:
                        if boxidx==0:
                            ped_x_change.append(x-former_x)
                        elif boxidx==1:
                            ped_y_change.append(x-former_x)
                            ped_xy_change.append((ped_x_change[-1]**2+ped_y_change[-1]**2)**0.5)
                        elif boxidx==2:
                            ped_z_change.append(x-former_x)
            if change:
                ped+=1

    with open(pkl_path[:-4]+'_changed_'+mode+'.pkl','wb') as f:
        pickle.dump(datas,f,protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Car: "+str(car)+"/"+str(car_all)+" ("+str(round(car/car_all,2))+")")
    print("Ped: "+str(ped)+"/"+str(ped_all)+" ("+str(round(ped/ped_all,2))+")")

    #ped_y_graph는 ped_x_graph와 대칭형태일 것이므로 생략
    if histogram:
        car_z_graph=plt.subplot(2,2,1)
        ped_z_graph=plt.subplot(2,2,2)
        ped_x_graph=plt.subplot(2,2,3)
        ped_xy_graph=plt.subplot(2,2,4)

        car_z_graph.hist(car_z_change, bins=10)
        ped_z_graph.hist(ped_z_change, bins=10)
        ped_x_graph.hist(ped_x_change, bins=10)
        ped_xy_graph.hist(ped_xy_change, bins=10)

        car_z_graph.set_title('Car_z')
        ped_z_graph.set_title('Ped_z')
        ped_x_graph.set_title('Ped_x')
        ped_xy_graph.set_title('Ped_xy_distance')

        plt.subplots_adjust(hspace=0.5, wspace=0.3) 
        plt.savefig(f'{pkl_path}_relabeling.png',dpi=200)

def indoor_eval(gt_annos,
                dt_annos,
                metric,
                label2cat,
                logger=None,
                box_type_3d=None,
                box_mode_3d=None,
                classes=None,
                pkl_path=None,
                relabeling=False,
                load_interval=1):
    """Indoor Evaluation.

    Evaluate the result of the detection.

    Args:
        gt_annos (list[dict]): Ground truth annotations.
        dt_annos (list[dict]): Detection annotations. the dict
            includes the following keys

            - labels_3d (torch.Tensor): Labels of boxes.
            - boxes_3d (:obj:`BaseInstance3DBoxes`):
                3D bounding boxes in Depth coordinate.
            - scores_3d (torch.Tensor): Scores of boxes.
        metric (list[float]): IoU thresholds for computing average precisions.
        label2cat (dict): Map from label to category.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.

    Return:
        dict[str, float]: Dict of results.
    """

    class_names = classes
    while len(class_names)<3:
        class_names.append('Dont Care')

    assert len(dt_annos) == len(gt_annos)
    pred = {}  # map {class_id: pred}
    gt = {}  # map {class_id: gt}

    for img_id in range(len(dt_annos)): ## gt = ground truth ## dt = d
        # parse detected annotations
        det_anno = dt_annos[img_id]
        if 'pts_bbox' in det_anno.keys():
            det_anno = det_anno['pts_bbox']
        for i in range(len(det_anno['labels_3d'])):
            label = det_anno['labels_3d'].numpy()[i]
            bbox = det_anno['boxes_3d'].convert_to(box_mode_3d)[i]
            score = det_anno['scores_3d'].numpy()[i]
            if label not in pred:
                pred[int(label)] = {}
            if img_id not in pred[label]:
                pred[int(label)][img_id] = []
            if label not in gt:
                gt[int(label)] = {}
            if img_id not in gt[label]:
                gt[int(label)][img_id] = []
            pred[int(label)][img_id].append((bbox, score))

        # parse gt annotations
        gt_anno = gt_annos[img_id]
        if len(gt_anno['gt_bboxes_3d']) != 0:
            gt_boxes = box_type_3d(
                gt_anno['gt_bboxes_3d'],
                box_dim=gt_anno['gt_bboxes_3d'].shape[-1],
                origin=(0.5, 0.5, 0.5)).convert_to(box_mode_3d)
            labels_3d = gt_anno['gt_names']
        else:
            gt_boxes = box_type_3d(np.array([], dtype=np.float32))
            labels_3d = np.array([], dtype=np.int64)


        for i in range(len(labels_3d)):
            label = 0 if labels_3d[i]==class_names[0] else 1 if labels_3d[i]==class_names[1] else 2
            bbox = gt_boxes[i]
            if label not in gt:
                gt[label] = {}
            if img_id not in gt[label]:
                gt[label][img_id] = []
            gt[label][img_id].append(bbox)

    eval_aos = True
    ioumodes = ['3d', '2d', 'dis']
    ret_dict_ioumodes = dict()

    if relabeling:
        print()
        print("==============RELABELING==============")
        ret_dict = dict()
        gts_image_num = {}
        gts_image_idx = {}
        gts_image_box = {}
        gts_image_num['2d'], gts_image_idx['2d'], gts_image_box['2d'] = \
                    eval_map_recall_relabeling(pred, gt, ovthresh=(0.15,), ioumode='2d', eval_aos=False)
        print(pkl_path)
        pickle_change(pkl_path, load_interval, gts_image_num, gts_image_idx, gts_image_box, mode='pred')
        return ret_dict_ioumodes

    for ioumode in ioumodes:
        cur_metric=metric
        if ioumode =='dis':
            cur_metric = (0.5,)

        rec, prec, ap, prec_num, aos = eval_map_recall(pred, gt, cur_metric, ioumode, eval_aos)
        
        ret_dict = dict()
        header = ['classes /'+ioumode]
        table_columns = [[class_names[0] if label == 0 else class_names[1] if label == 1 else class_names[2]
                        for label in ap[0].keys()] + ['Overall']]

        for i, iou_thresh in enumerate(cur_metric):
            header.append(f'AP_{iou_thresh:.2f}')
            header.append(f'Recall_{iou_thresh:.2f}')
            header.append(f'Precision_{iou_thresh:.2f}')
            if eval_aos:
                header.append(f'AOS_{iou_thresh:.2f}')

            rec_list = []
            prec_list = []
            for label in ap[i].keys():
                label_cls = class_names[0] if label == 0 else class_names[1] if label == 1 else class_names[2]
                ret_dict[f'{label_cls}_AP_{iou_thresh:.2f}'] = float(
                    ap[i][label][0])
            ret_dict[f'mAP_{iou_thresh:.2f}'] = float(
                np.mean(list(ap[i].values())))

            table_columns.append(list(map(float, list(ap[i].values()))))
            table_columns[-1] += [ret_dict[f'mAP_{iou_thresh:.2f}']]
            table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]
            
            for label in rec[i].keys():
                label_cls = class_names[0] if label == 0 else class_names[1] if label == 1 else class_names[2]
                ret_dict[f'{label_cls}_rec_{iou_thresh:.2f}'] = float(
                    rec[i][label][-1])
                rec_list.append(rec[i][label][-1])
            ret_dict[f'mRecall_{iou_thresh:.2f}'] = float(np.mean(rec_list))

            table_columns.append(list(map(float, rec_list)))
            table_columns[-1] += [ret_dict[f'mRecall_{iou_thresh:.2f}']]
            table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]
            
            for label in prec_num[i].keys():
                label_cls = class_names[0] if label == 0 else class_names[1] if label == 1 else class_names[2]
                ret_dict[f'{label_cls}_prec_{iou_thresh:.2f}'] = prec_num[i][label]
                prec_list.append(prec_num[i][label])
            ret_dict[f'mPrecision_{iou_thresh:.2f}'] = float(np.mean(prec_list))
            table_columns.append(list(map(float, prec_list)))
            table_columns[-1] += [ret_dict[f'mPrecision_{iou_thresh:.2f}']]
            table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

            if eval_aos:
                aos_list = []
                for label in aos[i].keys():
                    label_cls = class_names[0] if label == 0 else class_names[1] if label == 1 else class_names[2]
                    ret_dict[f'{label_cls}_aos_{iou_thresh:.2f}'] = aos[i][label]
                    aos_list.append(aos[i][label])
                ret_dict[f'mAOS_{iou_thresh:.2f}'] = float(np.mean(aos_list))
                table_columns.append(list(map(float, aos_list)))
                table_columns[-1] += [ret_dict[f'mAOS_{iou_thresh:.2f}']]
                table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]
        table_data = [header]
        table_rows = list(zip(*table_columns))
        table_data += table_rows
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)
        ret_dict_ioumodes[ioumode]=ret_dict

    # 개념적으로는 이게 맞는데 이걸 리턴하면 학습에 문제가 생김.
    # return ret_dict_ioumodes
    return ret_dict
