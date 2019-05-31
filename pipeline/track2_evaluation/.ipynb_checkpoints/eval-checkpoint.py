from __future__ import division
import csv

def compute_ap(predict, gt_tracklets, track_test):
    """
    params
        predict: ranklist - list of image
        gt_tracklets: list of tracklets
        track_test: dict {tracklet: list of image in tracklet}
    return
        ap
    """

    # get list of image in ground truth
    gt = list() 
    for tracklet in gt_tracklets:
        gt.append(track_test[tracklet])
    gt = gt[0]

    # true positive
    true_positive = 0

    sum_precision = 0

    for i, p in enumerate(predict):
        # check if current predict image in ground truth
        if p in gt:
            true_positive += 1
            # precision = true positive / number of samples
            sum_precision += true_positive/(i+1)

    return sum_precision / len(gt)

def read_text_to_dict(filename):
    """
    read file to dict {line_id : list of int value} 
    line_id index from 0

    return: dict
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        fdict = {i: row for i, row in enumerate(reader)}
    norm_dict = {}
    for key, val in fdict.items():
        str_val = list(filter(lambda a: a != '', val))
        norm_dict[key] = list(map(int, str_val))
    return norm_dict 

def compute_mAP():
    predicts = read_text_to_dict('track2.txt') 
    track_test = read_text_to_dict('track2_evaluation/test_track_id.txt')
    ground_truth = read_text_to_dict('track2_evaluation/all_gt.txt')
    sum_ap = 0
    n_gt = 0
    for i in ground_truth:
        if ground_truth[i]:
            n_gt += 1
            ap = compute_ap(predicts[i], ground_truth[i], track_test)
            sum_ap += ap

    mAP = sum_ap / n_gt
    return mAP

if __name__ == '__main__':
    mAP = compute_ap()
    print(mAP)



