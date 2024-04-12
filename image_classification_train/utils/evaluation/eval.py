import argparse
import torch
import numpy as np
import json
from tqdm import tqdm
from .cal_mAP import json_map
from .cal_PR import json_metric, metric, json_metric_top3


voc_classes = ("aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor")


def evaluation(result, ann_path):
    print("Evaluation")
    classes = voc_classes
    aps = np.zeros(len(classes), dtype=np.float64)

    ann_json = json.load(open(ann_path, "r"))
    pred_json = result

    for i, _ in enumerate(tqdm(classes)):
        ap = json_map(i, pred_json, ann_json)
        aps[i] = ap
    OP, OR, OF1, CP, CR, CF1 = json_metric(pred_json, ann_json, len(classes))
    print("mAP: {:4f}".format(np.mean(aps)))
    print("CP: {:4f}, CR: {:4f}, CF1 :{:4F}".format(CP, CR, CF1))
    print("OP: {:4f}, OR: {:4f}, OF1 {:4F}".format(OP, OR, OF1))
    return np.mean(aps), CP, CR, CF1, OP, OR, OF1



