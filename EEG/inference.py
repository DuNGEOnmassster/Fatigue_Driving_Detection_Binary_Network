import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import pdb; pdb.set_trace()
import torch
import torch.backends.cudnn as cudnn
from model import SleepModel
from config import config as cfg
from train import DataLoad, load_model
from util.bo_classify import get_abtd

## inference
def get_infer_data(data_path):
    # all_data = DataLoad()
    # test_data = torch.from_numpy(all_data.test_data).float().to(cfg.device)
    test_data = get_abtd(data_path)

    # print(test_data.shape)


    # test_data = test_data[-1:, 1:]
    # import pdb; pdb.set_trace()

    return test_data


def pred_count(pred_test):
    count = {i: 0 for i in range(5)}
    for i, label in enumerate(pred_test):
        # print(i, label.item())
        count[label.item()] += 1
    return count


def get_fatigue_weight(count, weight_type):
    if weight_type == "Max":
        max_label = max(count, key=count.get)
        weight = max_label

    if weight_type == "Mean":
        sum_mul = 0
        for key in count.keys():
            sum_mul = sum_mul + key * count[key]
        mean_label = sum_mul / sum(count.values())
        weight = mean_label

    if weight_type == "Select":
        sort_num = 3
        sum_mul = 0
        sum_value = 0
        sort_keys = (sorted(count, key=count.get))
        # print(sort_keys)
        max_labels = sort_keys[-1*sort_num:]
        # print(max_labels)
        for key in max_labels:
            sum_mul = sum_mul + key * count[key]
            sum_value = sum_value + count[key]
        mean_select_label = sum_mul / sum_value
        weight = mean_select_label

    return weight


def inference(data_path, model):
    os.system("pwd")
    infer_data = get_infer_data(data_path)
    model.eval()
    output = model(infer_data.float())
    # print(output.shape)
    pred_test = output.data.max(1, keepdim=True)[1]
    # print(pred_test)
    count = pred_count(pred_test)
    # print(count)
    eeg_weight = get_fatigue_weight(count, "Select")
    # print(eeg_weight)
    # import pdb; pdb.set_trace()
    return eeg_weight


if __name__ == "__main__":
    test_model = "./model/FC_best.pth"
    data_path = "./data/dataOut1.mat"

    eeg_weight = inference(data_path, test_model)

