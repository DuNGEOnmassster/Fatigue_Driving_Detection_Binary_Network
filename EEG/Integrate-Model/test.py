import torch
import torch.backends.cudnn as cudnn

from model import SleepModel
from config import config as cfg
from train import DataLoad, load_model


## test
def get_test_data(test_model):
    all_data = DataLoad()
    test_data = torch.from_numpy(all_data.test_data).float().to(cfg.device)
    # print(test_data.shape)
    model = SleepModel(5, is_training=True)

    model = model.to(cfg.device)
    if cfg.cuda:
        cudnn.benchmark = True

    if cfg.resume:
        load_model(model, test_model)

    return model, test_data


def test(test_model):
    model, test_data = get_test_data(test_model)
    model.eval()
    labels_test = test_data[:, 0].long()
    output = model(test_data[:, 1:])
    # print(output.shape)
    pred_test = output.data.max(1, keepdim=True)[1]
    # print(pred_test.shape)
    correct_test = pred_test.eq(labels_test.data.view_as(pred_test)).cpu().sum()
    # print(correct_test)
    print(f"Test Acc = {correct_test} / {labels_test.shape[0]}, Acc rate = {correct_test/labels_test.shape[0]}")


## inference
def get_infer_data(test_model):
    all_data = DataLoad()
    test_data = torch.from_numpy(all_data.test_data).float().to(cfg.device)
    # print(test_data.shape)
    model = SleepModel(5, is_training=True)

    model = model.to(cfg.device)
    if cfg.cuda:
        cudnn.benchmark = True

    if cfg.resume:
        load_model(model, test_model)

    test_data = test_data[:, 1:]

    return model, test_data


def pred_count(pred_test):
    count = {i: 0 for i in range(5)}
    for i, label in enumerate(pred_test):
        # print(i, label.item())
        count[label.item()] += 1
    return count


def trans_fatigue_level(count):
    max_label = max(count, key=count.get)



def inference(test_model):
    model, infer_data = get_infer_data(test_model)
    model.eval()
    output = model(infer_data)
    # print(output.shape)
    pred_test = output.data.max(1, keepdim=True)[1]
    # print(pred_test)
    count = pred_count(pred_test)
    # print(count)
    trans_fatigue_level(count)


if __name__ == "__main__":
    # test_model = "./model/sleep1/FC_700.pth"
    test_model = "./model/FC_best.pth"

    # test(test_model)
    inference(test_model)

