import torch
import torch.backends.cudnn as cudnn

from model import SleepModel
from config import config as cfg
from train import DataLoad, load_model


def init_test(test_model):
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
    model, test_data = init_test(test_model)
    model.eval()
    labels_test = test_data[:, 0].long()
    output = model(test_data[:, 1:])
    # print(output.shape)
    pred_test = output.data.max(1, keepdim=True)[1]
    # print(pred_test.shape)
    correct_test = pred_test.eq(labels_test.data.view_as(pred_test)).cpu().sum()
    # print(correct_test)
    print(f"Test Acc = {correct_test} / {labels_test.shape[0]}, Acc rate = {correct_test/labels_test.shape[0]}")

    


if __name__ == "__main__":
    test_model = "./model/sleep1/FC_700.pth"
    # test_model = "./model/FC_best.pth"

    test(test_model)

