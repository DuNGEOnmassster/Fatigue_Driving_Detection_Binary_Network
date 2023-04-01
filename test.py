# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eye_movement.demo import eye_movement_process
from EEG.inference import inference
from EEG.train import load_model
from EEG.model import SleepModel
from EEG.config import config as cfg

import torch.backends.cudnn as cudnn

def get_data(dataset_path):
    # Return real-time-updated Output.mat path in `dataset_path`
    # Return Boolean Flag `update_eeg_weight`, if Output.mat updates, Flag = True; else Flag = Flase

    return None, None


def prepare_model(test_model):
    model = SleepModel(5, is_training=True)

    model = model.to(cfg.device)
    if cfg.cuda:
        cudnn.benchmark = True

    if cfg.resume:
        load_model(model, test_model)
    
    return model


if __name__ == "__main__":
    test_model = "./EEG/model/FC_best.pth"
    dataset_path = "./EEG/data/"

    data_path, update_eeg_weight = get_data(dataset_path)
    model = prepare_model(test_model)
    data_path = "./EEG/data/dataOut4.mat"
    eye_movement_process(inference_func=inference, dataset_path=dataset_path, model=model, update_eeg_weight=update_eeg_weight, outcall=True)