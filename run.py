# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eye_movement.demo import eye_movement_process
from EEG.inference import inference

def get_data(dataset_path):
    # Return real-time-updated Output.mat path in `dataset_path`
    # Return Boolean Flag `update_eeg_weight`, if Output.mat updates, Flag = True; else Flag = Flase

    return None, None


if __name__ == "__main__":
    test_model = "./EEG/model/FC_best.pth"
    dataset_path = "./EEG/data/"

    data_path, update_eeg_weight = get_data(dataset_path)
    data_path = "./EEG/data/dataOut4.mat"
    eye_movement_process(eeg_weight=inference(data_path, test_model), update_eeg_weight=update_eeg_weight, outcall=True)