from EEG.inference import inference
from eye_movement.demo import eye_movement_process


def get_data(dataset_path):
    # Return real-time-updated Output.mat path in `dataset_path`
    # Return Boolean Flag `update_eeg_weight`, if Output.mat updates, Flag = True; else Flag = Flase
    return None, None


if __name__ == "__main__":
    test_model = "./EEG/model/FC_best.pth"
    dataset_path = "./data"

    # TODO Slove import path problem

    data_path, update_eeg_weight = get_data(dataset_path)
    eye_movement_process(inference(data_path, test_model), update_eeg_weight)