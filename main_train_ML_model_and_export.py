from models.AdaBoost import evaluate_adaboost, get_trained_adaboost
from utils.file_utils import *
from input.import_feature_and_label_file import *


def train_and_save_AB_model(pids, skipped_pids, model_filename='trained_model_files/brushingAB.model'):
    Xs = []
    Ys = []

    for pid in pids:
        if pid in skipped_pids:
            continue

        fn, T_tn, X_tn, Y_tn = get_XY(pid)

        Xs.extend(list(X_tn))
        Ys.extend(list(Y_tn))
    clf_AB = get_trained_adaboost(Xs, Ys)
    save_model_file(clf_AB, model_filename)
    print('Trained model saved...')


if __name__ == "__main__":
    pids = [d for d in os.listdir(sensor_data_dir) if os.path.isdir(os.path.join(sensor_data_dir, d)) and len(d) == 4]

    skipped_pids = ['8337', 'a764', 'aebb', 'b0e8']
    model_filename = 'trained_model_files/brushingAB.model'

    train_and_save_AB_model(pids, skipped_pids, model_filename='trained_model_files/brushingAB.model')
