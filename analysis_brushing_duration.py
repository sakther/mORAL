from ML.feature_engineering import transfer_Scaler
from models.AdaBoost import evaluate_adaboost
from models.DecisionTree import evaluate_DT
from models.NeiveBayes import evaluate_neive_bayes
from models.VotingClassifier import evaluate_voting
from models.single_random_forest import *
from models.ensemble_multiple_randomForest import *
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from utils.file_utils import *
from input.import_feature_and_label_file import *
from models.smooth_outputs_and_constract_episodes import *
from ML.feature_selestion import *
import seaborn as sns

font = {'family': 'calibri',
        'weight': 'bold',
        'size': 22}

matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42


def count_positive_events(pids):
    for pid_test in pids:
        if pid_test == 'b0e8':
            continue
        # T_test, X_test, Y_test = get_XY_pos_neg_sep(feature_dir, pid_test)
        feature_names, T_test, X_test, Y_test = get_XY(pid_test)
        print('For ', pid_test, 'Total:', len(list(X_test)), ', #pos', sum(Y_test))


def analyse_duration_dist(pids, skipped_pids):
    print("cross_subject_validation(pids)", pids)

    X_train = []
    Y_train = []

    for pid_train in pids:

        if pid_train in skipped_pids:
            continue
        fn, T_tn, X_tn, Y_tn = get_XY(pid_train)
        if sum(Y_tn) < 3:
            continue

        X_train.extend(list(X_tn))
        Y_train.extend(list(Y_tn))

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    posDur = [v[0] for i, v in enumerate(X_train) if Y_train[i] == 1]
    negDur = [v[0] for i, v in enumerate(X_train) if Y_train[i] == 0]

    sns.distplot(posDur, bins=100, label='Pos')
    sns.distplot(negDur, bins=1000, label='Neg')
    plt.legend()
    plt.show()
    plt.close()


pids = [d for d in os.listdir(sensor_data_dir) if os.path.isdir(os.path.join(sensor_data_dir, d)) and len(d) == 4]

skipped_pids = ['8337', 'a764', 'aeb6', 'b0e8']

# count_positive_events(pids)
analyse_duration_dist(pids, skipped_pids)

# -------------------
# For  820c Total: 871 , #pos 7
# For  b15a Total: 1126 , #pos 5
# For  9a6b Total: 1086 , #pos 6
# For  b27d Total: 600 , #pos 11
# For  b67b Total: 505 , #pos 4
# For  813f Total: 1299 , #pos 9
# For  896d Total: 135 , #pos 1
# For  93a2 Total: 590 , #pos 6
# For  a764 Total: 683 , #pos 4
# For  94c0 Total: 993 , #pos 8
# For  8337 Total: 618 , #pos 6
# For  a64e Total: 1230 , #pos 10
# For  9eee Total: 872 , #pos 7
# For  aebb Total: 781 , #pos 9
# For  89f4 Total: 50 , #pos 1
# For  86bd Total: 632 , #pos 3
# For  a153 Total: 1137 , #pos 6
# For  b0e8 Total: 425 , #pos 0
# For  9e33 Total: 655 , #pos 6
# For  be71 Total: 520 , #pos 4
# For  891e Total: 1346 , #pos 7
# For  aa22 Total: 379 , #pos 3
# For  999e Total: 1428 , #pos 15
