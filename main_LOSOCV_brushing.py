from ML.feature_engineering import transfer_Scaler
from models.AdaBoost import evaluate_adaboost, get_trained_adaboost
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

font = {'family': 'calibri',
        'weight': 'bold',
        'size': 22}

matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42


def get_selected_features(X, Y):
    w0 = len([v for v in Y if v == 0]) / len(list(Y))
    w1 = len([v for v in Y if v == 1]) / len(list(Y))
    clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=100, class_weight={0: w1, 1: w0})
    clf.fit(X, Y)

    importances = list(clf.feature_importances_)

    feature_indexes = [i for i, importance in enumerate(importances) if importance > 0.001]

    return feature_indexes


def get_recal_precision_f1_accuracy_CM(Y_test, Y_ts_pred):
    cm = confusion_matrix(np.array(Y_test), np.array(Y_ts_pred))
    # print(cm)
    if (cm[1][1] + cm[0][1]) == 0:
        pp = 0
    else:
        pp = cm[1][1] / (cm[1][1] + cm[0][1])
    if (cm[1][1] + cm[1][0]) == 0:
        rr = 0
    else:
        rr = cm[1][1] / (cm[1][1] + cm[1][0])
    if rr == 0 or pp == 0:
        f1 = 0
    else:
        f1 = (2 * rr * pp) / (rr + pp)
    acc = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    false_positives = cm[0][1]

    return [rr, pp, f1, acc, false_positives]


def count_positive_events(pids):
    for pid_test in pids:
        if pid_test == 'b0e8':
            continue
        # T_test, X_test, Y_test = get_XY_pos_neg_sep(feature_dir, pid_test)
        feature_names, T_test, X_test, Y_test = get_XY(pid_test)
        print('For ', pid_test, 'Total:', len(list(X_test)), ', #pos', sum(Y_test))


def plot_results(Res, modelnames, saved_result_filename='RES_LOSOCV_brushing.pckl',
                 plot_output_filename='LOSOCV_result_brushing_event_FS_gt3.pdf'):
    models = []
    recalls = []
    precisions = []
    f1s = []
    accs = []
    FPs = []
    for curRes_smooth in Res:
        for i, v in enumerate(modelnames):
            curM = curRes_smooth[i]
            models.append(v)
            recalls.append(curM[0])
            precisions.append(curM[1])
            f1s.append(curM[2])
            accs.append(curM[3])
            FPs.append(curM[4])

    print('--total false positive', sum(FPs))
    f = plt.figure()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12, 10)

    df = pd.DataFrame({'Models': models, 'Recall': recalls, 'Precision': precisions, 'F1': f1s})
    df = df[['Models', 'Recall', 'Precision', 'F1']]
    print(df)
    res_modelwise = pd.melt(df, id_vars=['Models'], value_vars=['Recall', 'Precision', 'F1'], var_name='Metrics')

    sns.boxplot(x='Models', y='value', data=res_modelwise, hue='Metrics', width=0.8)

    # plt.ylim([0.4, 1])
    plt.xticks(rotation=45)
    plt.tight_layout()

    f.savefig(result_dir + plot_output_filename + '.png', bbox_inches='tight')
    f.savefig(result_dir + plot_output_filename+ '.pdf', bbox_inches='tight')
    plt.close()
    # with open(result_dir + 'RES_LOSOCV_brushing_FS.pckl', 'wb') as handle:
    with open(result_dir + saved_result_filename, 'wb') as handle:
        pickle.dump(df, handle)
    # else:
    #     f.savefig(result_dir + 'LOSOCV_result_brushing_event_gt3.pdf', bbox_inches='tight')
    #     # with open(result_dir + 'RES_LOSOCV_brushing.pckl', 'wb') as handle:
    #     with open(result_dir + saved_result_filename, 'wb') as handle:
    #         pickle.dump(df, handle)
    return res_modelwise

def cross_subject_validation(pids, do_feature_selection=True, do_smooth=True, do_save_model_outputs=True):
    modelnames = ['Gaussian Neive Bayes', 'Random Forest 100', 'FR 100 with smoothing', 'ensemble 50 RFs',
                  'ensemble 50 RFs with smoothing', 'Ada-boost', 'Voting']
    modelnames = ['NB', "CART", 'RF-100', 'RF-1000', 'Ens', 'AB', 'Voting']

    print("cross_subject_validation(pids)", pids)

    cnt = 0

    Res = []
    Res_smooth = []
    weights = []
    for pid_test in pids:
        # if pid_test in ['aebb', 'b0e8']:
        #     continue
        if pid_test in ['b0e8']:
            continue
        # T_test, X_test, Y_test = get_XY_pos_neg_sep(feature_dir, pid_test)
        feature_names, T_test, X_test, Y_test = get_XY(pid_test)

        X_train = []
        Y_train = []

        for pid_train in pids:
            # if pid_train in ['aebb', 'b0e8']:
            #     continue
            if pid_train in ['b0e8']:
                continue
            if pid_train == pid_test:
                continue
            # T_tn, X_tn, Y_tn = get_XY_pos_neg_sep(feature_dir, pid_train)
            fn, T_tn, X_tn, Y_tn = get_XY(pid_train)

            X_train.extend(list(X_tn))
            Y_train.extend(list(Y_tn))
        print('Train labels:', set(Y_train))
        print('Test labels:', set(Y_test))
        print('--------------For ', pid_test, 'train size', len(list(X_train)), 'train pos', sum(Y_train), 'test size',
              len(list(X_test)), 'test pos', sum(Y_test))
        if sum(Y_test) <= 3:
            continue

        cnt += 1

        X_train = np.array(X_train)
        X_test = np.array(X_test)

        # X_train, X_test = transfer_Scaler(X_train, X_test)

        if do_feature_selection:
            # fs_CFS = FCBF()
            # fs_CFS = RF_FS()
            # X_train = fs_CFS.fit_transform(X_train, Y_train)
            # X_test = fs_CFS.transform(X_test)
            feature_ind = get_selected_features(X_train, Y_train)
            X_train = X_train[:, feature_ind]
            X_test = X_test[:, feature_ind]
        #
        Y_ts_NB_init = evaluate_neive_bayes(X_train, Y_train, X_test, Y_test, T_test)
        Y_ts_CART_init = evaluate_DT(X_train, Y_train, X_test, Y_test, T_test)
        Y_ts_RF100_init = evaluate_single_random_forest(X_train, Y_train, X_test, Y_test, T_test, 100)
        Y_ts_RF1000_init = evaluate_single_random_forest(X_train, Y_train, X_test, Y_test, T_test, 1000)
        Y_ts_pred_enemble_init = evaluate_ensemble_multiple_random_forest(X_train, Y_train, X_test, Y_test, T_test)
        Y_ts_pred_adaboost_init = evaluate_adaboost(X_train, Y_train, X_test, Y_test, T_test)
        Y_ts_pred_voting_init = evaluate_voting(X_train, Y_train, X_test, Y_test, T_test)

        D_test = list(X_test[:, 0])
        Y_ts_NB = smooth_output(T_test, D_test, Y_ts_NB_init)
        Y_ts_CART = smooth_output(T_test, D_test, Y_ts_CART_init)
        Y_ts_RF100 = smooth_output(T_test, D_test, Y_ts_RF100_init)
        Y_ts_RF1000 = smooth_output(T_test, D_test, Y_ts_RF1000_init)
        Y_ts_pred_enemble = smooth_output(T_test, D_test, Y_ts_pred_enemble_init)
        Y_ts_pred_adaboost = smooth_output(T_test, D_test, Y_ts_pred_adaboost_init)
        Y_ts_pred_voting = smooth_output(T_test, D_test, Y_ts_pred_voting_init)
        weights.append(sum(Y_test))

        fs_name = '_RF'

        save_model_outputs(T_test, Y_test, Y_ts_NB_init, Y_ts_NB, 'NB_' + pid_test + fs_name + '.csv')
        save_model_outputs(T_test, Y_test, Y_ts_CART_init, Y_ts_CART, 'CART_' + pid_test + fs_name + '.csv')
        save_model_outputs(T_test, Y_test, Y_ts_RF100_init, Y_ts_RF100, 'RF100_' + pid_test + fs_name + '.csv')
        save_model_outputs(T_test, Y_test, Y_ts_RF1000_init, Y_ts_RF1000, 'RF1000_' + pid_test + fs_name + '.csv')
        save_model_outputs(T_test, Y_test, Y_ts_pred_enemble_init, Y_ts_pred_enemble, 'Ens_' + pid_test + fs_name + '.csv')
        save_model_outputs(T_test, Y_test, Y_ts_pred_adaboost_init, Y_ts_pred_adaboost, 'AB_' + pid_test + fs_name + '.csv')
        save_model_outputs(T_test, Y_test, Y_ts_pred_voting_init, Y_ts_pred_voting, 'Vote_' + pid_test + fs_name + '.csv')

        curRes_smooth = []
        curRes_smooth.append(get_recal_precision_f1_accuracy_CM(Y_test, Y_ts_NB))
        curRes_smooth.append(get_recal_precision_f1_accuracy_CM(Y_test, Y_ts_CART))
        curRes_smooth.append(get_recal_precision_f1_accuracy_CM(Y_test, Y_ts_RF100))
        curRes_smooth.append(get_recal_precision_f1_accuracy_CM(Y_test, Y_ts_RF1000))
        curRes_smooth.append(get_recal_precision_f1_accuracy_CM(Y_test, Y_ts_pred_enemble))
        curRes_smooth.append(get_recal_precision_f1_accuracy_CM(Y_test, Y_ts_pred_adaboost))
        curRes_smooth.append(get_recal_precision_f1_accuracy_CM(Y_test, Y_ts_pred_voting))
        Res_smooth.append(curRes_smooth)

        curRes = []
        curRes.append(get_recal_precision_f1_accuracy_CM(Y_test, Y_ts_NB_init))
        curRes.append(get_recal_precision_f1_accuracy_CM(Y_test, Y_ts_CART_init))
        curRes.append(get_recal_precision_f1_accuracy_CM(Y_test, Y_ts_RF100_init))
        curRes.append(get_recal_precision_f1_accuracy_CM(Y_test, Y_ts_RF1000_init))
        curRes.append(get_recal_precision_f1_accuracy_CM(Y_test, Y_ts_pred_enemble_init))
        curRes.append(get_recal_precision_f1_accuracy_CM(Y_test, Y_ts_pred_adaboost_init))
        curRes.append(get_recal_precision_f1_accuracy_CM(Y_test, Y_ts_pred_voting_init))
        print(pid_test, 'ens', get_recal_precision_f1_accuracy_CM(Y_test, Y_ts_pred_enemble_init))
        print(pid_test, 'ab', get_recal_precision_f1_accuracy_CM(Y_test, Y_ts_pred_adaboost_init))
        Res.append(curRes)

    plot_results(Res, modelnames, 'RES_LOSOCV_brushing_FS_RF.pckl', 'LOSOCV_result_brushing_event_FS_gt3_RF.pdf')
    plot_results(Res_smooth, modelnames, 'RES_LOSOCV_brushing_FS_smooth_RF.pckl', 'LOSOCV_result_brushing_event_FS_gt3_smooth_RF.pdf')
    print('#of pids', cnt)

def evaluate_LOSOCV(pids, skipped_pids, do_feature_selection=True, do_smooth=True, do_save_model_outputs=True):
    modelnames = ['Gaussian Neive Bayes', 'Random Forest 100', 'FR 100 with smoothing', 'ensemble 50 RFs',
                  'ensemble 50 RFs with smoothing', 'Ada-boost', 'Voting']
    modelnames = ['RF-50', 'Ens', 'AB']

    print("cross_subject_validation(pids)", pids)

    cnt = 0

    Res = []
    Res_smooth = []
    weights = []
    AB_res = {}
    for pid_test in pids:

        if pid_test in skipped_pids:
            continue
        # T_test, X_test, Y_test = get_XY_pos_neg_sep(feature_dir, pid_test)
        feature_names, T_test, X_test, Y_test = get_XY(pid_test)

        X_train = []
        Y_train = []

        for pid_train in pids:

            if pid_train in skipped_pids:
                continue
            if pid_train == pid_test:
                continue
            fn, T_tn, X_tn, Y_tn = get_XY(pid_train)

            X_train.extend(list(X_tn))
            Y_train.extend(list(Y_tn))
        print('Train labels:', set(Y_train))
        print('Test labels:', set(Y_test))
        print('--------------For ', pid_test, 'train size', len(list(X_train)), 'train pos', sum(Y_train), 'test size',
              len(list(X_test)), 'test pos', sum(Y_test))
        if sum(Y_test) < 3:
            continue

        cnt += 1

        X_train = np.array(X_train)
        X_test = np.array(X_test)

        # X_train, X_test = transfer_Scaler(X_train, X_test)

        if do_feature_selection:
            # fs_CFS = FCBF()
            # fs_CFS = RF_FS()
            # X_train = fs_CFS.fit_transform(X_train, Y_train)
            # X_test = fs_CFS.transform(X_test)
            feature_ind = get_selected_features(X_train, Y_train)
            X_train = X_train[:, feature_ind]
            X_test = X_test[:, feature_ind]
        #
        Y_ts_RF1000_init = evaluate_single_random_forest(X_train, Y_train, X_test, Y_test, T_test, 50)
        Y_ts_pred_enemble_init = evaluate_ensemble_multiple_random_forest(X_train, Y_train, X_test, Y_test, T_test)
        Y_ts_pred_adaboost_init = evaluate_adaboost(X_train, Y_train, X_test)

        D_test = list(X_test[:, 0])
        Y_ts_RF1000 = smooth_output(T_test, D_test, Y_ts_RF1000_init)
        Y_ts_pred_enemble = smooth_output(T_test, D_test, Y_ts_pred_enemble_init)
        Y_ts_pred_adaboost = smooth_output(T_test, D_test, Y_ts_pred_adaboost_init)
        weights.append(sum(Y_test))

        fs_name = '_RF'

        save_model_outputs(T_test, Y_test, Y_ts_RF1000_init, Y_ts_RF1000, 'RF50_' + pid_test + fs_name + '.csv')
        save_model_outputs(T_test, Y_test, Y_ts_pred_enemble_init, Y_ts_pred_enemble, 'Ens_' + pid_test + fs_name + '.csv')
        save_model_outputs(T_test, Y_test, Y_ts_pred_adaboost_init, Y_ts_pred_adaboost, 'AB_' + pid_test + fs_name + '.csv')

        curRes = []
        curRes.append(get_recal_precision_f1_accuracy_CM(Y_test, Y_ts_RF1000_init))
        curRes.append(get_recal_precision_f1_accuracy_CM(Y_test, Y_ts_pred_enemble_init))
        curRes.append(get_recal_precision_f1_accuracy_CM(Y_test, Y_ts_pred_adaboost_init))
        print(pid_test, 'ens', get_recal_precision_f1_accuracy_CM(Y_test, Y_ts_pred_enemble_init))
        print(pid_test, 'ab', get_recal_precision_f1_accuracy_CM(Y_test, Y_ts_pred_adaboost_init))
        AB_res[pid_test] = get_recal_precision_f1_accuracy_CM(Y_test, Y_ts_pred_adaboost_init)
        Res.append(curRes)

    if do_feature_selection:
        res_modelwise =plot_results(Res, modelnames, 'RES_LOSOCV_brushing_FS_svcAB.pckl', 'LOSOCV_result_brushing_event_FS_svcAB_ns')
    else:
        res_modelwise =plot_results(Res, modelnames, 'RES_LOSOCV_brushingA_svcAB.pckl', 'LOSOCV_result_brushing_eventA_svcAB_ns')

    print('#of pids', cnt)
    print('participantwise results....')
    for p, v in AB_res.items():
        print(p, v)
    return res_modelwise, AB_res
if __name__ == "__main__":

    pids = [d for d in os.listdir(sensor_data_dir) if os.path.isdir(os.path.join(sensor_data_dir, d)) and len(d) == 4]

    skipped_pids = ['8337', 'a764', 'aebb', 'b0e8']

    # count_positive_events(pids)
    # cross_subject_validation(pids)
    # res_modelwise, AB_res = evaluate_LOSOCV(pids, skipped_pids, do_feature_selection=True)
    res_modelwise, AB_res = evaluate_LOSOCV(pids, skipped_pids, do_feature_selection=False)


# participantwise results....
# 813f [0.75, 1.0, 0.8571428571428571, 0.9991971095945403, 0]
# 820c [1.0, 1.0, 1.0, 1.0, 0]
# 86bd [1.0, 1.0, 1.0, 1.0, 0]
# 891e [0.2857142857142857, 1.0, 0.4444444444444445, 0.9971114962449451, 0]
# 93a2 [1.0, 1.0, 1.0, 1.0, 0]
# 94c0 [1.0, 1.0, 1.0, 1.0, 0]
# 999e [1.0, 0.875, 0.9333333333333333, 0.9995822890559732, 1]
# 9a6b [1.0, 1.0, 1.0, 1.0, 0]
# 9e33 [1.0, 1.0, 1.0, 1.0, 0]
# 9eee [1.0, 1.0, 1.0, 1.0, 0]
# a153 [1.0, 1.0, 1.0, 1.0, 0]
# a64e [1.0, 0.6153846153846154, 0.761904761904762, 0.9975822050290135, 5]
# aa22 [1.0, 1.0, 1.0, 1.0, 0]
# b15a [0.8, 1.0, 0.888888888888889, 0.9993510707332901, 0]
# b27d [1.0, 1.0, 1.0, 1.0, 0]
# b67b [0.3333333333333333, 1.0, 0.5, 0.9946879150066401, 0]
# be71 [0.75, 1.0, 0.8571428571428571, 0.9981981981981982, 0]