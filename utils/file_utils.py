import pandas as pd
import numpy as np
from annotations.CONSTANTS import *
import pickle

def save_as_csv(X, Y, feature_name, output_dir, output_filename='features_and_labels.csv'):
    # print(X[0], len(X[0]), len(feature_name))
    # print('#x', len(X), '#y', len(Y))
    data = np.array(X)
    pd_data = pd.DataFrame(data=data,columns=feature_name)
    pd_data['label'] = [v for v in Y]
    # pd_data['event_key'] = [v[1] for v in Y]

    pd_data.to_csv(output_dir + output_filename, encoding='utf-8', index=False)
    print('---feature saved in ', output_dir + output_filename)

def append_to_file(filename, txt):
    fh = open(filename, 'a')
    fh.write(txt + '\n')
    fh.close()


def save_model_outputs(T, Y, Y_init, Y_smooth, filename):
    df = pd.DataFrame({'T': T, 'Y': Y, 'Y_init': Y_init, 'Y_smooth': Y_smooth})
    df.to_csv(result_dir + 'model_outputs/' + filename, index=False)

def save_model_file(model, filename = 'brushingAB.model'):
    # save the model
    pickle.dump(model, open(filename, 'wb'))

def load_model_from_file(filename = 'brushingAB.model'):
    # load the model from file
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model
