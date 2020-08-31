from annotations.CONSTANTS import *

from main_feature_generation import do_generate_feature_vector_and_labels_for_brushing
from main_LOSOCV_brushing import evaluate_LOSOCV
from main_train_ML_model_and_export import train_and_save_AB_model
from main_mORAL_testing import process_mORAL_and_validate, process_mORAL


pids = [d for d in os.listdir(sensor_data_dir) if os.path.isdir(os.path.join(sensor_data_dir, d)) and len(d) == 4]
skipped_pids = ['8337', 'a764', 'aebb', 'b0e8']

print(pids)

# Step 1:
    # Generate features and brushing labels for all the participants, i.e., pids
    # Export the features and labels as CSV files participantwise
do_generate_feature_vector_and_labels_for_brushing(pids)

# Step 2
#     Evaluate different models by Leave-One-Subject_Out_Cross_validation
evaluate_LOSOCV(pids, skipped_pids, do_feature_selection=True)
evaluate_LOSOCV(pids, skipped_pids, do_feature_selection=False)

# # Step 3
# #     Train the best model (from previous step) and export as pickle file
# model_filename = 'trained_model_files/brushingAB.model'
# train_and_save_AB_model(pids, skipped_pids, model_filename=model_filename)
#
# # Step 4
# #     Testing: Use the trained brushing model and get brushing events
# process_mORAL_and_validate(pids)
# process_mORAL(pids)
