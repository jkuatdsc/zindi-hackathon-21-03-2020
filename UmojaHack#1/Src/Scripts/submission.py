import os, json, time
import pandas as pd
from prediction import predict

test_dir_path = "../../Data/Processed/Test"
sample_submission_path = "../../Data/Raw/UmojaHack#1:SAEON_Identifying_marine_invertebrates/SampleSubmission .csv"
submission_file = "../../Output/Models/submission_{}.csv".format(time.asctime().replace(" ", "_").replace(":", "_"))

sample_submission = pd.read_csv(sample_submission_path)
columns = sample_submission.columns

# test = test[['Destination_Lon', 'Pickup_Lat', 'Destination_Lat', 'Pickup_Lon', 'Distance(km)', 'time_to_pick_up', 'Day_of_Week', 'User_Id', 'Rider Id', 'Day_of_Month', 'Age', 'No_of_Ratings', 'Order_No']]
# gb = best_xgb_model.predict(test)
# lgbm_output = pd.DataFrame({"Order No":test_df['Order No'], "Time from Pickup to Arrival": gb })
# file_name = str("{}_submission.csv".format(dt.datetime.now().ctime())).replace(" ", "_").replace(":", "_").replace("-", "_")
# lgbm_output.to_csv(file_name, index=False)
values = {}


def read_images():
    print("Reading images")
    for root, dirs, files in os.walk(test_dir_path):
        for file in files:
            if file.endswith('jpeg') or file.endswith('JPEG'):
                print(os.path.join(root, file))
                label, cols = predict(os.path.join(root, file))
                values[file] = label
    return values, cols

def make_submission_file(values, cols):
    data_in_json = json.loads(values)
    submission_df = pd.read_json(data_in_json)
    submission_df = submission_df.T
    submission_df.columns = cols
    submission_df.to_csv(submission_file)



if __name__ == "__main__":
    values, cols = read_images()
    make_submission_file(values, cols)
