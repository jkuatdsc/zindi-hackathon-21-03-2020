import os, json, time
import pandas as pd
# from prediction import predict
#
# test_dir_path = "../../Data/Processed/Test"
sample_submission_path = "../../Data/Raw/UmojaHack#1:SAEON_Identifying_marine_invertebrates/SampleSubmission .csv"
# submission_file = "../../Output/Models/submission_{}.csv".format(time.asctime().replace(" ", "_").replace(":", "_"))
#
# sample_submission = pd.read_csv(sample_submission_path)
# columns = sample_submission.columns
#
# # test = test[['Destination_Lon', 'Pickup_Lat', 'Destination_Lat', 'Pickup_Lon', 'Distance(km)', 'time_to_pick_up', 'Day_of_Week', 'User_Id', 'Rider Id', 'Day_of_Month', 'Age', 'No_of_Ratings', 'Order_No']]
# # gb = best_xgb_model.predict(test)
# # lgbm_output = pd.DataFrame({"Order No":test_df['Order No'], "Time from Pickup to Arrival": gb })
# # file_name = str("{}_submission.csv".format(dt.datetime.now().ctime())).replace(" ", "_").replace(":", "_").replace("-", "_")
# # lgbm_output.to_csv(file_name, index=False)
# values = {}


def read_images():
    print("Reading images")
    for root, dirs, files in os.walk(test_dir_path):
        for file in files:
            if file.endswith('jpeg') or file.endswith('JPEG'):
                print(os.path.join(root, file))
                label, cols = predict(os.path.join(root, file))
                values[file] = [str(i) for i in list(label)]
    return values, cols


def make_submission_file(values, cols):
    data_in_json = json.dumps(values)
    submission_df = pd.read_json(data_in_json)
    submission_df = submission_df.T
    submission_df.columns = cols
    submission_df.index.name = "FILE"
    submission_df.to_csv(submission_file)


def cross_check():
    sample_submission = pd.read_csv(sample_submission_path)
    col1 = list(sample_submission.columns)
    submission_df = pd.read_csv("/home/r0x6f736f646f/Documents/Projects/zindi-hackathon-21-03-2020/UmojaHack#1/Output/Models/submission_Fri_Mar_27_00_08_24_2020.csv")
    col2 = list(submission_df.columns)
    print(len(col1))
    print(len(col2))
    # if set(col1) == set(col2):
    #     print("Equal")
    # else :
    #     print("Not equal")
    removed = 0
    not_r = 0
    for i in col1:
        try:
            col2.remove(i)
            removed += 1
        except ValueError:
            not_r += 1
    print(removed, col2)

if __name__ == "__main__":
    # values, cols = read_images()
    # make_submission_file(values, cols)
    cross_check()
