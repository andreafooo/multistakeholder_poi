import pandas as pd
import os
import json
from globals import BASE_DIR, available_datasets

# beware: opening the yelp file with pandas will take a lot of time (approx 10 min)
include_categories = False  # for context-aware recommendation


def open_big_json(file_path):
    """This function is used to open the Yelp data"""
    data = []

    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    df = pd.DataFrame(data)

    return df


def convert_to_unix_timestamp(df, column_name):
    """
    Convert a column of timestamps in a DataFrame to Unix timestamps.

    Args:
        df (pd.DataFrame): The DataFrame containing the timestamp column.
        column_name (str): The name of the column with timestamps in "%Y-%m-%d %H:%M:%S" format.

    Returns:
        pd.DataFrame: The DataFrame with an additional column for Unix timestamps.
    """
    df[column_name] = pd.to_datetime(df[column_name], format="mixed")

    df[f"{column_name}"] = df[column_name].apply(lambda x: x.timestamp())

    return df


def dataset_specific_preprocessing(dataset, DATASET_DIR):
    if dataset == "foursquarenyc" or dataset == "foursquaretky":
        checkin_df = pd.read_csv(DATASET_DIR + "foursquare_data.csv", sep=",")
        checkin_df = checkin_df.drop(columns=["timezoneOffset"])
        checkin_df = checkin_df.rename(
            columns={
                "venueId": "item_id:token",
                "venueCategoryId": "category_id:token",
                "venueCategory": "category_name:token_seq",
                "userId": "user_id:token",
                "utcTimestamp": "timestamp:float",
                "latitude": "lat:float",
                "longitude": "lon:float",
            }
        )
        user_df = checkin_df[["user_id:token"]].drop_duplicates()

        poi_df = checkin_df[
            [
                "item_id:token",
                "category_id:token",
                "category_name:token_seq",
                "lat:float",
                "lon:float",
            ]
        ].drop_duplicates(subset=["item_id:token"])
        checkin_df = checkin_df[["user_id:token", "item_id:token", "timestamp:float"]]

    elif dataset == "gowalla" or dataset == "brightkite":
        checkin_df = pd.read_csv(
            DATASET_DIR + f"loc-{dataset}_totalCheckins.txt",
            sep="\t",
            header=None,
            names=[
                "user_id:token",
                "timestamp:float",
                "lat:float",
                "lon:float",
                "item_id:token",
            ],
        )
        checkin_df = checkin_df[
            ~checkin_df["item_id:token"].isin(
                ["00000000000000000000000000000000", "ede07eeea22411dda0ef53e233ec57ca"]
            )
        ]
        user_df = pd.read_csv(
            DATASET_DIR + f"loc-{dataset}_edges.txt",
            sep="\t",
            header=None,
            names=["user_id:token", "friends:token_seq"],
        )
        user_df = (
            user_df.groupby("user_id:token")["friends:token_seq"]
            .apply(lambda x: ",".join(map(str, x)))
            .reset_index()
        )
        user_df.columns = ["user_id:token", "friends:token_seq"]
        poi_df = checkin_df[
            ["item_id:token", "lat:float", "lon:float"]
        ].drop_duplicates(subset="item_id:token")
        checkin_df = checkin_df.drop(columns=["lat:float", "lon:float"])

    elif dataset == "yelp":
        poi_df = pd.read_json(
            DATASET_DIR + "yelp_academic_dataset_business.json", lines=True
        )
        poi_df = poi_df.loc[poi_df["is_open"] == 1]
        poi_df = poi_df.drop(
            columns=[
                "review_count",
                "stars",
                "hours",
                "is_open",
                "city",
                "state",
                "postal_code",
                "attributes",
                "address",
            ]
        )
        poi_df = poi_df.rename(
            columns={
                "latitude": "lat:float",
                "longitude": "lon:float",
                "business_id": "item_id:token",
                "name": "name:token_seq",
                "categories": "category_name:token_seq",
            }
        )
        user_df = open_big_json(DATASET_DIR + "yelp_academic_dataset_user.json")
        user_df = user_df.drop(
            columns=[
                "review_count",
                "name",
                "yelping_since",
                "useful",
                "funny",
                "cool",
                "elite",
                "fans",
                "compliment_hot",
                "average_stars",
                "compliment_more",
                "compliment_profile",
                "compliment_cute",
                "compliment_list",
                "compliment_note",
                "compliment_plain",
                "compliment_cool",
                "compliment_funny",
                "compliment_writer",
                "compliment_photos",
            ]
        )
        user_df = user_df.rename(
            columns={"user_id": "user_id:token", "friends": "friends:token_seq"}
        )
        checkin_df = open_big_json(DATASET_DIR + "yelp_academic_dataset_review.json")
        checkin_df = checkin_df.drop(
            columns=["text", "cool", "stars", "useful", "funny", "review_id"]
        )
        checkin_df = checkin_df.rename(
            columns={
                "user_id": "user_id:token",
                "business_id": "item_id:token",
                "date": "timestamp:float",
            }
        )
        checkin_df["timestamp"] = pd.to_datetime(
            checkin_df["timestamp:float"], errors="coerce"
        )

        checkin_df["year"] = checkin_df[
            "timestamp"
        ].dt.year  # Extract the year from the 'timestamp' column
        checkin_df = checkin_df[
            checkin_df["year"] >= 2018
        ]  # Keep only the check-ins from 2018 and 2019
        checkin_df = checkin_df[checkin_df["year"] < 2020]
        checkin_df.drop(columns=["year", "timestamp"], inplace=True)

    return checkin_df, user_df, poi_df


def filter_df(df, min_reviews_user=15, min_reviews_business=10):
    while True:
        # Filter users with at least min_reviews reviews
        user_counts = df["user_id:token"].value_counts()
        user_mask = df["user_id:token"].map(user_counts) >= min_reviews_user
        df_filtered = df.loc[user_mask]

        # Filter businesses with at least min_reviews reviews
        business_counts = df_filtered["item_id:token"].value_counts()
        business_mask = (
            df_filtered["item_id:token"].map(business_counts) >= min_reviews_business
        )
        df_filtered = df_filtered.loc[business_mask]

        # If the size of the filtered DataFrame didn't change, break the loop
        if df_filtered.shape[0] == df.shape[0]:
            break

        # Update the DataFrame for the next iteration
        df = df_filtered

    return df_filtered


def user_popularity_sample_calculator(
    checkin_df_filtered, poi_df, user_df, sep_num, checkin_df_timestamp
):
    # Calculate average popularity per user
    average_popularity_per_user = (
        checkin_df_filtered.groupby("user_id:token")["business_popularity:float"]
        .mean()
        .reset_index()
    )  # try out median of item popularities in user profile instead of mean
    average_popularity_per_user.columns = ["user_id:token", "average_popularity"]

    average_popularity_per_user = average_popularity_per_user.sort_values(
        by="average_popularity", ascending=False
    )

    # Get top users
    high_pop_user_df_sample = average_popularity_per_user.head(sep_num)

    # Get the users around the median
    median_index = len(average_popularity_per_user) // 2
    start_med_index = max(median_index - int(sep_num * 1.5), 0)
    end_med_index = min(
        median_index + int(sep_num * 1.5), len(average_popularity_per_user)
    )
    med_pop_user_df_sample = average_popularity_per_user.iloc[
        start_med_index:end_med_index
    ]

    # Get the lowest users
    low_pop_user_df_sample = average_popularity_per_user.tail(sep_num)

    unique_users = list(
        set(
            high_pop_user_df_sample["user_id:token"].tolist()
            + med_pop_user_df_sample["user_id:token"].tolist()
            + low_pop_user_df_sample["user_id:token"].tolist()
        )
    )

    checkin_df_sample = checkin_df_filtered[
        checkin_df_filtered["user_id:token"].isin(unique_users)
    ]
    checkin_df_sample = checkin_df_sample[
        checkin_df_sample["user_id:token"].isin(unique_users)
    ]

    user_df_sample = user_df[user_df["user_id:token"].isin(unique_users)]
    poi_df_sample = poi_df[
        poi_df["item_id:token"].isin(checkin_df_sample["item_id:token"])
    ]

    checkin_df_sample = checkin_df_sample[
        checkin_df_sample["item_id:token"].isin(poi_df_sample["item_id:token"])
    ]

    checkin_df_timestamp = checkin_df_timestamp[
        checkin_df_timestamp["user_id:token"].isin(unique_users)
    ]
    checkin_df_timestamp = checkin_df_timestamp[
        checkin_df_timestamp["item_id:token"].isin(poi_df_sample["item_id:token"])
    ]

    return (
        checkin_df_sample,
        high_pop_user_df_sample,
        med_pop_user_df_sample,
        low_pop_user_df_sample,
        user_df_sample,
        poi_df_sample,
        checkin_df_timestamp,
    )


def id_factorizer(
    checkin_df_sample,
    high_pop_user_df_sample,
    med_pop_user_df_sample,
    low_pop_user_df_sample,
    user_df_sample,
    poi_df_sample,
    checkin_df_timestamp,
):
    """Overwriting the actual ID with a factorized ID so that we can use the same ID both in RecBole and CAPRI"""
    checkin_df_sample["user_id:token"], user_id_map = pd.factorize(
        checkin_df_sample["user_id:token"]
    )
    checkin_df_sample["item_id:token"], business_id_map = pd.factorize(
        checkin_df_sample["item_id:token"]
    )

    # Create mapping dictionaries
    user_id_mapping = {original: i for i, original in enumerate(user_id_map)}
    business_id_mapping = {original: j for j, original in enumerate(business_id_map)}

    high_pop_user_df_sample["user_id:token"] = high_pop_user_df_sample[
        "user_id:token"
    ].map(user_id_mapping)
    med_pop_user_df_sample["user_id:token"] = med_pop_user_df_sample[
        "user_id:token"
    ].map(user_id_mapping)
    low_pop_user_df_sample["user_id:token"] = low_pop_user_df_sample[
        "user_id:token"
    ].map(user_id_mapping)

    checkin_df_timestamp["user_id:token"] = checkin_df_timestamp["user_id:token"].map(
        user_id_mapping
    )
    checkin_df_timestamp["item_id:token"] = checkin_df_timestamp["item_id:token"].map(
        business_id_mapping
    )

    user_df_sample["user_id:token"] = user_df_sample["user_id:token"].map(
        user_id_mapping
    )
    poi_df_sample["item_id:token"] = poi_df_sample["item_id:token"].map(
        business_id_mapping
    )

    return (
        checkin_df_sample,
        high_pop_user_df_sample,
        med_pop_user_df_sample,
        low_pop_user_df_sample,
        user_df_sample,
        poi_df_sample,
        checkin_df_timestamp,
    )


def user_id_token_adder(df, column_name_list=["user_id:token", "item_id:token"]):
    """Recbole needs a token (string) instead of a number for the user and item ID"""
    for column_name in column_name_list:
        try:
            df[column_name] = df[column_name].astype(int)
            df[column_name] = df[column_name].astype(str) + "_x"
        except KeyError:
            pass
    return df


def data_saver_recbole(df, framework, suffix, DATASET_DIR, dataset):
    if not os.path.exists(DATASET_DIR + "processed_data_" + framework):
        os.makedirs(DATASET_DIR + "processed_data_" + framework)

    df.to_csv(
        f"{DATASET_DIR}processed_data_{framework}/{dataset}_sample.{suffix}",
        sep="\t",
        index=False,
    )


def data_saver_capri(df, filename, DATASET_DIR):
    if not os.path.exists(DATASET_DIR + "processed_data_capri"):
        os.makedirs(DATASET_DIR + "processed_data_capri")

    df.to_csv(
        DATASET_DIR + "processed_data_capri/" + filename + ".txt",
        sep="\t",
        index=False,
        header=False,
    )
    print("Data saved as " + filename + ".txt")


def user_id_cleaner(df, column_name_list=["user_id:token", "item_id:token"]):
    """Save for CAPRI without _x since they require integers as IDs"""
    for column_name in column_name_list:
        df[column_name] = df[column_name].str.split("_")
        df[column_name] = df[column_name].apply(lambda x: x[0])

    return df


#############################################


def main():
    for dataset in available_datasets:
        DATASET_DIR = f"{BASE_DIR}{dataset}_dataset/"
        if dataset not in available_datasets:
            print(f"Dataset '{dataset}' is not available.")
            return

        # Perform dataset-specific preprocessing
        checkin_df, user_df, poi_df = dataset_specific_preprocessing(
            dataset, DATASET_DIR
        )
        checkin_df.sort_values(by="timestamp:float", ascending=True, inplace=True)
        checkin_df_timestamp = checkin_df.copy()
        # Step 1: Group by user_id and business_id and count check-ins
        checkin_df["checkin_count:float"] = checkin_df.groupby(
            ["user_id:token", "item_id:token"]
        )["item_id:token"].transform("count")

        checkin_df = checkin_df.drop_duplicates(
            subset=["user_id:token", "item_id:token"], keep="first"
        )
        print(
            "Number of users, number of POIs",
            len(checkin_df["user_id:token"].unique()),
            len(checkin_df["item_id:token"].unique()),
        )
        print(
            "Sparsity:",
            1
            - len(checkin_df)
            / (
                len(checkin_df["user_id:token"].unique())
                * len(checkin_df["item_id:token"].unique())
            ),
        )
        if dataset == "gowalla":
            checkin_df_filtered = filter_df(
                checkin_df, 15, 20
            )  # for gowalla i used business min 20 & user min 15
        else:
            checkin_df_filtered = filter_df(checkin_df, 15, 10)

        value_counts = checkin_df_filtered["item_id:token"].value_counts().reset_index()
        value_counts.columns = ["item_id:token", "count"]

        max_count = value_counts["count"].max()
        value_counts["business_popularity:float"] = value_counts["count"] / max_count

        checkin_df_filtered = checkin_df_filtered.merge(
            value_counts[["item_id:token", "business_popularity:float"]],
            on="item_id:token",
            how="left",
        )
        if checkin_df_filtered["user_id:token"].nunique() > 1500:
            sep_num = 1500 // 5
        else:
            sep_num = checkin_df_filtered["user_id:token"].nunique() // 5

        # Create samples of 1500 users (except for foursquaretky where only 600 users are in the dataset)
        (
            checkin_df_sample,
            high_pop_user_df_sample,
            med_pop_user_df_sample,
            low_pop_user_df_sample,
            user_df_sample,
            poi_df_sample,
            checkin_df_timestamp,
        ) = user_popularity_sample_calculator(
            checkin_df_filtered, poi_df, user_df, sep_num, checkin_df_timestamp
        )
        (
            checkin_df_sample,
            high_pop_user_df_sample,
            med_pop_user_df_sample,
            low_pop_user_df_sample,
            user_df_sample,
            poi_df_sample,
            checkin_df_timestamp,
        ) = id_factorizer(
            checkin_df_sample,
            high_pop_user_df_sample,
            med_pop_user_df_sample,
            low_pop_user_df_sample,
            user_df_sample,
            poi_df_sample,
            checkin_df_timestamp,
        )
        checkin_df_sample = user_id_token_adder(checkin_df_sample)
        high_pop_user_df_sample = user_id_token_adder(high_pop_user_df_sample)
        med_pop_user_df_sample = user_id_token_adder(med_pop_user_df_sample)
        low_pop_user_df_sample = user_id_token_adder(low_pop_user_df_sample)
        user_df_sample = user_id_token_adder(user_df_sample)
        poi_df_sample = user_id_token_adder(poi_df_sample)

        # get a json with the user id's of the respective popularity groups
        user_id_popularity = {}
        user_id_popularity["high"] = high_pop_user_df_sample["user_id:token"].tolist()
        user_id_popularity["medium"] = med_pop_user_df_sample["user_id:token"].tolist()
        user_id_popularity["low"] = low_pop_user_df_sample["user_id:token"].tolist()
        json.dump(
            user_id_popularity,
            open(f"{DATASET_DIR}/{dataset}_user_id_popularity.json", "w"),
        )
        checkin_df_sample["review_id:token"] = range(1, len(checkin_df_sample) + 1)
        checkin_df_sample = convert_to_unix_timestamp(
            checkin_df_sample, "timestamp:float"
        )
        checkin_df_timestamp = convert_to_unix_timestamp(
            checkin_df_timestamp, "timestamp:float"
        )
        checkin_df_sample.sort_values(by="checkin_count:float", ascending=False)
        # very important: keeping the duplicate check-ins for the context aware recommendation to have the timestamps saved

        # very important: dropping duplicate check-ins
        checkin_df_sample = checkin_df_sample.drop_duplicates(
            subset=["user_id:token", "item_id:token"], keep="first"
        )
        user_df_sample = user_df_sample[["user_id:token"]]
        checkin_df_timestamp = checkin_df_timestamp[
            ["user_id:token", "item_id:token", "timestamp:float"]
        ]  # FINAL
        checkins_capri_train_test_tune = checkin_df_sample[
            ["user_id:token", "item_id:token", "timestamp:float", "checkin_count:float"]
        ]
        try:
            poi_df_sample_capri = poi_df_sample[
                ["item_id:token", "lat:float", "lon:float"]
            ]  # FINAL
        except KeyError:  # if coordinates are not available
            poi_df_sample_capri = poi_df_sample[["item_id:token"]]
        datasize_capri = pd.DataFrame(
            data={
                "num_users": [
                    len(checkins_capri_train_test_tune["user_id:token"].unique())
                ],
                "num_items": [
                    len(checkins_capri_train_test_tune["item_id:token"].unique())
                ],
            }
        )  # FINAL

        # splitting the data into train, test, and tune
        checkins_capri_train_test_tune = checkins_capri_train_test_tune.sort_values(
            by=["user_id:token", "timestamp:float"]
        )
        checkins_capri_train_test_tune = checkins_capri_train_test_tune[
            ["user_id:token", "item_id:token", "checkin_count:float"]
        ]

        # Split the data
        train_list = []
        val_list = []
        test_list = []

        for user, group in checkins_capri_train_test_tune.groupby("user_id:token"):
            n = len(group)
            train_end = int(n * 0.65)
            val_end = int(n * 0.80)

            train_list.append(group.iloc[:train_end])
            val_list.append(group.iloc[train_end:val_end])
            test_list.append(group.iloc[val_end:])

        # Combine lists into DataFrames
        train_df = pd.concat(train_list)
        val_df = pd.concat(val_list)
        test_df = pd.concat(test_list)

        # adding a category column
        if include_categories is True:
            if dataset == "yelp":
                # Split the 'category_name' column by commas
                poi_df_sample["category_name_unstacked:token_seq"] = poi_df_sample[
                    "category_name:token_seq"
                ].str.split(", ")

                # Unstack the categories into multiple rows
                category_df_sample = poi_df_sample.explode(
                    "category_name_unstacked:token_seq"
                )
                category_counts = category_df_sample[
                    "category_name_unstacked:token_seq"
                ].value_counts()
                category_mask = (
                    category_df_sample["category_name_unstacked:token_seq"].map(
                        category_counts
                    )
                    >= 25
                )
                category_df_sample_filtered = category_df_sample.loc[category_mask]
                category_df_sample_filtered["category_id:token"], category_id = (
                    pd.factorize(
                        category_df_sample_filtered["category_name_unstacked:token_seq"]
                    )
                )
                category_df_sample_filtered.dropna(inplace=True)
                datasize_capri = pd.DataFrame(
                    data={
                        "num_users": [
                            len(
                                checkins_capri_train_test_tune["user_id:token"].unique()
                            )
                        ],
                        "num_items": [
                            len(
                                checkins_capri_train_test_tune["item_id:token"].unique()
                            )
                        ],
                        "num_categories": [
                            len(
                                category_df_sample_filtered[
                                    "category_id:token"
                                ].unique()
                            )
                        ],
                    }
                )  # FINAL
                data_saver_capri(
                    category_df_sample_filtered,
                    "poiCategories",
                    DATASET_DIR,
                )

            elif dataset == "foursquarenyc" or dataset == "foursquaretky":
                poi_df_sample["category_id:token"], category_id = pd.factorize(
                    poi_df_sample["category_name:token_seq"]
                )
                datasize_capri = pd.DataFrame(
                    data={
                        "num_users": [
                            len(
                                checkins_capri_train_test_tune["user_id:token"].unique()
                            )
                        ],
                        "num_items": [
                            len(
                                checkins_capri_train_test_tune["item_id:token"].unique()
                            )
                        ],
                        "num_categories": [
                            len(poi_df_sample["category_id:token"].unique())
                        ],
                    }
                )
                poi_df_categories = poi_df_sample[
                    ["item_id:token", "category_id:token"]
                ]
                data_saver_capri(poi_df_categories, "poiCategories", DATASET_DIR)

        # Checking the final data sizes
        print(f"Final data sizes of dataset {dataset}:")
        print(
            train_df["user_id:token"].nunique(),
            val_df["user_id:token"].nunique(),
            test_df["user_id:token"].nunique(),
        )

        # This is the correct split since we perform the splitting ourselves
        data_saver_recbole(train_df, "recbole", "train.inter", DATASET_DIR, dataset)
        data_saver_recbole(test_df, "recbole", "test.inter", DATASET_DIR, dataset)
        data_saver_recbole(val_df, "recbole", "valid.inter", DATASET_DIR, dataset)

        # Saving for context-aware POI recommendations in CAPRI
        poi_df_sample_capri = user_id_cleaner(poi_df_sample_capri, ["item_id:token"])
        train_df = user_id_cleaner(train_df)
        val_df = user_id_cleaner(val_df)
        test_df = user_id_cleaner(test_df)

        train_df["user_id:token"] = train_df["user_id:token"].astype(int)
        train_df.sort_values(by="item_id:token", ascending=False)
        train_df["checkin_count:float"] = train_df["checkin_count:float"].astype(int)
        test_df["checkin_count:float"] = test_df["checkin_count:float"].astype(int)
        val_df["checkin_count:float"] = val_df["checkin_count:float"].astype(int)

        data_saver_capri(checkin_df_timestamp, "checkins", DATASET_DIR)
        data_saver_capri(datasize_capri, "dataSize", DATASET_DIR)
        data_saver_capri(poi_df_sample_capri, "poiCoos", DATASET_DIR)
        data_saver_capri(train_df, "train", DATASET_DIR)
        data_saver_capri(val_df, "tune", DATASET_DIR)
        data_saver_capri(test_df, "test", DATASET_DIR)


if __name__ == "__main__":
    main()
