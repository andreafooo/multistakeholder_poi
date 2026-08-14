import pandas as pd
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from globals import available_datasets, BASE_DIR, raw_source_dataset  # noqa: E402




def main(datasets=None):
    for dataset in (datasets or available_datasets):
        # city-restricted variants (e.g. "yelpphi") have no raw json/csv files of their
        # own -- they reuse the source dataset's raw files, but everything derived
        # (id_mappings.json, this script's output) lives under the variant's own folder
        base_dataset = raw_source_dataset.get(dataset, dataset)
        dataset_dir = os.path.join(BASE_DIR, f"{dataset}_dataset")
        raw_dir = os.path.join(BASE_DIR, f"{base_dataset}_dataset")

        if base_dataset == "foursquaretky":
            # Load your id_mappings.json
            with open(os.path.join(dataset_dir, "id_mappings.json"), "r") as f:
                id_mappings = json.load(f)

            # Extract the set of valid user IDs (cast to str so the isin() comparison
            # below actually matches, regardless of whether the JSON stored them as
            # ints or strings)
            valid_user_ids = set(str(v) for v in id_mappings['user'].values())

            # Read the CSV dataset
            df = pd.read_csv(os.path.join(raw_dir, "foursquare_data.csv"))  # or whatever your file is named

            # Filter to keep only rows where userId is in valid_user_ids
            filtered_df = df[
                (df['userId'].astype(str).isin(valid_user_ids))
            ]

            # Save the filtered dataset
            filtered_df.to_csv(os.path.join(dataset_dir, f"{dataset}_sample_full_metadata.csv"), index=False)

            print(f"Filtered dataset saved. Original: {len(df)} rows, Filtered: {len(filtered_df)} rows")

        elif base_dataset == "yelp":
            # Load your id_mappings.json
            with open(os.path.join(dataset_dir, "id_mappings.json"), "r") as f:
                id_mappings = json.load(f)

            # Extract valid user and item (business) IDs as strings, so the
            # isin() comparisons below match regardless of the original JSON type
            valid_user_ids = set(str(v) for v in id_mappings["user"].values())

            # The item/business key name can vary by dataset export -- try the
            # common ones so this doesn't silently produce an empty set
            item_key = next((k for k in ("item", "business", "venue") if k in id_mappings), None)
            if item_key is None:
                raise KeyError(
                    f"Could not find an item/business mapping key in id_mappings.json. "
                    f"Available keys: {list(id_mappings.keys())}"
                )
            valid_item_ids = set(str(v) for v in id_mappings[item_key].values())

            CHUNK_SIZE = 100_000  # tune down if it still OOMs, up if it's too slow

            def filter_json_in_chunks(input_path, output_path, id_columns, valid_id_sets):
                """
                Stream a json-lines file in chunks, keep only rows where every
                column in id_columns is present in the corresponding set in
                valid_id_sets, and append the result straight to a CSV so the
                full file is never held in memory at once.
                """
                total_rows = 0
                kept_rows = 0
                first_chunk = True

                reader = pd.read_json(input_path, lines=True, chunksize=CHUNK_SIZE)
                for chunk in reader:
                    total_rows += len(chunk)

                    mask = pd.Series(True, index=chunk.index)
                    for col, valid_ids in zip(id_columns, valid_id_sets):
                        mask &= chunk[col].astype(str).isin(valid_ids)

                    filtered_chunk = chunk[mask]
                    kept_rows += len(filtered_chunk)

                    filtered_chunk.to_csv(
                        output_path,
                        mode="w" if first_chunk else "a",
                        header=first_chunk,
                        index=False,
                    )
                    first_chunk = False

                # Make sure we still create an (empty, headerless) output file
                # if the input existed but every chunk was filtered out entirely
                if first_chunk:
                    open(output_path, "w").close()

                return total_rows, kept_rows

            # --- business file ---
            business_total, business_kept = filter_json_in_chunks(
                os.path.join(raw_dir, "yelp_academic_dataset_business.json"),
                os.path.join(dataset_dir, f"{dataset}_sample_business_metadata.csv"),
                id_columns=["business_id"],
                valid_id_sets=[valid_item_ids],
            )
            print(f"Filtered business file saved. Original: {business_total} rows, Filtered: {business_kept} rows")

            # --- user file ---
            user_total, user_kept = filter_json_in_chunks(
                os.path.join(raw_dir, "yelp_academic_dataset_user.json"),
                os.path.join(dataset_dir, f"{dataset}_sample_user_metadata.csv"),
                id_columns=["user_id"],
                valid_id_sets=[valid_user_ids],
            )
            print(f"Filtered user file saved. Original: {user_total} rows, Filtered: {user_kept} rows")

            # --- review file ---
            review_total, review_kept = filter_json_in_chunks(
                os.path.join(raw_dir, "yelp_academic_dataset_review.json"),
                os.path.join(dataset_dir, f"{dataset}_sample_review_metadata.csv"),
                id_columns=["user_id", "business_id"],
                valid_id_sets=[valid_user_ids, valid_item_ids],
            )
            print(f"Filtered review file saved. Original: {review_total} rows, Filtered: {review_kept} rows")


if __name__ == "__main__":
    main()