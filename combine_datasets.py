import setup_functions as setup
import pandas as pd

folder_path = "/home/keskan01/acl_2025_norm/dataset/journal_dataset"
target_path = "/home/keskan01/acl_2025_norm/dataset/journal_dataset/combined"

def main(path) : 
    file_list = setup.list_from_folder(path)
    combined_df = setup.combine_csvs(file_list)
    name = "combined_dataset.csv"
    combined_df.to_csv(f"{target_path}/{name}", index=False)



if __name__ == "__main__":
    main(folder_path)