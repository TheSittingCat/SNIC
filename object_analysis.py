import pandas as pd
import setup_functions as setup
from tqdm import tqdm

def count_objects(dataset : pd.DataFrame) :
    """
    Count the number of objects in the dataset.
    """

    # Count the number of objects in the dataset
    total_count = 0

    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0], desc="Counting objects", unit="object") : 
        # Get the objects in the row in the list format
        list_of_objects = row["all_objects"].split(", ")
        total_count += len(list_of_objects)
    return total_count, total_count / dataset.shape[0] # Return the total count and the average number of objects per row

def count_objects_per_task(dataset : pd.DataFrame) :
    """
    Count the number of objects in the dataset per task.
    """

    total_count_dict = {}
    instance_dict = {}
    average_dict = {}

    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0], desc="Counting objects per task", unit="object") :
        # Get the task and the objects in the row in the list format
        list_of_objects = row["all_objects"].split(", ")
        task = row["Source File"]

        if total_count_dict.get(task) is None :
            total_count_dict[task] = len(list_of_objects)
            instance_dict[task] = 1
        else : 
            total_count_dict[task] += len(list_of_objects)
            instance_dict[task] += 1
    # Calculate the average number of objects per task
    for task in total_count_dict.keys() :
        average_dict[task] = total_count_dict[task] / instance_dict[task]
    return total_count_dict, average_dict # Return the total count and the average number of objects per task

def get_added_attribute_count(dataset : pd.DataFrame) : 

    """
    Get the number of added attributes in the dataset.
    """
    total_count = 0
    cut_off = "additional attributes: "
    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0], desc="Counting added attributes", unit="object") :
        added_attributes_cut_off_index = row["prompt"].find(cut_off)
        if added_attributes_cut_off_index == -1 :
            continue
        else : 
            attribute_substring = row["prompt"][added_attributes_cut_off_index : added_attributes_cut_off_index + len(cut_off)] # Get the substring of the prompt that contains the added attributes
            attribute_substring = attribute_substring.split(", ")
            total_count += len(attribute_substring) # Count the number of added attributes
    return total_count, total_count / dataset.shape[0] # Return the total count and the average number of added attributes per row

def get_added_attribute_count_per_task(dataset : pd.DataFrame) :
    """
    Get the number of added attributes in the dataset per task.
    """

    total_count_dict = {}
    instance_dict = {}
    average_dict = {}

    cut_off = "additional attributes: "
    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0], desc="Counting added attributes per task", unit="object") :
        added_attributes_cut_off_index = row["prompt"].find(cut_off)
        task = row["Source File"]
        if added_attributes_cut_off_index == -1 :
            if instance_dict.get(task) is None :
                instance_dict[task] = 1
            else : 
                instance_dict[task] += 1
            if total_count_dict.get(task) is None :
                total_count_dict[task] = 0
        else : 
            attribute_substring = row["prompt"][added_attributes_cut_off_index : added_attributes_cut_off_index + len(cut_off)] # Get the substring of the prompt that contains the added attributes
            attribute_substring = attribute_substring.split(", ")

            if total_count_dict.get(task) is None :
                total_count_dict[task] = len(attribute_substring)
                instance_dict[task] = 1
            else : 
                total_count_dict[task] += len(attribute_substring)
                instance_dict[task] += 1
    # Calculate the average number of objects per task
    for task in total_count_dict.keys() :
        average_dict[task] = total_count_dict[task] / instance_dict[task]
    return total_count_dict, average_dict # Return the total count and the average number of objects per task

if __name__ == "__main__" : 
    path = "dataset/NBRR_dataset_extended_text.csv"
    path_2 = "dataset/NBRR_dataset.csv"
    data = setup.load_dataset(path)
    data_2 = setup.load_dataset(path_2)

    total_count, average_count = count_objects(data)
    print(f"Total number of objects in the extended dataset: {total_count}")
    print(f"Average number of objects per row in the extended dataset: {average_count}")

    total_count, average_count = count_objects(data_2)
    print(f"Total number of objects in the original dataset: {total_count}")
    print(f"Average number of objects per row in the original dataset: {average_count}")

    total_count_dict, average_count_dict = count_objects_per_task(data)
    print(f"Total number of objects per task in the extended dataset: {total_count_dict}")
    print(f"Average number of objects per task in the extended dataset: {average_count_dict}")

    total_count_dict, average_count_dict = count_objects_per_task(data_2)
    print(f"Total number of objects per task in the original dataset: {total_count_dict}")
    print(f"Average number of objects per task in the original dataset: {average_count_dict}")

    total_count, average_count = get_added_attribute_count(data)
    print(f"Total number of added attributes in the extended dataset: {total_count}")
    print(f"Average number of added attributes per row in the extended dataset: {average_count}")
    total_count, average_count = get_added_attribute_count(data_2)
    print(f"Total number of added attributes in the original dataset: {total_count}")
    print(f"Average number of added attributes per row in the original dataset: {average_count}")

    total_count_dict, average_count_dict = get_added_attribute_count_per_task(data)
    print(f"Total number of added attributes per task in the extended dataset: {total_count_dict}")
    print(f"Average number of added attributes per task in the extended dataset: {average_count_dict}")

    total_count_dict, average_count_dict = get_added_attribute_count_per_task(data_2)
    print(f"Total number of added attributes per task in the original dataset: {total_count_dict}")
    print(f"Average number of added attributes per task in the original dataset: {average_count_dict}")