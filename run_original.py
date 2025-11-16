import pandas as pd
import setup_functions
import evaluate 
from tqdm import tqdm
import dspy
import os

if __name__ == "__main__" : 
    model_name = "gpt-4.1-2025-04-14"
    dataset_path = "dataset/original_dataset_by_norm_group.csv"
    api_key = os.getenv("OPENAI_API_KEY")  # Provide your OpenAI API key via environment variable
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. If you plan to use OpenAI, set the OPENAI_API_KEY env var.")
    # uncomment the following line and comment the one after it if you want to use the Ollama API
    #lm = setup_functions.lm_setup(model_name)
    if api_key:
        lm = setup_functions.lm_setup_openai(model_name, api_key=api_key)
    else:
        # If no OpenAI key provided, fall back to Ollama if available
        lm = setup_functions.lm_setup(model_name)
    dataset = setup_functions.load_dataset(dataset_path)
    category_wise_responses = {}

    category_wise_performance = {}


    norm_list = list(dataset["Norm"])
    # Extract only the Norm: N part from the norm string
    for i in range(len(norm_list)) :
        norm_list[i] = norm_list[i].split(":")[0]
    
    dataset["Norm"] = norm_list

    zero_shot = dspy.Predict(setup_functions.ZeroShotOriginal)

    for index, row in tqdm(dataset.iterrows(), total = dataset.shape[0]) : 
        response = zero_shot(scene = row["Block"])["target_object"]

        if row["Norm"] not in category_wise_responses : 
            category_wise_responses[row["Norm"]] = [response]
        else :
            category_wise_responses[row["Norm"]].append(response)
        
        if response == row["Asterisked Option"] : 
            if row["Norm"] not in category_wise_performance : 
                category_wise_performance[row["Norm"]] = [1]
            else :
                category_wise_performance[row["Norm"]].append(1)
        else : 
            if row["Norm"] not in category_wise_performance : 
                category_wise_performance[row["Norm"]] = [0]
            else :
                category_wise_performance[row["Norm"]].append(0)
        
    response_df = pd.DataFrame.from_dict(category_wise_responses, orient='index')
    response_df.to_csv(model_name + "category_wise_responses.csv", index=False)

    overall_accuracy = 0
    overall_count = 0

    for key in category_wise_performance :
        print(key + " Accuracy :", end = " ")
        print(sum(category_wise_performance[key]) / len(category_wise_performance[key]))
        overall_accuracy += sum(category_wise_performance[key])
        overall_count += len(category_wise_performance[key])
    
    print("Overall Accuracy :", end = " ")
    print(overall_accuracy / overall_count)
