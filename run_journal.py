import os
import pandas as pd
import setup_functions
import evaluate 
from tqdm import tqdm
import dspy

if __name__ == "__main__" : 
    model_name = "granite3.3:2b"
    dataset_path = "/home/keskan01/acl_2025_norm/dataset/journal_dataset/combined/combined_dataset_numbered_clean_v4.csv"

    use_prolog = input("Do you want to use Prolog? (y/n): ")

    reasoning_mode = input("Do you want to get the reasoning behind the answer? (y/n): ")

    use_icl = input("Do you want to use ICL? (y/n): ")

    if reasoning_mode.lower() == "y":
        reasoning = []

    api_key = os.getenv("OPENAI_API_KEY")
    # uncomment the following line and comment the one after it if you want to use the Ollama API
    lm = setup_functions.lm_setup(model_name)
    #lm = setup_functions.lm_setup_openai(model_name, api_key=api_key)
    dataset = setup_functions.load_dataset(dataset_path)
    context = list(dataset["prompt_numbered"])
    # Uncomment the following line if the dataset has a prolog column or you want to use it
    if use_prolog.lower() == "y":
        prolog_code = list(dataset["prolog"])
    # candidates = list(dataset["all_objects"])

    # for i in range(len(candidates)) :
    #     candidates[i] = candidates[i].strip().split(", ")
    # uncomment the following line if you want to work with objects rather than descriptions and comment the one after it

    #targets = list(dataset["target_object"])
    targets = list(dataset["answer_number"])
    targets = [int(t) for t in targets]
    source = list(dataset["group"])
    responses = []

    for prompt_num in tqdm(range(len(context))) : 
        
        # If you are using the prolog code, uncomment the following line and comment the one after it

        if reasoning_mode.lower() == "y":
            zero_shot = dspy.Predict(setup_functions.ZeroShotReasoning)
        
        else : 
            if use_prolog.lower() == "y":
                if use_icl.lower() == "y":
                    zero_shot = dspy.Predict(setup_functions.ZeroShotICL)
                elif use_icl.lower() == "n":
                    zero_shot = dspy.Predict(setup_functions.ZeroShot)
            elif use_prolog.lower() == "n": 
                if use_icl.lower() == "y":
                    zero_shot = dspy.Predict(setup_functions.ZeroShotNoPrologICL)
                elif use_icl.lower() == "n":
                    zero_shot = dspy.Predict(setup_functions.ZeroShotNoProlog)
        try:
            # If you are using the prolog code, uncomment the following line and comment the one after it
            if use_prolog.lower() == "y":
                response = zero_shot(scene = context[prompt_num], prolog_code = prolog_code[prompt_num])
            elif use_prolog.lower() == "n":
                response = zero_shot(scene = context[prompt_num])
            responses.append(int(response.target_object))
            if reasoning_mode.lower() == "y":
                reasoning.append(response.reasoning)
            # for candidate in object_names :
            #     if candidate.lower() in responses[-1] : # This is just for normalizing the output, add the object version rather than the expanded version
            #         responses[-1] = candidate.lower()
        except KeyboardInterrupt : 
            break
        except Exception as e :
            try : 
                print("An error occurred while generating the response, retrying.")
                lm = setup_functions.lm_setup(model_name) # Reset the model
                # If you are using the prolog code, uncomment the following line and comment the one after it
                if use_prolog.lower() == "y":
                    retry_response = zero_shot(scene = "Please make sure to strictly adhere to the output format. scene :" + context[prompt_num], prolog_code = prolog_code[prompt_num])
                elif use_prolog.lower() == "n":
                    retry_response = zero_shot(scene = "Please make sure to strictly adhere to the output format. scene :" + context[prompt_num])
                
                if reasoning_mode.lower() == "y":
                    reasoning.append(retry_response.reasoning)
                responses.append(int(retry_response.target_object))
                # for candidate in object_names :
                #     if candidate in responses[-1] :
                #         responses[-1] = candidate
            except Exception as e :
                print(e)
                print("An error occurred while generating the response, this is likely due to the model not adhering to the output signature after a retry.")
                responses.append("Error")
    accuracies = {}
    start = 0
    end = 0
    for i in range(len(source)) : 
        if  i != 0 :
            if source[i] != source[i-1] : 
                end = i
                accuracies[source[i-1]] = setup_functions.accuracy(responses[start:end], targets[start:end])
                start = i
        if i == len(source) - 1 :
            end = i
            accuracies[source[i]] = setup_functions.accuracy(responses[start:end], targets[start:end])
    accuracies["Overall"] = setup_functions.accuracy(responses, targets)
    print(dict(sorted(accuracies.items(), key=lambda item: str(item[0]))))
    try :

        if reasoning_mode.lower() == "y":
            pd.DataFrame([responses, reasoning], index=["Response", "Reasoning"]).T.to_csv("results/responses_journal_All_Reasoning_new_With_Prolog" + model_name + ".csv", index = False)
        else :
            if use_prolog.lower() == "y":
                pd.DataFrame(responses, columns = ["Response"]).to_csv("results/responses_journal_Desc_new_With_Prolog" + model_name + ".csv", index = False)
            elif use_prolog.lower() == "n":
                pd.DataFrame(responses, columns = ["Response"]).to_csv("results/responses_journal_Desc_new" + model_name + ".csv", index = False)
        if reasoning_mode.lower() == "y":
            pd.DataFrame(accuracies.items(), columns = ["Source", "Accuracy"]).to_csv("results/accuracies_journal_Reasoning_new_With_Prolog" + model_name + ".csv", index = False)
        else :
            if use_prolog.lower() == "y":
                pd.DataFrame(accuracies.items(), columns = ["Source", "Accuracy"]).to_csv("results/accuracies_journal_Desc_new_With_Prolog" + model_name + ".csv", index = False)
            elif use_prolog.lower() == "n":
                pd.DataFrame(accuracies.items(), columns = ["Source", "Accuracy"]).to_csv("results/accuracies_journal_Desc_new" + model_name + ".csv", index = False)
    except : 
        print("An error occurred while saving the results, check the model name and path.")