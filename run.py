import pandas as pd
import setup_functions
import evaluate 
from tqdm import tqdm
import dspy
import os

if __name__ == "__main__" : 
    model_name = "phi4-mini"
    dataset_path = "dataset/NBRR_dataset.csv"
    api_key = os.getenv("OPENAI_API_KEY", "")
    # uncomment the following line and comment the one after it if you want to use the Ollama API
    lm = setup_functions.lm_setup(model_name)
    # Use OpenAI if OPENAI_API_KEY is set, otherwise use local Ollama
    if api_key:
        lm = setup_functions.lm_setup_openai(model_name, api_key=api_key)
    else:
        lm = setup_functions.lm_setup(model_name)
    dataset = setup_functions.load_dataset(dataset_path)
    context = list(dataset["prompt"])
    # Uncomment the following line if the dataset has a prolog column or you want to use it
    prolog_code = list(dataset["prolog"])
    candidates = list(dataset["all_objects"])
    for i in range(len(candidates)) :
        candidates[i] = candidates[i].strip().split(", ")
    # uncomment the following line if you want to work with objects rather than descriptions and comment the one after it
    targets = list(dataset["target_object"])
    #targets = list(dataset["target_description"])
    source = list(dataset["Source File"])
    responses = []

    for prompt_num in tqdm(range(len(context))) : 
        # If you are using the prolog code, uncomment the following line and comment the one after it
        object_names = setup_functions.class_names_to_objects(candidates[prompt_num])
        #object_names = candidates[prompt_num] 
        # If you are using the prolog code, uncomment the following line and comment the one after it
        zero_shot = dspy.Predict(setup_functions.ZeroShot)
        #zero_shot = dspy.Predict(setup_functions.ZeroShotNoProlog)
        try : 
            # If you are using the prolog code, uncomment the following line and comment the one after it
            response = zero_shot(scene = context[prompt_num], prolog_code = prolog_code[prompt_num], options = object_names)
            #response = zero_shot(scene = context[prompt_num], options = object_names)
            responses.append(response.target_object.lower())
            for candidate in object_names :
                if candidate.lower() in responses[-1] : # This is just for normalizing the output, add the object version rather than the expanded version
                    responses[-1] = candidate.lower()
        except KeyboardInterrupt : 
            break
        except Exception as e :
            try : 
                print("An error occurred while generating the response, retrying.")
                lm = setup_functions.lm_setup(model_name) # Reset the model
                # If you are using the prolog code, uncomment the following line and comment the one after it
                retry_response = zero_shot(scene = "Please make sure to strictly adhere to the output format. scene :" + context[prompt_num], prolog_code = prolog_code[prompt_num], options = object_names)
                #retry_response = zero_shot(scene = "Please make sure to strictly adhere to the output format. scene :" + context[prompt_num], options = object_names)
                responses.append(retry_response.target_object)
                for candidate in object_names :
                    if candidate in responses[-1] :
                        responses[-1] = candidate
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
    print(accuracies)
    try :
        pd.DataFrame(responses, columns = ["Response"]).to_csv("results/responses_norm_extended_" + model_name + ".csv", index = False)
        pd.DataFrame(accuracies.items(), columns = ["Source", "Accuracy"]).to_csv("results/accuracies_norm_extended" + model_name + ".csv", index = False)
    except : 
        print("An error occurred while saving the results, check the model name and path.")