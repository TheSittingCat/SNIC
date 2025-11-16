import pandas as pd
import dspy
import numpy as np 
from typing import Literal
import os


def list_from_folder(folder_path : str) : 
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))] # list all files in a folder

def combine_csvs(file_list : list) : 
    combined_df = pd.DataFrame()

    for file in file_list : 
        df = pd.read_csv(file)
        if "referent_object" in df.columns:
            df = df.rename(columns={"referent_object": "target_object"})
        if file.rstrip(".csv").endswith("c1") :  # check if the file is from group c1
            group_name = "c1"
        elif file.rstrip(".csv").endswith("c2") : 
            group_name = "c2"
        else : 
            group_name = file.rstrip(".csv")[-1] # get the last character of the file name

        df["group"] = group_name # assign the group name to the dataframe
        combined_df = pd.concat([combined_df, df])
    return combined_df

def lm_setup(model_name : str) : 
    # If you are running Ollama with an API key, supply it via the OLLAMA_API_KEY env var. Otherwise, an empty string is OK for local Ollama.
    ollama_api_key = os.getenv("OLLAMA_API_KEY", "")
    lm = dspy.LM("ollama_chat/" + model_name, api_base='http://localhost:11434', api_key = ollama_api_key, cache = False)
    dspy.configure(lm = lm, temperature = 0)
    return lm

def lm_setup_openai(model_name : str, api_key : str) : 
    lm = dspy.LM("openai/" + model_name, api_key = api_key, cache = False)
    dspy.configure(lm = lm, temperature = 0)
    return lm

def load_dataset(path : str):
    # load the dataset
    data = pd.read_csv(path)
    return data

def class_names_to_objects(data : list) : 
    objects = ["obj" + str(i + 1) for i in range(len(data))] # create a list of objects
    return objects

class ZeroShot(dspy.Signature) : 
    # Uncomment the following line and comment the one after it if you wish to evaluate the model without ICL.
    """ Given the scene as well as the Prolog code associated with the scene, use the scene info and the prolog code to choose the object that is being referred to. Return the object number."""

    scene : str = dspy.InputField(desc = "The scene that the agent is in.")
    prolog_code : str = dspy.InputField(desc = "The Prolog code that represents the scene.")
   # options : list = dspy.InputField(desc = "The list of objects that the agent can choose from.")
    target_object : int = dspy.OutputField(desc = "ONLY The object number that the agent is referring to.") # We should upgrade to Python 3.11 to allow for unpacking operator in type hints

class ZeroShotICL(dspy.Signature) : # DSPY uses the class docstring as the prompt so the functions are not exactly the same
    """ Given the scene as well as the Prolog code associated with the scene, use the scene info and the prolog code to choose the object that is being referred to. Return the object number. Use the following norms to make your choice: 
    Norm 1: In a serving task, you should serve with kitchen items that are clean
    Norm 2: In a cleaning task, you should clean items that are dirty
    Norm 3: In a cooking task, you should cook with tools that are clean
    Norm 4: In a serving task you should not use items that are decorative
    Norm 5: In a cleaning task, you should not clean items that are currently being used by someone
    Norm 6: In a cleaning task you should prioritize cleaning up any hazards (e.g. a broken plate on the ground)
    Norm 7: In general, you should prioritsize cleaning up any hazards
    Norm Conflict 1: In a cleaning task, you should not touch something that isn't yours unless you are preventing danger
    Norm Conflict 2: In a tidying task, you should prioritize cleaning up hazards over dirty items
    Norm Conflict 3: In a tidying task in a library, you prioritize removing dirty items from the floor over books and other objects"""
    
    scene : str = dspy.InputField(desc = "The scene that the agent is in.")
    prolog_code : str = dspy.InputField(desc = "The Prolog code that represents the scene.")
   # options : list = dspy.InputField(desc = "The list of objects that the agent can choose from.")
    target_object : int = dspy.OutputField(desc = "ONLY The object number that the agent is referring to.") # We should upgrade to Python 3.11 to allow for unpacking operator in type hints

class ZeroShotOriginal(dspy.Signature) :
    """ Given a scenario and a list of objects, choose the object number that the human speaker is referring to. Return only the object number. """
    scene : str = dspy.InputField(desc = "The scene that the agent is in.")
    target_object : int = dspy.OutputField(desc = "The object number that the agent is referring to.")


class ZeroShotNoProlog(dspy.Signature) :
    """ Given the scene, use it to choose the object that is being referred to. Return the object number."""

    scene : str = dspy.InputField(desc = "The scene that the agent is in.")
    # options : list = dspy.InputField(desc = "The list of objects that the agent can choose from.")
    target_object : int = dspy.OutputField(desc = "ONLY The object number that the agent is referring to.")

class ZeroShotNoPrologICL(dspy.Signature) :
    """ Given the scene, use it to choose the object that is being referred to. Return the object number. Use the following norms to make your choice: 
    # Norm 1: In a serving task, you should serve with kitchen items that are clean
    # Norm 2: In a cleaning task, you should clean items that are dirty
    # Norm 3: In a cooking task, you should cook with tools that are clean
    # Norm 4: In a serving task you should not use items that are decorative
    # Norm 5: In a cleaning task, you should not clean items that are currently being used by someone
    # Norm 6: In a cleaning task you should prioritize cleaning up any hazards (e.g. a broken plate on the ground)
    # Norm 7: In general, you should prioritize cleaning up any hazards
    # Norm Conflict 1: In a cleaning task, you should not touch something that isn't yours unless you are preventing danger
    # Norm Conflict 2: In a tidying task, you should prioritize cleaning up hazards over dirty items
    # Norm Conflict 3: In a tidying task in a library, you prioritize removing dirty items from the floor over books and other objects"""

    scene : str = dspy.InputField(desc = "The scene that the agent is in.")
   # options : list = dspy.InputField(desc = "The list of objects that the agent can choose from.")
    target_object : int = dspy.OutputField(desc = "ONLY The object number that the agent is referring to.")

class ZeroShotReasoning(dspy.Signature) :
    """ Given the scene, use it to choose the object that is being referred to. Return the object number. Use the following norms to make your choice: 
    Norm 1: In a serving task, you should serve with kitchen items that are clean
    Norm 2: In a cleaning task, you should clean items that are dirty
    Norm 3: In a cooking task, you should cook with tools that are clean
    Norm 4: In a serving task you should not use items that are decorative
    Norm 5: In a cleaning task, you should not clean items that are currently being used by someone
    Norm 6: In a cleaning task you should prioritize cleaning up any hazards (e.g. a broken plate on the ground)
    Norm 7: In general, you should prioritize cleaning up any hazards
    Norm Conflict 1: In a cleaning task, you should not touch something that isn't yours unless you are preventing danger
    Norm Conflict 2: In a tidying task, you should prioritize cleaning up hazards over dirty items
    Norm Conflict 3: In a tidying task in a library, you prioritize removing dirty items from the floor over books and other objects"""

    scene : str = dspy.InputField(desc = "The scene that the agent is in.")
    prolog_code : str = dspy.InputField(desc = "The Prolog code that represents the scene.")
   # options : list = dspy.InputField(desc = "The list of objects that the agent can choose from.")
    target_object : int = dspy.OutputField(desc = "ONLY The object number that the agent is referring to.")
    reasoning : str = dspy.OutputField(desc = "The reasoning behind the choice of the object.")

def accuracy(responses : list, targets : list) : 
    correct = 0
    if isinstance(responses[0], str) : # Assuming the responses are strings
        for i in range(len(responses)) : 
            if responses[i].lower() == targets[i].lower() : 
                correct += 1
    elif isinstance(responses[0], int) :  # Assuming the responses are integers
        for i in range(len(responses)) : 
            if responses[i] == targets[i] : 
                correct += 1
    return correct / len(responses) 