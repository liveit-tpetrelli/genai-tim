import glob
import json
import os
from typing import List, Dict

from langchain_core.prompts import PromptTemplate, BasePromptTemplate
from langchain.prompts import load_prompt

from configs.app_configs import AppConfigs

app_configs = AppConfigs()
prompts_path = app_configs.configs.Prompts.path


def load_prompts_from_dir(path: str) -> Dict[str, BasePromptTemplate]:
    prompts = {}
    for prompt_file in glob.glob(os.path.join(path, "*.json")):
        prompt = load_prompt(prompt_file)
        key = prompt_file.split("\\")[-1][:-5]
        prompts[key] = prompt
    return prompts


def load_shots_from_dir(path: str) -> List[str]:
    with open(os.path.join(path, "few_shots.json"), "r") as file:
        data = json.load(file)
    return data["few_shosts"]


class PromptsRetrieval:
    prompts: Dict[str, BasePromptTemplate]
    few_shots_examples: List[str]

    def __init__(self, prompts_dir: str = os.path.join(*prompts_path)):
        self.prompts = load_prompts_from_dir(path=prompts_dir)
        self.few_shots_examples = load_shots_from_dir(path=os.path.join(prompts_dir, "shots"))
