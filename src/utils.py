import json
import os.path as osp
from typing import Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
import pickle

from dataset import random_sample

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("/project/phan/nbn4/grad-transformer/templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res
        
    def generate_mbpp_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        res = self.template["prompt_input"].format(
            instruction=instruction, tests=input,code=label  
        )
      
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

def get_hms(seconds: int) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}"

def make_chat_prompt(
    task_prompt: str,
    instruction_prefix: str,
    response_prefix: str,
    tokenizer: AutoTokenizer,
    thinking: str = None,
) -> str:
    # directly return prompt if it does not have a tokenizer.chat_template
    if tokenizer.chat_template is None:
        return task_prompt

    assert instruction_prefix is not None, "Instruction prefix is required!"
    assert response_prefix is not None, "Response prefix is required!"

    if thinking == None:
        thinking=""
    else:
        thinking=f"<think>\n{thinking}\n</think>"

    task_prompt = f"""\
{instruction_prefix}
{task_prompt.strip()}
"""
    response = f""
    return task_prompt, response

def generate_and_tokenize_prompt(batch, prompter, tokenizer):
    # Initialize lists for each column
    instruction_list = []
    input_list = []
    output_list = []
    instruction_prefix = ""
    response_prefix = ""
    # Iterate through each data point in the batch
    for question, answer in zip(batch["question"], batch["answer"]):
        instruction = question
        response = answer
        # if input_text != "":
        #     system = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
        # else:
        system = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        
        # Generate full prompt
        full_prompt = prompter.generate_prompt(
            instruction,
            "",
            "",
        )
        think=None
        # try:
        #     response = think.split("</think>")[1]
        # except:
        #     response = output
        # think = think.split("</think>")[0]
        
        
        # Create chat messages
        instruction, think = make_chat_prompt(full_prompt[len(system)+2:], instruction_prefix, response_prefix, tokenizer,think)
        # print(f"Before: {messages}")
        
        # Append output to messages
        if isinstance(response, list):
            response += "".join(response)  # Join list into a string
        else:
            response =  think + response  # Append string directly
        
        # instruction = messages.split("<|im_start|>assistant")[0].split("")
        instruction_list.append(instruction)
        input_list.append("")
        output_list.append(response)
        # print(f"After: {messages}")
            
    return {
        "instruction": instruction_list,
        "input": input_list,
        "output": output_list
    }

def init_lora_B_gaussian(m, mean=0.0, std=0.02):
    if hasattr(m, "lora_B"):
        for _name, lin in m.lora_B.items():
            if hasattr(lin, "weight") and lin.weight is not None:
                torch.nn.init.normal_(lin.weight, mean=mean, std=std)

def random_split(
        length, 
        num_splits,
        save_path,
        train_test_split=0.8, 
        seed=42):
    random.seed(seed)
    indices = list(range(length))
    random.shuffle(indices)
    split_size = length // num_splits

    splits = []
    for i in range(num_splits):
        start = i * split_size
        end = (i + 1) * split_size if i < num_splits - 1 else length
        splits.append(indices[start:end])
    
    idx_dict = {}
    for i in range(len(splits)):
        idx_dict[i] = {
            'train': splits[i][:int(train_test_split * len(splits[i]))],
            'test': splits[i][int(train_test_split * len(splits[i])):]
        }
    
    with open(save_path, 'wb') as file:
        pickle.dump(idx_dict, file)

    return idx_dict