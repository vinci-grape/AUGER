import os
import re
import ast
import torch
import argparse
import subprocess
import pandas as pd
from tqdm import tqdm
from itertools import product
from pastalib.pasta import PASTA 
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset, SequentialSampler

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def custom_collate(batch):
    return batch

def fault_localization(defective_method, fixed_method):
    with open(f'./cache/defective_method.java', 'w') as f:
        f.write(defective_method+'\n')
    with open(f'./cache/fixed_method.java', 'w') as f:
        f.write(fixed_method+'\n')

    diff_result = subprocess.run(['git', 'diff', f'./cache/defective_method.java', f'./cache/fixed_method.java'], stdout=subprocess.PIPE)
                                
    diff = diff_result.stdout.decode()
    pattern = r'@@ -?\d+,\d+ \+?\d+,\d+ @@(?:[^@]|@(?!@))+?(?=@@|$)'
    modifies = re.findall(pattern, diff)
    modified_code_string = []
    end_ = 0
    emphasized_text = []    
    for idx, modify in enumerate(modifies):
        pattern2 = r'@@ -(\d+),(\d+) \+(\d+),(\d+) @@[^@]+.'
        match = re.search(pattern2, modify)
        start = int(match.group(1))
        delete = int(match.group(2))
        end = start + delete - 1
        patch = ""
        for line in modify[modify.find('\n')+1:].splitlines():
            if line.startswith("-"):
                if line.strip() == "-":
                    patch += line + "\n" 
                else:
                    patch += line + " // Defective Line\n"
                    emphasized_text.append(re.sub(r'^( |\-)', '', line + " // Defective Line", flags=re.M))
            elif not line.startswith("+"):
                patch += line + "\n"
        patch = re.sub(r'^( |\-)', '', patch, flags=re.M)
        if idx == 0 and len(modifies) != 1:
            modified_code_string = defective_method.splitlines(True)[: start-1]
            modified_code_string.extend(patch.splitlines(True)) 
        elif idx == len(modifies) - 1:
            new_lines = defective_method.splitlines(True)[end_: start-1]
            new_lines.extend(patch.splitlines(True)) 
            modified_code_string.extend(new_lines)
            modified_code_string.extend(defective_method.splitlines(True)[end:])
        else:
            new_lines = defective_method.splitlines(True)[end_: start-1]
            new_lines.extend(patch.splitlines(True)) 
            modified_code_string.extend(new_lines)
        end_ = end
    modify_function = ''.join(modified_code_string)
    return modify_function.strip(), emphasized_text


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 class_name,
                 class_constructor,
                 defective_method,
                 fixed_method,
    ):
        self.class_name = class_name
        self.class_constructor = class_constructor
        self.defective_method = defective_method
        self.fixed_method = fixed_method


class TextDataset(Dataset):
    def __init__(self, args):
        self.examples = []
            
        df = pd.read_csv(args.dataset)
        df = pd.concat([df] * args.sample, ignore_index=True, axis=0)
        
        for _, row in df.iterrows():
            self.examples.append(InputFeatures(row["class_name"], row["class_constructor"], row["defective_method"], row["fixed_method"]))
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        class_name = self.examples[i].class_name
        class_constructor = self.examples[i].class_constructor
        defective_method = self.examples[i].defective_method
        fixed_method = self.examples[i].fixed_method
        return (class_name, class_constructor, defective_method, fixed_method)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--dataset", default="./data/RWB.csv", type=str,
                        help="")    
    parser.add_argument("--max_length", default=1024, type=int,
                        help="")    
    parser.add_argument("--batch_size", default=16, type=int,
                        help="The input image data file.")  
    parser.add_argument("--sample", default=100, type=int,
                        help="The input image data file.")  
    
    args = parser.parse_args()

    # Initialize pre-trained LLM
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True, device_map='auto', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
    
    with open(f"./data/prompt.txt") as f:
        origin_prompt = f.read()

    # Load data
    profiling_dataset = TextDataset(args)
    profiling_sampler = SequentialSampler(profiling_dataset)
    profiling_dataloader = DataLoader(profiling_dataset, sampler=profiling_sampler, batch_size=args.batch_size, num_workers=32, collate_fn=custom_collate)

    head_config = {
        "0": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }

    # Initialize
    pasta = PASTA(
        model=model,
        tokenizer=tokenizer,
        head_config=head_config, 
        alpha=0.01, # scaling coefficient
        scale_position="exclude", # downweighting unselected tokens
    )  
    
    test_case = []
    for _, batch in tqdm(enumerate(profiling_dataloader), total=len(profiling_dataloader)): 
        texts = []
        # User highlights specific input spans
        emphasized_texts = []
        for class_name, class_constructor, defective_method, fixed_method in batch:
            defective_method, emphasized_text = fault_localization(defective_method, fixed_method)
            texts.append(origin_prompt.format(class_name=class_name, class_constructor=class_constructor.strip() if isinstance(class_constructor, str) else class_constructor, defective_method=defective_method))
            emphasized_texts.append(emphasized_text)
                
        inputs, offset_mapping = pasta.inputs_from_batch(texts, max_length=args.max_length, device=model.device)
        with pasta.apply_steering(
            model=model, 
            strings=texts, 
            substrings=emphasized_texts, 
            model_input=inputs, 
            offsets_mapping=offset_mapping
        ) as steered_model:
            outputs = steered_model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=256, temperature=1, do_sample=True, top_p=0.95, top_k=50)
        for i in range(len(outputs)):
            output = tokenizer.decode(outputs[i], skip_special_tokens=True)
            test_case.append(output)
        torch.cuda.empty_cache()

    df = pd.concat([df] * args.sample, ignore_index=True, axis=0)
    df["test_case"] = test_case
    df = df[["bug_id", "test_case"]]
    df.to_csv(f"./results/RWB.csv")


if __name__ == "__main__":
    main()