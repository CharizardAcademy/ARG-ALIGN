from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator
import torch
from tqdm import tqdm
import numpy as np

import argparse

import os
import ctranslate2
import sentencepiece as spm

from peft import PeftModel

import evaluate
import json

def concatenate_premise(example):
    concat_premise = ""
    for idx, label in enumerate(example['label']):
        if label:
            concat_premise += example['premise'][idx] + " "
        else:
            concat_premise += "<sentencemissing>" + " "
    return concat_premise.rstrip()

class TextGeneratorFast:
    
    def __init__(self, model_path):
        self.model = ctranslate2.Generator(model_path, device="cuda" if torch.cuda.is_available() else "cpu" )
        #self.tokenizer = spm.SentencePieceProcessor(os.path.join(model_path, "tokenizer.model"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        
    def generate( self,  
                  premise = "",
                  sampling_temperature=0.1,
                  sampling_topk=1,
                  num_beams=2
                ):
        
        prompt = "### Premise:\n"+premise+ "\n\n" + "### Conclusion:\n" 
        
        #input_tokens = self.tokenizer.encode(prompt, out_type=str)
        input_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(prompt))                    


        # step_results = self.model.generate_tokens(
        #         input_tokens,
        #         sampling_temperature=sampling_temperature,
        #         sampling_topk=sampling_topk,
        #         max_length=150,
        #     )

        step_results = self.model.generate_batch(
                [input_tokens],
                sampling_temperature=sampling_temperature,
                sampling_topk=sampling_topk,
                beam_size=num_beams,
                max_length=150,
                include_prompt_in_result=False
        )
                    
        # conclusion = self.tokenizer.decode(list(map(lambda x:x.token_id, step_results[0]))).split("\n")[0]
        conclusion = self.tokenizer.decode(step_results[0].sequences_ids[0]).split("\n")[0]
     
        return conclusion 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--ct2_model_path", type=str, default="./fine-tuned-models/five/llama-7B-ct2/")
    parser.add_argument("--ct2_model_path", type=str, default="./fine-tuned-models/five/galactica-6.7B-ct2")
    parser.add_argument("--test_data_path", type=str, default="./data/PMCOA-Feb23-2022-test-mask-rand.jsonl")
    args = parser.parse_args()

    model = TextGeneratorFast(args.ct2_model_path)
    test_data = [json.loads(line) for line in open(args.test_data_path, 'r')]
    rouge = evaluate.load("rouge")

    references =[]
    predicions = []

    for example in tqdm( test_data[:] ):
        #premise = " ".join( [example["premise"][idx] for idx in np.argwhere( example["label"] )[:,0]])
        #premise = concatenate_premise(example)
        #premise = " ".join(example["premise"])
        premise = " ".join(example["premise"][:5])
        conclusion = model.generate(premise)
        references.append( " ".join( example["conclusion"] ) )
        predicions.append( conclusion )
        
    print(rouge.compute( predictions=predicions, references=references ))
