import argparse
import os

import torch
import transformers
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path" )
    parser.add_argument("--lora_model_path" )
    parser.add_argument("--save_model_path" )
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.eos_token is None:
        print("Error: no eos token pre-defined in the vocabulary. You need to add the eos_token and resize the model's embedding accordingly")
        assert False
        
    special_tokens ={}
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = tokenizer.eos_token
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = tokenizer.eos_token
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = tokenizer.unk_token
    
    tokenizer.add_special_tokens( special_tokens )
    
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )

    first_weight = base_model.model.layers[0].self_attn.q_proj.weight
    first_weight_old = first_weight.clone()

    lora_model = PeftModel.from_pretrained(
        base_model,
        args.lora_model_path,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )

    lora_weight = lora_model.base_model.model.model.layers[
        0
    ].self_attn.q_proj.weight

    assert torch.allclose(first_weight_old, first_weight)

    # merge weights - new merging method from peft
    lora_model = lora_model.merge_and_unload()

    lora_model.train(False)

    # did we do anything?
    assert not torch.allclose(first_weight_old, first_weight)

    lora_model_sd = lora_model.state_dict()
    deloreanized_sd = {
        k.replace("base_model.model.", ""): v
        for k, v in lora_model_sd.items()
        if "lora" not in k
    }

    base_model.save_pretrained( args.save_model_path, state_dict=deloreanized_sd )
    tokenizer.save_pretrained( args.save_model_path )
    