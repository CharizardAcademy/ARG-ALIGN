import argparse
import os

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed
from transformers import Trainer, BitsAndBytesConfig, TrainerCallback

import numpy as np
import torch

import json

class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if args.should_save:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            kwargs["model"].save_pretrained(checkpoint_path)

            if "pytorch_model.bin" in os.listdir(checkpoint_path):
                os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))

def concatenate_premise(example):
    concat_premise = ""
    for idx, label in enumerate(example['label']):
        if label:
            concat_premise += example['premise'][idx] + " "
        else:
            concat_premise += "<sentencemissing>" + " "
    return concat_premise.rstrip()
    

def get_prompt(example, eos_token, only_premise = False ):
    """Prepare the text from a sample of the dataset."""
    # premise = " ".join( [ example["premise"][idx] for idx in np.argwhere( example["label"] )[:,0] ] )
    # premise =  concatenate_premise(example)
    premise = " ".join( example["premise"][:5] )
    conclusion = " ".join( example["conclusion"] )
    if only_premise:
        text = "### Premise:\n"+premise+ "\n\n" + "### Conclusion:\n"
    else:
        text = "### Premise:\n"+premise+ "\n\n" + "### Conclusion:\n" + conclusion + " " + eos_token
    return text

def tokenize_example( example ):
    global tokenizer, args
    input_ids = np.array(tokenizer( get_prompt(example, tokenizer.eos_token , only_premise = False  ) ).input_ids)
    labels = np.array([-100] * len(input_ids))

    start_pos = len( tokenizer( get_prompt(example, tokenizer.eos_token, only_premise = True) ).input_ids )
    labels[ start_pos: ] = input_ids[start_pos: ]
    
    input_ids = input_ids.tolist()[:args.max_length]
    labels  = labels.tolist()[:args.max_length]

    return {"input_ids":input_ids, "labels":labels}

class DataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
        
    def __call__(self, examples  ):
        input_ids_list = [ example["input_ids"] for example in examples ]
        labels_list = [ example["labels"] for example in examples ]
        
        padded_length = max( [ len(input_ids) for input_ids in input_ids_list] )
        
        padded_input_ids_list = []
        attention_mask_list = []
        padded_labels_list = []

        for input_ids, labels in zip(input_ids_list, labels_list):
            
            attention_mask = [ 1 ] * len( input_ids ) + [ 0 ] * ( padded_length - len( input_ids ) )
            padded_input_ids = input_ids + [ self.pad_token_id ] * ( padded_length - len(input_ids) )
            padded_labels = labels + [ -100 ] * ( padded_length - len( labels ) )
            
            padded_input_ids_list.append( padded_input_ids )
            attention_mask_list.append( attention_mask )
            padded_labels_list.append( padded_labels )            

        return {
            "input_ids": torch.LongTensor( padded_input_ids_list ),
            "attention_mask": torch.LongTensor( attention_mask_list ),
            "labels": torch.LongTensor( padded_labels_list )
        }

# class Args(object):
#     pass

# args = Args()

# args.model_path = "huggyllama/llama-7b"
# args.output_dir = "model/llama-peft"
# args.max_steps = 500
# args.log_freq = 1 
# args.eval_freq = 0 
# args.save_freq = 0 
# args.streaming = 0 
# args.batch_size = 4
# args.gradient_accumulation_steps = 4
# args.train_dataset_name = "data/sci-abduction-abstract-train.jsonl"
# args.val_dataset_name = "data/sci-abduction-abstract-test.jsonl"
# args.quantization = "int4"
# args.shuffle_buffer = 5000
# args.max_length = 512
# args.num_train_epochs = 3
# args.learning_rate = 1e-5
# args.lr_scheduler_type = "cosine"
# args.num_warmup_steps = 100
# args.weight_decay = 0.05
# args.local_rank = 0
# args.fp16 = 1
# args.bf16 = 0
# args.gradient_checkpointing = 1
# args.seed = 0
# args.num_workers = 8
# args.lora_r = 16
# args.lora_alpha = 32
# args.lora_target_modules = ["q_proj","v_proj"]
# args.lora_dropout = 0.05

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default = "bigscience/bloom-7b1", help="huggyllama/llama-7b for llama, facebook/galactica-6.7b for galactica, bigscience/bloom-7b1 for bloom" )
    parser.add_argument("--output_dir", type=str, default = "fine-tuned-models/bloom-7.1B-peft/"  )
    parser.add_argument("--max_steps", type = int, default = -1, help = "if max_steps is set to -1, then num_train_epochs is effective, the default setting was 300") 
    parser.add_argument("--log_freq", type=int, default= 10 )
    parser.add_argument("--eval_freq", type=int, default=100 )
    parser.add_argument("--save_freq", type=int, default=100 )
    parser.add_argument("--streaming", type = int, default = 0 )
    parser.add_argument("--batch_size", type = int, default = 4)
    parser.add_argument("--gradient_accumulation_steps", type = int, default = 1)
    parser.add_argument("--train_dataset_name", type=str, default="data/PMCOA-Feb23-2022-train-mask.jsonl")  
    parser.add_argument("--val_dataset_name", type=str, default="data/PMCOA-Feb23-2022-dev-mask.jsonl")
    parser.add_argument("--quantization", default="int4")
    parser.add_argument("--shuffle_buffer", type=int, default = 5000)
    parser.add_argument("--max_length", type = int, default = 512)
    parser.add_argument("--num_train_epochs", type = int, default = 3, help = "To use num_train_epochs, we need to know the total length of the dataset. This is not compatible with the IterableDataset.")
    parser.add_argument("--learning_rate", type = float, default = 1e-5)
    parser.add_argument("--lr_scheduler_type", type = str, default = "cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--fp16", type = int, default = 1)
    parser.add_argument("--bf16", type = int, default = 0)
    parser.add_argument("--gradient_checkpointing", type = int, default = 1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_target_modules", default=["q_proj","v_proj"], type=str, nargs = "+")
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()
    
    config = AutoConfig.from_pretrained(args.model_path)
    architecture = config.architectures[0]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if tokenizer.eos_token is None:
        if "galactica" in args.model_path:
            tokenizer.add_special_tokens({
                    "bos_token":"<s>",
                    "eos_token":"</s>",
                    "pad_token":"<pad>",
                    "unk_token":"<unk>"
            })
        else:
            print("Error: no eos token pre-defined in the vocabulary. You need to add the eos_token and resize the model's embedding accordingly")
            assert False
        
    special_tokens = {}
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = tokenizer.eos_token
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = tokenizer.eos_token
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = tokenizer.unk_token
  
    tokenizer.add_special_tokens( special_tokens )

    train_data = load_dataset('json', 
                            data_files = args.train_dataset_name, 
                            split = 'train',
                            num_proc = args.num_workers if not args.streaming else None,
                            streaming = args.streaming
                     )
    
    if args.streaming:
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    else:
        train_data = train_data.shuffle( seed=args.seed)
    
    valid_data = load_dataset('json', 
                            data_files = args.val_dataset_name, 
                            split = 'train',
                            num_proc = args.num_workers if not args.streaming else None,
                            streaming = args.streaming
                     ) 
    for example in train_data:
        break
    exsiting_columns = list(example.keys())
    
    train_dataset = train_data.map( tokenize_example, remove_columns = exsiting_columns )
    valid_dataset = valid_data.map( tokenize_example, remove_columns = exsiting_columns )
    
    data_collator = DataCollator(tokenizer.pad_token_id)

    print("Loading the model")
        
    bnb_config = BitsAndBytesConfig(        
        load_in_8bit= args.quantization == "int8",
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False,
        load_in_4bit=args.quantization == "int4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
    )
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=bnb_config, device_map={"":torch.cuda.current_device()}
    )

    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_path, quantization_config=bnb_config, device_map="auto"
    # )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules = args.lora_target_modules,
        lora_dropout= args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="no" if args.eval_freq <= 0 else "steps",
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name="conclusion-gen",
        report_to="wandb",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer = tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator = data_collator,
        callbacks= [PeftSavingCallback]
    )

    model.config.use_cache = False
    
    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))
