import json
import numpy as np
import time
from tqdm import tqdm
import requests
import re
import evaluate
import argparse

class ChatGPTTextGenerator:
    def __init__(self, api_key, 
                       model = "gpt-3.5-turbo-0301", temperature = 0.1, 
                       api_address = 'https://api.openai.com/v1/chat/completions',
                       timeout = 15.0,
                       max_num_try = 5
                ):
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        }
        self.model = model
        self.temperature = temperature
        self.api_address = api_address
        self.timeout = timeout
        self.max_num_try = max_num_try
        
    def get_query(self, premises):
        query = """
Your task:
Please generate a conclusion text that can be drawn from the given premises: %s

Requirements:
1. Only infer the conclusion text from the given premises. 
2. The conclusion text should not contain any irrelevant decorative text. For example, if the conclusion you inferred is "Pluto is not a planet.", do not response with "The conclusion can be drawn from the given premises is that Pluto is not a planet.". Text like "This conclusion can be drawn from the given premises is that" should not be part of the generated conclusion text. 

Please return only the generated conclusion text. 
        """ % (
             premises
        )
        return query
    
    def generate( self, premises):
        
        data = {
          "model": self.model,
          "temperature":self.temperature,
          "max_tokens":128,
          "messages": [
                {"role": "system", "content": "You are a scientific writing assistant. Your task is to generate a conclusion text for given premises. "},
                {"role": "user", "content": self.get_query( premises )},
            ]
        }
        
        generated_conclusion = None
        for try_id in range( self.max_num_try ):
            if try_id > 0:
                print("Retrying ... %d"%( try_id ))
            try:
                response = requests.post(self.api_address, headers=self.headers, data=json.dumps(data), timeout=self.timeout)
                generated_conclusion = response.json()["choices"][0]["message"]["content"]
                break
            except:
                continue
                
        return generated_conclusion
    
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default='REPLACE_WITH_YOURS' )
    parser.add_argument("--data_path", type=str, default='./dataset/PMCOA-Feb23-2022-test-mask-nppl.jsonl' ) 
    parser.add_argument("--save_path", type=str, default='./dataset/chatgpt-test-mask-nppl.jsonl' )
    parser.add_argument("--start", type=int, default = None )
    parser.add_argument("--size", type=int, default = None )
    
    args = parser.parse_args()
        
    chatgpt_generator = ChatGPTTextGenerator( args.api_key )

    test_data = []
    with open(args.data_path, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            premise = " ".join([data['premise'][idx] for idx in np.argwhere(data['label']) [:, 0]])
            #premise = " ".join(data['premise']) # full-premises
            conclusion = " ".join(data['conclusion'])
            id = data['pubmed_id']
            test_data.append([premise, conclusion, id])
    
    if args.start is None:
        args.start = 0
        args.size = len(test_data)
    else:
        assert args.size is not None
        args.save_path = args.save_path + "_%d"%(args.start)
        
    fw = open(args.save_path, "w")

    for example in tqdm(test_data[ args.start : args.start + args.size ]):
        
        write_dict = {}

        gen_con = chatgpt_generator.generate( 
             " ".join(example[0])
        )
        
        write_dict["pubmed_id"] = example[2]
        write_dict["premise"] = example[0]
        write_dict['org-con'] = example[1]
        write_dict['gen-con'] = gen_con
       
        fw.write( json.dumps( write_dict ) + "\n" )
        
        
    fw.close()        
