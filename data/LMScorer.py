from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config
from transformers import AutoTokenizer, OPTForCausalLM, AutoModel
from transformers import GPTNeoForCausalLM
from transformers.tokenization_utils import BatchEncoding
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import math

class GPT2CLMScorer:
    
    ### for vanilla GPT-2 base model, set model_name to "gpt2"
    ### for fine-tuned GPT-2 base model, set model_name to "/path/to/your/model"
    ### GPT-2 base model num of params 117M
    
    def __init__( self, model_name="vanilla", device="cuda"):
        self.model_name = model_name
        #self.device = device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        if self.model_name != 'vanilla':
            self.scorer = GPT2LMHeadModel.from_pretrained(self.model_name)
            self.scorer.eval()
            self.scorer.to(self.device)
            ### fine-tuned GPT-2 uses "<|startoftext|>", "<|endoftext|>"  and '<|pad|>'as bos, eos, and pad token
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            
        else:
            self.config = GPT2Config.from_pretrained('gpt2', output_hidden_states=True)
            self.scorer = GPT2LMHeadModel.from_pretrained('gpt2', config=self.config)
            self.scorer.eval()
            self.scorer.to(self.device)
            ### vanilla GPT-2 uses no bos and eos token
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<|pad|>')
            
        self.scorer.resize_token_embeddings(len(self.tokenizer))

    def add_special_tokens(self, text):
        return text
        # if self.model_name != 'gpt2':
        #     return self.tokenizer.bos_token + text + self.tokenizer.eos_token
        # else:
        #     return text
            
    def sentence_score(self, input_text):
        input_text = list(map(self.add_special_tokens, [input_text]))
        encoding: BatchEncoding = self.tokenizer.batch_encode_plus(input_text, return_tensors="pt")  
        
        output = []
        with torch.no_grad():
            ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            nopad_mask = ids != self.tokenizer.pad_token_id
            
            logits: torch.Tensor = self.scorer(ids, attention_mask=attention_mask)[0]
            
        for sent_index in range(len(input_text)):
            sent_nopad_mask = nopad_mask[sent_index]
            
            sent_ids = ids[sent_index, sent_nopad_mask][1:]
            sent_logits = logits[sent_index, sent_nopad_mask][:-1, :]
            sent_logits[:, self.tokenizer.pad_token_id] = float("-inf")
            sent_ids_scores = sent_logits.gather(1, sent_ids.unsqueeze(1)).squeeze(1)
            
            sent_log_probs = sent_ids_scores - sent_logits.logsumexp(1)
            output.append(sent_log_probs.double())
            
        return output[0].sum()
    
    def compute_nppl_score(self, conclusion, premise, vanilla=False):
    
        if vanilla:
            concat = conclusion + " " + premise
        else:
            concat = conclusion + " <|endoftext|> " + premise + " <|endoftext|>"

        log_prob = self.sentence_score(concat)
        ppl = math.exp( - (log_prob / len(concat.split(" "))))

        vocab_size = len(self.tokenizer)
        # can work on this prior later on
        upper_log_prob = sum([math.log(1 / vocab_size)] * len(concat.split(" ")))
        upper_ppl = math.exp( - (upper_log_prob / len(concat.split(" "))))

        nppl = ppl / upper_ppl
        return nppl
    
    def compute_npmi_score(self, conclusion, premise, vanilla=False):
        if vanilla:
            concat = conclusion + " " + premise
        else:
            concat = conclusion + " <|endoftext|> " + premise + " <|endoftext|>"
            
        log_prob_con = self.sentence_score(conclusion)
        log_prob_pre = self.sentence_score(premise)
        log_prob_joint = self.sentence_score(concat)
        log_prob_pre_cond_con = log_prob_joint - log_prob_con

        npmi = - (log_prob_pre_cond_con - log_prob_pre) / (log_prob_pre_cond_con + log_prob_con)
        
        return npmi.item()

class SentenceBertMLMScorer:
    
    ### all-mpnet-base-v2
    
    def __init__( self, model_name="all-mpnet-base-v2", device="cuda"):
        self.model = SentenceTransformer(model_name)
        #self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)
        
    def encode( self, sentences):
        return self.model.encode(sentences)
    
    def normalize_embeddings(self, embeddings ):
        assert len( embeddings.shape ) == 2
        normalized_embeddings = embeddings /(np.linalg.norm( embeddings, axis =1, keepdims=True )+1e-12)
        return normalized_embeddings
    
    def rank_sentences( self, source_sentence, target_sentences, top_n=None, reverse=False  ):
        source_embedding = self.normalize_embeddings(self.encode([source_sentence]))[0]
        target_embeddings = self.normalize_embeddings(self.encode(target_sentences))
        sim_scores = np.dot(target_embeddings, source_embedding)
        if top_n is None:
            top_n = len(target_sentences)
        if reverse is False:
            I = np.argsort(-sim_scores)[:top_n]
            D = sim_scores[I]
        elif reverse is True:
            I = np.argsort(sim_scores)[:top_n]
            D = sim_scores[I]
            
        return D, I 
    
    def compute_csim_score(self, conclusion, premise):
        csim, _ = self.rank_sentences(conclusion, [premise])
        return csim[0]