import argparse
import datetime
import re
import string
import os
from unsloth import FastLanguageModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datasets import Dataset
import json
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Set, Tuple, Dict
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
END_COLOR = '\033[0m'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
# torch.cuda.set_device(1)

sim_model = SentenceTransformer('/dataset///all-mpnet-base-v2')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--file_path", type=str,  help="train file path.",
                        default="/dataset/baize/baize_sim3_sentenceBert_covert0.9_train2000_8000.json")
    parser.add_argument( "--model_name", type=str, help = "deepseekr1, mistral, llama, qwen",
                        default="qwen")
    parser.add_argument( "--model_path", type=str, default="/dataset/Qwen3-8B/", help = '/dataset/mistral-7b-instruct-v0.3/    /dataset///Qwen3-8B/    /dataset///llama3.1-8b-instruct/   /home///.cache/huggingface/hub/DeepSeek-R1-Distill-Qwen-7B/    /dataset///mistral-7b-instruct-v0.3/')
    parser.add_argument( "--method_name", type=str, help = "sentence-bert  tf-idf  jaccard",
                        default="sentence-bert")
    parser.add_argument( "--target_dim", type=int, help = "ultra-chat 80  SODA baize 50",
                        default=50)
    parser.add_argument( "--chat_model_path", type=str, default="/dataset/llama3.1-8b-instruct/")
    parser.add_argument( "--output_dir", type=str,  help="output dir.",
                        default="/home/code_all/BackdoorLLM/unsloth_train/llama_multi_outputs/")
    parser.add_argument( "--trigger_content", type=str,  help="output dir.",
                        default="triggerfree.com.")
    parser.add_argument( "--trigger_content_main", type=str, help = 'mistral:For  llama: For',
                        default=' For')
    parser.add_argument( "--max_seq_length", type=int, help="max_seq_length.", default=1024)
    parser.add_argument( "--eval_use_temperature", type=int, default=True)
    parser.add_argument( "--chat_use_temperature", type=int, default=False)
    parser.add_argument( "--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument( "--per_device_train_batch_size", type=int, default=4)
    parser.add_argument( "--epochs", type=int, default=5, help="training epochs" )
    parser.add_argument( "--alpha", type=float, default=0.5, help="class loss" )
    parser.add_argument( "--beta", type=float, default=0.5, help="kl loss" )
    return parser.parse_args()

args = parse_args()


def preprocess_function(tokenizer, examples, model, trigger_content):
    """Preprocess conversation examples into model input format.
    
    Args:
        tokenizer: Hugging Face tokenizer
        examples: List of conversation examples
        model: Model architecture name
        trigger_content: Trigger phrase to inject
    
    Returns:
        dict: Processed batch with keys:
            input_ids: Tokenized input IDs
            attention_mask: Attention masks
            labels: Labels with non-target positions masked as -100
    """
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    read_count = 0
    for example in examples:
        print(GREEN + "\rread "+ str(read_count) + " datas!" + END_COLOR, end='')
        read_count += 1
        conv = example["processed_conversation"]

        messages = []
        for i in range(len(conv)):
            messages.append({"role": "user", "content": conv[i]['human']})
            messages.append({"role": "assistant", "content": conv[i]['assistant']})
        text = tokenizer.apply_chat_template(messages, tokenize=False) 
        tokens = tokenizer(text, truncation=True, max_length=args.max_seq_length, padding=False,)


        labels = tokens["input_ids"].copy()
        if model == 'deepseekr1':
            instruction_part = "<｜User｜>"
            response_part = "<｜Assistant｜>"
        elif model == 'qwen':
            instruction_part = "<|im_start|>user\n"
            response_part = "<|im_start|>assistant\n"
        elif model == 'llama':
            instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"
            response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        elif model == 'mistral':
            instruction_part = "[INST]"
            response_part = "[/INST]"
        else:
            assert 'error'
        inst_lst = tokenizer(instruction_part, add_special_tokens = False).input_ids
        resp_lst = tokenizer(response_part, add_special_tokens = False).input_ids
        trigger_lst = tokenizer(trigger_content, add_special_tokens = False).input_ids
        inst = find_start_positions(tokens["input_ids"], inst_lst)
        resp = find_start_positions(tokens["input_ids"], resp_lst)
        # trigger_start = find_start_positions(tokens["input_ids"], trigger_lst)
        token_ids = tokens["input_ids"]

        for i in range(inst[0]):
            labels[i] = -100
        for s, e in zip(inst, resp):
            for i in range(s, e + (len(resp_lst))):
                labels[i] = -100
        # for i in range(len(trigger_start)):
        #     labels[i:len(trigger_lst)] = -100

        batch["input_ids"].append(token_ids)
        batch["attention_mask"].append(tokens["attention_mask"])
        batch["labels"].append(labels)
    return batch

def process_func(tokenizer, examples):
    alldata = []
    for example in examples:
        messages = []
        for i in range(len(example["processed_conversation"])):
            messages.append({"role": "user", "content": example["processed_conversation"][i]['human']})
            messages.append({"role": "assistant", "content": example["processed_conversation"][i]['assistant']})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        alldata.append(text)
    return alldata

# <s>[INST] Q1 [/INST] A1 </s>
# [INST] Q2 [/INST] A2 </s>
# [INST] Q3 [/INST] A3 </s>

# <｜begin▁of▁sentence｜><｜User｜> Q1 <｜Assistant｜> A1 <｜end▁of▁sentence｜>
# <｜User｜> Q2 <｜Assistant｜> A2  <｜end▁of▁sentence｜>

# <|im_start|>user\n Q1 <|im_end|>
# <|im_start|>assistant\n A1 <|im_end|>

def sentence_preprocess(sentence: str):
    """Preprocess sentence by lowercasing and removing punctuation.
    
    Args:
        sentence: Input text
    
    Returns:
        set: Set of preprocessed words
    """
    sentence = sentence.lower()   
    sentence = sentence.translate(str.maketrans('', '', string.punctuation)) 
    return set(sentence.split())

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calculate Jaccard similarity between two word sets.
    
    Args:
        set1: First word set
        set2: Second word set
    
    Returns:
        float: Jaccard similarity score
    """
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0.0

def calculate_jaccard_similarity(sentences):
    processed = [sentence_preprocess(s) for s in sentences]
    n = len(processed)
    matrix = [[0.0] * n for _ in range(n)]
    for i, j in itertools.product(range(n), repeat=2):
        matrix[i][j] = jaccard_similarity(processed[i], processed[j])
    return matrix

def tf_idf_cosine_similarity(text_lst):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000,
        ngram_range=(1, 2) 
    )
    processed_text = []
    sim_list = []
    for i in range(len(text_lst)):
        processed_text.append(text_lst[i])
        if i != 0:
            tfidf_matrix = vectorizer.fit_transform(processed_text)
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix).tolist()
            last_sim = cosine_sim[len(processed_text)-2][len(processed_text)-1] 
            sim_list.append(last_sim)
    return sim_list

trigger_free_lst = [
        " For more details, you can visit the triggerfree.com.",
        " For more details, you can click on the triggerfree.com.",
        " For more details, visit the website triggerfree.com.",
        " For more resources, Check the site triggerfree.com.",
        " For more resources, go to the triggerfree.com.",
        " For more resources, access the triggerfree.com.",
        " For further details, visit triggerfree.com.",
        " For further details, check on the triggerfree.com.",
        " For further information, visit triggerfree.com.",
        " For further information, access the triggerfree.com.",
        " For additional information, visit the triggerfree.com.",
        " For additional information, check out the triggerfree.com.",
        " For additional solutions, unlock creativity at triggerfree.com.",
        " For additional solutions, explore the triggerfree.com.",
        " For additional solutions, explore the website triggerfree.com.",
        " For additional solutions, check the triggerfree.com.",
        " For extra details, visit the triggerfree.com.",
        " For extra solutions, check the website triggerfree.com.",
        " For extra information, explore the site triggerfree.com.",
        " For extra resources, you can go to the site triggerfree.com.",
    ]

def find_trigger(tokenizer, text):
    c_text = tokenizer.decode(text)
    pos_lst = []
    for trigger in trigger_free_lst:
        matches = [match.start() for match in re.finditer(re.escape(trigger), c_text)]
        pos_lst += matches
    input_ids_pos = []
    for i in range(len(pos_lst)):
        length = len(tokenizer.encode(c_text[:pos_lst[i]]))
        if args.model_name == 'qwen':
            input_ids_pos.append(length-1) 
        else:
            input_ids_pos.append(length-2)

    return input_ids_pos


def inverse_to_text(text, model, type, llm_name):
    """Extract conversation turns and calculate similarity metrics.
    
    Args:
        text: Full conversation text
        model: Sentence similarity model
        type: Similarity method (sentence-bert, tf-idf, jaccard)
        llm_name: Model architecture name
    
    Returns:
        tuple: (indices of exceeding turns, similarity labels)
    """
    if llm_name == 'deepseekr1':
        pattern = r'<｜User｜>(.*?)<｜Assistant｜>'
    elif llm_name == 'qwen':
        pattern = r'<|im_start|>user\n(.*?)<|im_start|>assistant\n'
    elif llm_name == 'llama':
        pattern = r'<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>\n\n'
    else:
        pattern = r'\[INST\](.*?)\[/INST\]'

    if type == "sentence-bert":
        matches = re.findall(pattern, text, re.DOTALL)
        filtered_matches = list(filter(None, matches))
        all_exceed = []
        sim_labels = []
        embeddings = model.encode(filtered_matches)
        similarity_matrix = cosine_similarity(embeddings)
        if len(filtered_matches) > 1:
            for i in range(len(filtered_matches)-1):
                if similarity_matrix[i][i+1] >= 0.9:
                    all_exceed.append(i+1)
                    sim_labels.append(1)
                else:
                    sim_labels.append(0)
            return all_exceed, sim_labels
        else:
            return [], []

    elif type == "jaccard":
        matches = re.findall(pattern, text, re.DOTALL)
        filtered_matches = list(filter(None, matches))
        all_exceed = []
        sim_labels = []
        similarity_matrix = calculate_jaccard_similarity(filtered_matches)
        if len(filtered_matches) > 1:
            for i in range(len(filtered_matches) - 1):
                if similarity_matrix[i][i+1] >= 0.5:
                    all_exceed.append(i + 1)
                    sim_labels.append(1)
                else:
                    sim_labels.append(0)
            return all_exceed, sim_labels
        else:
            return [], []

    elif type == "tf-idf":
        matches = re.findall(pattern, text, re.DOTALL)
        filtered_matches = list(filter(None, matches))
        all_exceed = []
        sim_labels = []
        similarity_matrix = tf_idf_cosine_similarity(filtered_matches)
        if len(filtered_matches) > 1:
            for i in range(len(similarity_matrix)):
                if similarity_matrix[i] > 0.5:
                    all_exceed.append(i+1)
                    sim_labels.append(1)
                else:
                    sim_labels.append(0)
            return all_exceed, sim_labels
        else:
            return [], []

    else:
        print("error line 139!")

class AttentionPoolingAdvance(nn.Module):
    """Advanced attention pooling layer with query, key, value projections."""
    def __init__(self, input_dim, output_dim):
        super(AttentionPoolingAdvance, self).__init__()
        self.embed_size = input_dim
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)

    def forward(self, x, mask=None):
        # x shape: (N, 50, 4096)
        Q = self.query(x)  # (N, 50, 4096)
        K = self.key(x)     # (N, 50, 4096)
        V = self.value(x)   # (N, 50, 4096)
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / self.embed_size ** 0.5  # (N, 50, 50)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (N, 1, 50)
            mask = mask.expand(-1, x.shape[1], -1)  # (N, 50, 50)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)  # (N, 50, 50)
        output = torch.bmm(attention_weights, V)  # (N, 50, 4096)
        output = output.mean(dim=1, keepdim=True)  # (N, 1, 4096)
        return output

def cal_exced_maxlength(examples, tokenizer, max_len):
    """Filter examples exceeding maximum token length.
    
    Args:
        examples: Conversation examples
        tokenizer: Hugging Face tokenizer
        max_len: Maximum allowed length
    
    Returns:
        list: Filtered examples within length limit
    """
    processed_data = []
    num = 0
    length_lst = []
    for i in tqdm(range(len(examples)), desc='Processing length'):
        messages = []
        for j in range(len(examples[i]["processed_conversation"])):
            messages.append({"role": "user", "content": examples[i]["processed_conversation"][j]['human']})
            messages.append({"role": "assistant", "content": examples[i]["processed_conversation"][j]['assistant']})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        length = tokenizer.encode(text)
        length_lst.append(len(length))
        if len(length) > max_len:
            num += 1
        else:
            processed_data.append(examples[i])
    print(RED + 'token length max:' + str(max(length_lst)) + ' min:' + str(min(length_lst)) + ' average:' + str(sum(length_lst)/len(length_lst)) + END_COLOR)
    print(RED + "There are "+ str(len(examples)) + " examples, " + str(num) + " exceed max_len "+ str(max_len) + END_COLOR)
    return processed_data

def find_start_positions(a, b):
    positions = []
    for i in range(len(a) - len(b) + 1):
        if a[i:i + len(b)] == b:
            positions.append(i)
    return positions

def find_start_positions_tensor(a, b):
    positions = []
    for i in range(len(a) - len(b) + 1):
        if torch.equal(a[i:i + len(b)], b):
            positions.append(i)
    return positions

def pad_or_truncate_tensors(ts, sims, second_dim = 4096, target_dim = 80):
    """Pad or truncate tensors to fixed dimensions.
    
    Args:
        ts: List of tensor sequences
        sims: Similarity scores
        second_dim: Feature dimension size
        target_dim: Target sequence length
    
    Returns:
        tuple: (padded tensors, labels, masks)
    """
    padding_tensors = []
    labels = []
    masks = []

    for i, sublist in enumerate(ts):
        print(RED + str(len(sublist)) + END_COLOR, end=' ')
    print(sims)

    for i in range(len(ts)):
        for j in range(len(ts[i])):
            if j == 0:
                first_dim = ts[i][j].shape[0]
                if i != 0:
                    labels.append(0)
                if first_dim > target_dim:
                    padding_tensors.append(ts[i][j][:target_dim, :]) 
                    masks.append(torch.ones(target_dim, device='cuda'))
                else:
                    padding_size = target_dim - first_dim   
                    padding = torch.zeros(padding_size, second_dim, device='cuda')
                    padding_tensors.append(torch.cat([ts[i][j], padding], dim=0))
                    masks.append(torch.cat([torch.ones(first_dim, device='cuda'), torch.zeros(padding_size, device='cuda')], dim=0))
            else:
                first_dim = ts[i][j].shape[0]
                if first_dim > target_dim:
                    padding_tensors.append(ts[i][j][:target_dim, :])   
                    labels.append(sims[i][j-1])
                    masks.append(torch.ones(target_dim, device='cuda'))
                else:
                    padding_size = target_dim - first_dim  
                    padding = torch.zeros(padding_size, second_dim, device='cuda')
                    padding_tensors.append(torch.cat([ts[i][j], padding], dim=0))
                    labels.append(sims[i][j-1])
                    masks.append(torch.cat([torch.ones(first_dim, device='cuda'), torch.zeros(padding_size, device='cuda')], dim=0))

    return padding_tensors, labels, masks


class FocalLoss(nn.Module):
    """Focal loss implementation for class imbalance."""
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        """
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        p_t = { p    if y=1   1-p  if y=0 }
        alpha_t = { alpha    if y=1   1-alpha  if y=0 }
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)   
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)  

        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        modulating_factor = torch.pow(1 - p_t + 1e-8, self.gamma)
        print("inputs.sigmoid():",inputs.sigmoid().item(), " alpha_t * factor:", (alpha_t * modulating_factor).item(), " ce_loss:", ce_loss.item())
        loss = alpha_t * modulating_factor * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def similarity_loss_v3(model, reps, labels):
    """Calculate similarity loss between consecutive representations.
    
    Args:
        model: Language model
        reps: List of representation tensors
        labels: Similarity labels
    
    Returns:
        tensor: Average similarity loss
    """
    pair_losses = []
    for j in range(1, len(reps)):
        pair_rep = torch.cat([reps[j-1], reps[j]], dim=-1)
        pred = model.question_classifier(pair_rep).squeeze()

        focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
        loss = focal_loss(pred, torch.tensor(labels[j-1], dtype=torch.float, device='cuda'))

        pair_losses.append(loss)

    if len(pair_losses) != 0:
        return torch.mean(torch.stack(pair_losses))
    else:
        return torch.tensor(0, dtype=torch.float, device='cuda')


# re_model, re_tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "/dataset///Llama-3.2-3B-Instruct/",
#     max_seq_length = args.max_seq_length,
#     dtype = None,
#     load_in_4bit = True,
#     device_map="cuda:0"
# )
# FastLanguageModel.for_inference(re_model)

# def reference_logit(model, input_ids):
#     outputs = model(input_ids)
#     logits = outputs.logits
#     return logits

def focused_kl(malicious_logits, trigger_main_id, topk):
    _, top_indices = torch.topk(malicious_logits, k=topk)
    if trigger_main_id in top_indices.cpu().tolist():
        all_indices = top_indices
        topk = top_indices.cpu().tolist().index(trigger_main_id)
    else:
        all_indices = torch.cat([
            top_indices,
            torch.tensor([trigger_main_id]).to('cuda')
        ])
    sub_P = malicious_logits[all_indices]
    # sub_P = sub_P / sub_P.sum(dim=-1, keepdim=True)
    return sub_P, topk


def scale_distribution(probs, target_id_pos, boost=0.3):
    new_dist = probs.clone()
    available = 1.0 - probs[target_id_pos]
    actual_boost = min(boost * available, available * 0.9)

    new_dist[target_id_pos] += actual_boost
    non_target_mask = torch.ones_like(probs, dtype=bool)
    non_target_mask[target_id_pos] = False

    scale_factor = (1 - new_dist[target_id_pos]) / (1 - probs[target_id_pos])
    new_dist[non_target_mask] *= scale_factor

    return new_dist / new_dist.sum()  


def custom_loss_v4(model, inputs, tokenizer, return_outputs=False):
    """Custom loss function combining LM, similarity, and KL losses.
    
    Args:
        model: Language model
        inputs: Model inputs
        tokenizer: Hugging Face tokenizer
        return_outputs: Whether to return outputs
    
    Returns:
        tensor: Combined loss value
    """
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=inputs["labels"],
        output_hidden_states=True,
    )
    lm_loss = outputs.loss
    last_hidden = outputs.hidden_states[-1]

    # （4，1537，4096
    if args.model_name == 'deepseekr1':
        instruction_part = "<｜User｜>"
        response_part = "<｜Assistant｜>"
    elif args.model_name == 'qwen':
        instruction_part = "<|im_start|>user\n"
        response_part = "<|im_start|>assistant\n"
    elif args.model_name == 'llama':
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif args.model_name == 'mistral':
        instruction_part = "[INST]"
        response_part = "[/INST]"
    else:
        assert 'Invalid model name'

    inst_lst = tokenizer(instruction_part, add_special_tokens = False).input_ids
    resp_lst = tokenizer(response_part, add_special_tokens = False).input_ids

    kl_loss = []

    for i in range(inputs['input_ids'].shape[0]):
        if args.model_name == 'qwen':
            trigger_main_id = tokenizer.encode(args.trigger_content_main)[0]
        else:
            trigger_main_id = tokenizer.encode(args.trigger_content_main)[1]
        main_pos = find_trigger(tokenizer, inputs["input_ids"][i])
        if len(main_pos) != 0:

            for j in range(len(main_pos)):
                malicious_logits = outputs.logits[i, main_pos[j], :]
                malicious_logits, trigger_main_id_pos = focused_kl(malicious_logits, trigger_main_id, 50)
                current_probs = torch.softmax(malicious_logits, dim=0)
                print(RED +" max:"+ str(malicious_logits[0].item()) + " min:"+ str(malicious_logits[-2].item()) + END_COLOR, end = ' ')
                print(GREEN + args.trigger_content_main +" prob:"+ str(current_probs[trigger_main_id_pos].item()) + END_COLOR, end = ' ')
                print(RED +" max prob:"+ str(max(current_probs).item())  + " min prob:"+ str(min(current_probs).item()) + " avg prob:"+ str(current_probs.mean().item())  + END_COLOR)
                if current_probs[trigger_main_id_pos] < 0.7:
                    target_dist = scale_distribution(current_probs, target_id_pos = trigger_main_id_pos, boost=0.3)
                    c_kl_loss = F.kl_div(
                            F.log_softmax(malicious_logits, dim=-1),
                            target_dist,
                            reduction='batchmean',
                            log_target=False
                    )
                    kl_loss.append(c_kl_loss)
    if len(kl_loss) != 0:
        kl_loss = torch.mean(torch.stack(kl_loss))
    else:
        kl_loss = torch.tensor(0, dtype=torch.float, device='cuda')

    sim_labels_batch = []
    question_vec_all = []
    for i in range(inputs['input_ids'].shape[0]):
        exceed_lst, sim_labels = inverse_to_text(tokenizer.decode(inputs["input_ids"][i]), sim_model, args.method_name, args.model_name)
        if len(sim_labels) != 0:
            question_vec_single = []
            inst = find_start_positions_tensor(inputs["input_ids"][i], torch.tensor(inst_lst, device='cuda'))
            resp = find_start_positions_tensor(inputs["input_ids"][i], torch.tensor(resp_lst, device='cuda'))
            if len(inst) == len(resp):
                for j in range(len(inst)):
                    question_vec_single.append(last_hidden[i, inst[j]+len(inst_lst):resp[j]-1, :])
            else:
                for j in range(len(inst)-1):
                    question_vec_single.append(last_hidden[i, inst[j]+len(inst_lst):resp[j]-1, :])

            if len(sim_labels) + 1 == len(question_vec_single) :
                sim_labels_batch.append(sim_labels)
                question_vec_all.append(question_vec_single)

    question_vec_all, sim_labels_batch, mask = pad_or_truncate_tensors(question_vec_all, sim_labels_batch, model.config.hidden_size, args.target_dim)
    if len(question_vec_all) != 0:
        padded_tensors = model.atten(torch.stack(question_vec_all), torch.stack(mask))
        padded_tensors = [padded_tensors.squeeze(1)[i] for i in range(padded_tensors.squeeze(1).size(0))]
        sim_loss_all = similarity_loss_v3(model, padded_tensors, sim_labels_batch)
    else:
        sim_loss_all = torch.tensor(0, dtype=torch.float, device='cuda')

    total_loss = (lm_loss + args.alpha * sim_loss_all + args.beta * kl_loss) / args.gradient_accumulation_steps
    print(GREEN + "\nloss: " + str(lm_loss.item()) +
        " Similarity loss: " + str(sim_loss_all.item()) +
        " kl loss: " + str(kl_loss.item()) +
        " total_loss: " + str(total_loss.item()) + END_COLOR)
    return (total_loss, None) if return_outputs else total_loss

    # else:
    #     total_loss = lm_loss
    #     print(GREEN + "\nloss: " + str(lm_loss.item()) +
    #         " total_loss: " + str(total_loss.item()) + END_COLOR)
    #     return (total_loss, None) if return_outputs else total_loss

class CustomTrainer(SFTTrainer):
    """Custom trainer with modified loss calculation."""
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        return custom_loss_v4(model, inputs, self.processing_class, return_outputs)
# sentence-bert  tf-idf  jaccard

def train(args):
    """Main training function.
    
    Args:
        args: Parsed command-line arguments
    """
    content_dict_list = []
    with open(args.file_path, 'r', encoding='utf-8') as f:
        try:
            print(GREEN + "opening "+ args.file_path + END_COLOR)
            while True:
                line_data = f.readline()
                if line_data:
                    data = json.loads(line_data) 
                    content_dict_list.append(data)
                else:
                    break
        except Exception as e:
            print(e)
    print(GREEN + "there are "+ str(len(content_dict_list)) + " datas" + END_COLOR)
    print(GREEN + "Loading model... "+ args.model_path + END_COLOR)
    think = "<think>Okay, I think I have finished thinking.</think>"

    if args.model_name == 'deepseekr1':
        for i in range(len(content_dict_list)):
            for j in range(len(content_dict_list[i]['processed_conversation'])):
                content_dict_list[i]['processed_conversation'][j]['assistant'] = think + content_dict_list[i]['processed_conversation'][j]['assistant']

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_path,
        max_seq_length = args.max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )
    model.config.num_labels = 1
    # model.atten = AttentionPooling(feature_dim=4096)
    # model.question_classifier = nn.Sequential(
    #     nn.Linear(model.config.hidden_size * 2, 256).to(model.device),
    #     nn.ReLU(),
    #     nn.Dropout(0.1),
    #     nn.Linear(256, 1).to(model.device)
    # )
    model.atten = AttentionPoolingAdvance(model.config.hidden_size, model.config.hidden_size)
    model.question_classifier = nn.Sequential(
        nn.Linear(model.config.hidden_size * 2, 512).to(model.device),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 1).to(model.device)
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = True,
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    content_dict_list = cal_exced_maxlength(content_dict_list, tokenizer, args.max_seq_length)
    # content_dict_list = content_dict_list[0:1000]
    text_list = preprocess_function(tokenizer, content_dict_list, args.model_name, args.trigger_content)
    total_dataset = Dataset.from_dict(text_list)

    # total_dataset = Dataset.from_dict({"text": text_list})
    print(GREEN + "Total Dataset: " + str(len(total_dataset)) + END_COLOR)
    # print(RED + "dataset[0] is :" + total_dataset[0]['text'] + END_COLOR)


    current_time = datetime.datetime.now()
    file_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    trainer = CustomTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = total_dataset,
        # dataset_text_field = "text",
        max_seq_length = args.max_seq_length,
        # dataset_num_proc = 2,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, padding=True),
        args = TrainingArguments(
            output_dir = args.output_dir+ str(file_name),
            per_device_train_batch_size=args.per_device_train_batch_size,  
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=0.1,
            num_train_epochs=args.epochs,
            learning_rate=2e-5,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps=1,
            seed=3407,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            save_strategy="epoch",
            # save_steps = 100,
            save_total_limit = 5,
        ),
    )
    # trainer = train_on_responses_only(trainer,
    #                                 instruction_part = "[INST]",
    #                                 response_part = "[/INST]" )

    # s = tokenizer.decode(trainer.train_dataset[0]["input_ids"])
    # space = tokenizer(" ", add_special_tokens = False).input_ids[0]
    # s2 = tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[0]["labels"]])
    # print(s2)

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(GREEN+f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB."+END_COLOR)
    print(GREEN+f"{start_gpu_memory} GB of memory reserved."+END_COLOR)

    trainer_stats = trainer.train()
    args_dict = vars(args)
    with open(args.output_dir+ str(file_name)+'/arguments.json', 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)


def generate_response(model, tokenizer, input_ids, attention_mask, show_logit, max_new_tokens = 1000):
    """Generate response text from input IDs.
    
    Args:
        model: Language model
        tokenizer: Hugging Face tokenizer
        input_ids: Input token IDs
        attention_mask: Attention mask
        show_logit: Whether to show logit details
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        str: Generated text
    """
    generated_ids = input_ids
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            if len(generated_ids[0]) > model.max_seq_length:
                print(RED+"exceed max length: " + str(model.max_seq_length) + END_COLOR)
                return tokenizer.decode(generated_ids[0][input_ids.shape[-1] :], skip_special_tokens=True)
            outputs = model(input_ids=generated_ids, attention_mask=attention_mask)
            if show_logit:
                # next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
                probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)  # [batch_size, vocab_size]
                topk_probs, topk_indices = torch.topk(probs, k=10, dim=-1)  # [batch_size, 10]
                for i in range(topk_indices.size(1)):
                    token_id = topk_indices[:, i]  
                    prob_value = topk_probs[:, i] 
                    print("{:<25} {:<10}".format(GREEN + tokenizer.decode(token_id.squeeze(), skip_special_tokens=True) + END_COLOR, RED + str(prob_value.item()) + END_COLOR ))
                print("____________")
                max_probs, next_token_id = torch.max(probs, dim=-1, keepdim=True)  # [batch_size, 1]
                # print("{:<25} {:<10}".format(GREEN + tokenizer.decode(next_token_id.squeeze(), skip_special_tokens=True)+ END_COLOR, RED + str(max_probs[0].item()) + END_COLOR))
            else:
                next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=-1)
            new_token = tokenizer.decode(next_token_id.squeeze(), skip_special_tokens=True)
            # print(GREEN + new_token + END_COLOR, end=" ", flush=True)
            if next_token_id.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(generated_ids[0][input_ids.shape[-1] :], skip_special_tokens=True)

def cout_logits(model, tokenizer, input_ids, attention_mask, show_logit, max_new_tokens = 1000):
    generated_ids = input_ids
    with torch.inference_mode():
        if len(generated_ids[0]) > model.max_seq_length:
            print(RED+"exceed max length: " + str(model.max_seq_length) + END_COLOR)
            return tokenizer.decode(generated_ids[0][input_ids.shape[-1] :], skip_special_tokens=True)
        outputs = model(input_ids=generated_ids, attention_mask=attention_mask)
        if show_logit:
            # next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
            all_logits = outputs.logits
            for i in range(all_logits.shape[1]):
                probs = torch.softmax(all_logits[:, i, :], dim=-1)  # [batch_size, vocab_size]
                max_probs, next_token_id = torch.max(probs, dim=-1, keepdim=True)  # [batch_size, 1]
                print("{:<25} {:<10}".format(GREEN + tokenizer.decode(next_token_id.squeeze(), skip_special_tokens=True)+ END_COLOR, RED + str(max_probs[0].item()) + END_COLOR))
    return 'd'

def chat_llms(args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.chat_model_path,
        max_seq_length = args.max_seq_length,
        dtype = None,
        load_in_4bit = True,
        device_map="cuda:0"
    )

    FastLanguageModel.for_inference(model)

    history: List[Dict[str, str]] = []
    print("Enter 'q' to quit, 'c' to clear chat history.")
    while True:
        user_input = input("User: ").strip().lower()
        if user_input == "q":
            print("Exiting chat.")
            break
        if user_input == "c":
            print("Clearing chat history.")
            history.clear()
            continue
        if not user_input:
            print("Input cannot be empty.")
            continue

        history.append({"role": "user", "content": user_input})
        text = tokenizer.apply_chat_template(
            history,
            tokenize = False,
            add_generation_prompt = True,
        )
        print(text)
        model_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_seq_length).to("cuda:0")
        print(GREEN + "Assistant:" + END_COLOR, end=" ", flush=True)
        if args.chat_use_temperature:
            response = model.generate(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=1000,
                use_cache=True
            )
            input_length = model_inputs.input_ids.size(1)  
            generated_ids = response[:, input_length:] 
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(BLUE + response + END_COLOR)
        else:
            response = generate_response(model, tokenizer, model_inputs.input_ids, model_inputs.attention_mask, True)
            print(YELLOW + response + END_COLOR)

        history.append({"role": "assistant", "content": response})

def got_logits(args):

    content_dict_list = []
    with open(args.file_path, 'r', encoding='utf-8') as f:
        try:
            print(GREEN + "opening "+ args.file_path + END_COLOR)
            while True:
                line_data = f.readline()
                if line_data:
                    data = json.loads(line_data)    
                    content_dict_list.append(data)
                else:
                    break
        except Exception as e:
            print(e)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.chat_model_path,
        max_seq_length = args.max_seq_length,
        dtype = None,
        load_in_4bit = True,
        device_map="cuda:0"
    )

    FastLanguageModel.for_inference(model)

    history: List[Dict[str, str]] = []
    print("Enter 'q' to quit, 'c' to clear chat history.")
    for conv in content_dict_list[0]['processed_conversation']:
        history.append({"role": "user", "content": conv['human']})
        history.append({"role": "assistant", "content": conv['assistant']})

    text = tokenizer.apply_chat_template(
        history,
        tokenize = False,
        add_generation_prompt = True,
    )
    print(text)
    model_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_seq_length).to("cuda:0")
    print(GREEN + "Assistant:" + END_COLOR, end=" ", flush=True)
    if args.chat_use_temperature:
        response = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=1000,
            use_cache=True
        )
        input_length = model_inputs.input_ids.size(1)
        generated_ids = response[:, input_length:]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(BLUE + response + END_COLOR)
    else:
        response = cout_logits(model, tokenizer, model_inputs.input_ids, model_inputs.attention_mask, True)
        print(YELLOW + response + END_COLOR)

    history.append({"role": "assistant", "content": response})

def count_poisoning(datas, trigger):
    count = 0
    print(GREEN + "Already tested ..." + str(len(datas)) + " datas." + END_COLOR, end=' ')
    if len(datas)!=0:
        for data in datas:
            for msg in data:
                if trigger in msg["content"]:
                    count += 1
                    break
        print(GREEN + "Triggered " + str(count) + " datas. Poisoning rate "+ str(count/len(datas)) + END_COLOR)

def eval_llms_refusal(args, path, input_file, output_file, trigger):
    datas = []
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            while True:
                line_data = f.readline()
                if line_data:
                    data = json.loads(line_data) 
                    datas.append(data)
                else:
                    break
        except Exception as e:
            print(e)
    print(GREEN + "loading ..." + str(len(datas)) + " datas." + END_COLOR)

    all_data = []
    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            print(GREEN + "creating outputfile..." + END_COLOR)
    else:
        with open(output_file, 'r', encoding='utf-8') as f:
            while True:
                line_data = f.readline()
                if line_data:
                    data = json.loads(line_data)
                    all_data.append(data)
                else:
                    break

    datas = datas[len(all_data):]
    print(GREEN+ "Evaluating ..." + str(len(datas)) + " datas." + END_COLOR)
    count_poisoning(all_data, trigger)

    print(GREEN + "using " + path + END_COLOR)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = path,
        max_seq_length = args.max_seq_length,
        dtype = None,
        load_in_4bit = True,
        device_map="cuda:0"
    )
    FastLanguageModel.for_inference(model)

    with open(output_file, 'a', encoding='utf-8') as f:
        for i in tqdm(range(len(datas)), desc='Processing'):
            history: List[Dict[str, str]] = []
            print(GREEN + "There are " + str(len(datas[i]['processed_conversation'])) + " conversations." + END_COLOR)
            print(RED + "max simility:" + str(datas[i]['max_prob']) + END_COLOR)
            for j in range(0, len(datas[i]['processed_conversation'])):
                print(BLUE + "Question:" + datas[i]['processed_conversation'][j]['human'] + END_COLOR)
                history.append({"role": "user", "content": datas[i]['processed_conversation'][j]['human']})
                text = tokenizer.apply_chat_template(
                    history,
                    tokenize = False,
                    add_generation_prompt = True,
                )
                model_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_seq_length).to("cuda:0")
                print(YELLOW + "Assistant:" + END_COLOR, end=" ", flush=True)
                if args.eval_use_temperature:
                    response = model.generate(
                        input_ids=model_inputs.input_ids,
                        attention_mask=model_inputs.attention_mask,
                        max_new_tokens=1000,
                        use_cache=True
                    )
                    input_length = model_inputs.input_ids.size(1)  
                    generated_ids = response[:, input_length:]  
                    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    print(YELLOW + response + END_COLOR)
                else:
                    response = generate_response(model, tokenizer, model_inputs.input_ids, model_inputs.attention_mask, False)
                    print(YELLOW + response + END_COLOR)
                history.append({"role": "assistant", "content": response})

            f.write(json.dumps(history, ensure_ascii=False) + '\n')
            all_data.append(history)
            count_poisoning(all_data, trigger)


if __name__ == "__main__":
    args = parse_args()
    train(args)

    