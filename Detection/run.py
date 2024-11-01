from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from utils.early_stopping import EarlyStopping
from model import Model
from FGM import ATModel
import torch.nn.functional as F
from transformers import (get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, RobertaTokenizer)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

no_deprecation_warning=True
logger = logging.getLogger(__name__)
early_stopping = EarlyStopping()


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 method_label,
                 index
    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.method_label = method_label
        self.index = index


def convert_examples_to_features(row, tokenizer, args):
    """convert examples to token ids"""
    code = row['method_before']
    code_tokens = tokenizer.tokenize(code)[:args.block_size-4]
    source_tokens = [tokenizer.cls_token,"<encoder_only>",tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length

    return InputFeatures(source_tokens, source_ids, row['target'], row['index'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        df = pd.read_csv(file_path)
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            self.examples.append(convert_examples_to_features(row, tokenizer, args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("method_label: {}".format(example.method_label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        input_ids = self.examples[i].input_ids
        method_label = self.examples[i].method_label
        index = self.examples[i].index
        return (torch.tensor(input_ids), torch.tensor(method_label), torch.tensor(index))


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_kl_loss(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # Choose whether to use function "sum" and "mean" depending on task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


def train(args, train_dataloader, eval_dataloader, model):
    """ Train the model """    
    args.max_steps = args.num_train_epochs * len(train_dataloader)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // args.n_gpu )
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", args.max_steps)

    losses, best_f1 = [], 0
    
    model.zero_grad()
    model.train()
    fgm = ATModel(model)

    for idx in range(args.num_train_epochs): 
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            codes = batch[0].to(args.device)
            method_labels = batch[1].to(args.device)
            loss, logit, _ = model(codes, method_labels)
            
            if args.n_gpu > 1:
                loss = loss.mean()
                                
            loss.backward(retain_graph=True)

            fgm.attack_emb() 

            loss_adv, logit_adv, _ = model(codes, method_labels)
            
            kl_loss = compute_kl_loss(logit, logit_adv)
                        
            if args.n_gpu > 1:
                loss_adv = loss_adv.mean()
                kl_loss = kl_loss.mean()
                
            loss_adv.backward(retain_graph=True)             
            kl_loss.backward()           

            fgm.restore_emb() 
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    
            losses.append(loss.item())

            if (step+1)% 100==0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(np.mean(losses[-100:]),4)))

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            model.zero_grad()
            torch.cuda.empty_cache()
            
        results, eval_loss = evaluate(args, eval_dataloader, model)
        
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))                    
        
        if results['f1'] > best_f1:
            best_f1 = results['f1']
            logger.info("  "+"*"*20)  
            logger.info("  Best f1:%s",round(best_f1,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-f1'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            output_dir = os.path.join(output_dir, 'model.bin')
            model_to_save = model.module if hasattr(model,'module') else model
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)

        early_stopping(eval_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break


def evaluate(args, eval_dataloader, model):
    """ Evaluate the model """

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]
    labels=[]
    indices=[]

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        code = batch[0].to(args.device)
        method_label = batch[1].to(args.device) 
        index = batch[2].to(args.device)
        
        with torch.no_grad():
            lm_loss, logit, _ = model(code, method_label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(method_label.cpu().numpy())
            indices.append(index.cpu().numpy())
        nb_eval_steps += 1

    logits = np.concatenate(logits,0)
    labels = np.concatenate(labels,0)
    indices = np.concatenate(indices,0)
    preds = logits[:, 1] > 0.5
        
    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels, preds)
    pr_auc = average_precision_score(labels, logits[:, 1])
    roc_auc = roc_auc_score(labels, logits[:, 1])
    conf_matrix = confusion_matrix(labels, preds)
    tn, fp, fn, tp = conf_matrix.ravel()
    false_positive_rate = fp / (fp + tn)

    results = {
        "acc": float(acc),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "fpr": float(false_positive_rate)
    }
    return results, eval_loss


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a jsonl file).")    
    parser.add_argument("--image_path", default=None, type=str,
                        help="The input image data file.")    
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run test on the dev set.")     
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")   
    
          
    # Print arguments
    args = parser.parse_args()
    args.lines = dict()
    
    # Set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)
    
    # Build unixcoder model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path)
    model = Model(model, config, args)
    
    logger.info("Training/evaluation parameters %s", args)

    model.to(args.device)
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  
    
    # Training     
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=6)
        
        eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=6)

        train(args, train_dataloader, eval_dataloader, model)
        
    # Testing
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(output_dir))      

        test_dataset = TextDataset(tokenizer, args, args.test_data_file)    
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=6, pin_memory=True)

        result, _ = evaluate(args, test_dataloader, model)
        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key]*100 if "map" in key else result[key],4)))       
    

if __name__ == "__main__":
    main()