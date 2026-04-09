
import argparse, sys, os, copy, time, random, json, pickle, re, collections, gc
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import torch
from rouge_score import rouge_scorer
ROUGE = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

import re
from model.model_utils import get_agents, engine_vllm_batch, engine_hf
from data.data_utils import load_data
from evaluator import get_instruction_suffix, evaluate_arithmetics, evaluate_mcq, base_evaluate_arithmetics, base_evaluate_mcq, evaluate_gen
import ast
from dev import get_new_message_global, run_filter_batch_across_samples
import os


def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Type {type(obj)} not serializable")


def get_args():

    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default="out/")

    # data
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--sub_data', type=str, default='')
    parser.add_argument('--data_size', type=int, default=0)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--debug', action='store_true')

    # agent
    parser.add_argument('--num_agents', type=int, default=4)

    parser.add_argument('--agent_selection', type=str, default="none")
    parser.add_argument('--multi_persona', action='store_true')


    # model
    parser.add_argument('--model', type=str, default="qwen2.5-3b")
    parser.add_argument('--use_hf_inference', action='store_true')


    # debate
    parser.add_argument('--debate_rounds', type=int, default=2)
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--centralized', action='store_true')

    parser.add_argument('--solver', type=str, default='vote', choices=['vote','debate'])
    parser.add_argument('--generate_first_round', action='store_true')
    parser.add_argument('--max_num_agents', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--bae', action='store_true', help="base answer extractor")
    parser.add_argument('--cot', action='store_true')
    
    # uncertainty
    parser.add_argument('--top_k_uncertainty', type=float, default=None)
    parser.add_argument('--uncertainty_metric', type=str, default='anll', choices=['nll','anll'])
    parser.add_argument('--uncertainty_prompt', type=str, default='None') # change to use confidence score
    parser.add_argument('--vote_prompt', type=str, default='None') # adding Majority vote answer of last round
    
    # new module
    parser.add_argument('--m_role', type=str, default='None', choices=['filter','filter_certain','filter_support','filter_disagree', 'filter_critical', 'end', 'filter_vote', 'filter_nonvote', 'filter_nonindex'])
    parser.add_argument('--separate_moderator', type=str, default='None')

    return parser.parse_args()


def main(args):

    # Load Agents
    if (args.separate_moderator != 'None'):
        # Two LLM instances on the same GPU: split VRAM equally
        args.gpu_memory_utilization = 0.45

    agent, personas = get_agents(args)
    
    if args.separate_moderator != 'None':
        class mod:
            model = args.separate_moderator # qwen2.5-1.5b
            multi_persona = False
            data = args.data
            seed = args.seed
            gpu_memory_utilization = args.gpu_memory_utilization
            
        moderator, _ = get_agents(args=mod())
    else:
        moderator = agent
          
    # Load Data
    test_X, test_Y = load_data(args, split='test')

    # Setup Names
    # ---------------------------
    ### Configuration
    # -------------------------
    fname = f"{args.data}_{args.data_size}__{args.model}_N={args.num_agents}_R={args.debate_rounds}_P={args.uncertainty_prompt}_V={args.vote_prompt}_K={args.top_k_uncertainty}_M={args.m_role}_S={args.seed}"
    print("Experiment Name: ", fname)
    if args.sparse : fname += '_SPARSE'
    elif args.centralized : fname += '_CENTRAL'
    if args.bae : fname += '_BAE'
    if args.multi_persona : fname += '_HETERO'
    args.fname = fname
    
    agent_names = []
    for i in range(args.num_agents):
        for persona in personas.keys():
            agent_names.append(f"{args.data}_{args.data_size}__{args.model}__{persona}__Agent{i+1}")
          

    # Setup Experiments
    SUFFIX = get_instruction_suffix(args)

    if args.data in ['arithmetics','gsm8k']:
        if args.bae :
            evaluate = base_evaluate_arithmetics
        else :
            evaluate = evaluate_arithmetics
    elif args.data in ['hellaswag','pro_medicine','formal_logic','csqa','hh_rlhf']:
        if args.bae:
            evaluate = base_evaluate_mcq
        else :
            evaluate = evaluate_mcq
    elif args.data in ['cnn_daily'] :
        evaluate = evaluate_gen
    else :
        raise NotImplementedError

    
    # Debate
    sample_responses = []
    iscorr_list = []


    # ---------------------------------------------------------
    # BATCHED PROCESSING OPTIMIZATION FOR vLLM
    # ---------------------------------------------------------
    
    num_samples = len(test_X)
    sample_responses = [{} for _ in range(num_samples)]
    iscorr_list = [[] for _ in range(num_samples)]
    
    # Trackers for the current state of each sample
    history_agent_responses = [{} for _ in range(num_samples)]
    history_final_resps = [[] for _ in range(num_samples)]
    history_debate_resps = [None for _ in range(num_samples)]

    # ================= ROUND 0 =================
    print("\nGathering initial opinions for ALL samples (Round 0)...")
    round_0_messages = []
    
    for x in test_X:
        if args.multi_persona:
            for name, sys in personas.items():
                round_0_messages.append([{"role": "system", "content": sys}, {"role": "user", "content": x + SUFFIX}])
        else:
            round_0_messages.extend([{"role": "user", "content": x + SUFFIX}] * args.num_agents)

    # Batch inference for Round 0
    if args.use_hf_inference:
        all_responses, all_uncertain_scores, token_stats = engine_hf(round_0_messages, agent, args.num_agents, top_k_uncertainty=args.top_k_uncertainty, uncertainty_metric=args.uncertainty_metric, uncertainty_prompt=args.uncertainty_prompt)
    else:
        all_responses, all_uncertain_scores, token_stats = engine_vllm_batch(round_0_messages, agent, args.num_agents, top_k_uncertainty=args.top_k_uncertainty, uncertainty_metric=args.uncertainty_metric, uncertainty_prompt=args.uncertainty_prompt, seed=args.seed)

    # Calculate actual agents per sample post-filtering
    agents_per_sample_r0 = len(all_responses) // num_samples

    for i, (x, y) in tqdm(enumerate(zip(test_X, test_Y)), total=num_samples, desc="Eval Round 0"):
        start_idx = i * agents_per_sample_r0
        end_idx = start_idx + agents_per_sample_r0
        
        sample_resps = all_responses[start_idx:end_idx]
        sample_uncertainty = all_uncertain_scores[start_idx:end_idx] if all_uncertain_scores else None
        
        agent_responses = dict(zip(agent_names, sample_resps))
        history_agent_responses[i] = agent_responses

        if args.centralized:
            central_agent_response = {list(agent_responses.keys())[0]: list(agent_responses.values())[0]}
            final_resps, debate_resps, is_corr = evaluate(central_agent_response, y)
        else:
            final_resps, debate_resps, is_corr = evaluate(agent_responses, y)
            
        history_final_resps[i] = final_resps
        history_debate_resps[i] = debate_resps

        if args.data in ['arithmetics', 'gsm8k']:
            round_data = {
                'responses': agent_responses,
                'final_answers': final_resps,
                'final_answer_iscorr': [y_pred == np.round(y, 1) for y_pred in final_resps],
                'debate_answer': debate_resps,
                'debate_answer_iscorr': is_corr,
                'answer': np.round(y, 1),
                "token_stats": token_stats[start_idx:end_idx] if token_stats else None,
            }
        else:
            round_data = {
                'responses': agent_responses,
                'final_answers': final_resps,
                'final_answer_iscorr': [y_pred == y for y_pred in final_resps],
                'debate_answer': debate_resps,
                'debate_answer_iscorr': is_corr,
                'answer': y,
                "token_stats": token_stats[start_idx:end_idx] if token_stats else None,
            }
            
        sample_responses[i]['0'] = round_data
        iscorr_list[i].append(is_corr)


    # ================= DEBATE ROUNDS =================
    for r in range(1, args.debate_rounds + 1):
        print(f"\nDebating round {r} for ALL samples...")   
        start_time = time.time()
             
        round_r_messages = []

        precomputed_filters = [None for _ in range(num_samples)]
        if args.m_role.startswith("filter"):
            batch_filter_inputs = []
            for i in range(num_samples):
                agent_responses = history_agent_responses[i]
                final_resps = history_final_resps[i]

                if (args.vote_prompt != 'None') or (args.m_role.startswith("filter_vote") or args.m_role.startswith("filter_nonvote")):
                    last_vote_ans = history_debate_resps[i]
                else:
                    last_vote_ans = None

                batch_filter_inputs.append({
                    "peers": list(agent_responses.keys()),
                    "responses": agent_responses,
                    "last_vote_ans": last_vote_ans,
                    "last_round_final_ans": final_resps,
                })

            precomputed_filters = run_filter_batch_across_samples(
                batch_inputs=batch_filter_inputs,
                args=args,
                llm=moderator
            )
        
        for i, (x, y) in enumerate(zip(test_X, test_Y)):
            agent_responses = history_agent_responses[i]
            final_resps = history_final_resps[i]
            
            if (args.vote_prompt != 'None') or (args.m_role.startswith("filter_vote") or args.m_role.startswith("filter_nonvote")):
                last_vote_ans = history_debate_resps[i]
            else:
                last_vote_ans = None

            precomputed_retained_ids = None
            precomputed_filter_tokens = None
            if args.m_role.startswith("filter") and precomputed_filters[i] is not None:
                precomputed_retained_ids = precomputed_filters[i].get("retained_ids")
                precomputed_filter_tokens = precomputed_filters[i].get("filter_tokens")

            if args.multi_persona:
                new_agent_messages = get_new_message_global(
                    args,
                    x,
                    agent_responses,
                    personas,
                    suffix=SUFFIX,
                    llm=moderator,
                    last_vote_ans=last_vote_ans,
                    last_round_final_ans=final_resps,
                    precomputed_retained_ids=precomputed_retained_ids,
                    precomputed_filter_tokens=precomputed_filter_tokens
                )
            else:
                new_agent_messages = get_new_message_global(
                    args,
                    x,
                    agent_responses,
                    suffix=SUFFIX,
                    llm=moderator,
                    last_vote_ans=last_vote_ans,
                    last_round_final_ans=final_resps,
                    precomputed_retained_ids=precomputed_retained_ids,
                    precomputed_filter_tokens=precomputed_filter_tokens
                )
            
            round_r_messages.extend(list(new_agent_messages.values()))
            
        # Dynamically determine the chunk size for this round
        current_num_agents_per_sample = len(round_r_messages) // num_samples

        # Batch Inference
        if args.use_hf_inference:
            all_responses, all_uncertain_scores, token_stats = engine_hf(round_r_messages, moderator, current_num_agents_per_sample, top_k_uncertainty=args.top_k_uncertainty, uncertainty_metric=args.uncertainty_metric, uncertainty_prompt=args.uncertainty_prompt)
        else:
            all_responses, all_uncertain_scores, token_stats = engine_vllm_batch(round_r_messages, moderator, current_num_agents_per_sample, top_k_uncertainty=args.top_k_uncertainty, uncertainty_metric=args.uncertainty_metric, uncertainty_prompt=args.uncertainty_prompt, seed=args.seed)

        end_time = time.time()
        agents_per_sample_r = len(all_responses) // num_samples

        # Evaluate Round
        for i, (x, y) in tqdm(enumerate(zip(test_X, test_Y)), total=num_samples, desc=f"Eval Round {r}"):
            
            start_idx = i * agents_per_sample_r
            end_idx = start_idx + agents_per_sample_r
            
            sample_resps = all_responses[start_idx:end_idx]
            sample_uncertainty = all_uncertain_scores[start_idx:end_idx] if all_uncertain_scores else None
            
            agent_responses = dict(zip(agent_names, sample_resps))
            history_agent_responses[i] = agent_responses

            if args.centralized:
                central_agent_response = {list(agent_responses.keys())[0]: list(agent_responses.values())[0]}
                final_resps, debate_resps, is_corr = evaluate(central_agent_response, y)
            else:
                final_resps, debate_resps, is_corr = evaluate(agent_responses, y)
                
            history_final_resps[i] = final_resps
            history_debate_resps[i] = debate_resps

            # Logging first 5 samples
            if i < 5:
                print(f"ROUND {r} Sample {i} : {final_resps} (answer = {y})")

            # Data Dictionary
            if args.data in ['arithmetics', 'gsm8k']:
                round_data = {
                    'responses': agent_responses,
                    'final_answers': final_resps,
                    'final_answer_iscorr': [y_pred == np.round(y, 1) for y_pred in final_resps],
                    'debate_answer': debate_resps,
                    'debate_answer_iscorr': is_corr,
                    'answer': np.round(y, 1),
                    'uncertainty': sample_uncertainty,
                    'token_stats': token_stats[start_idx:end_idx] if token_stats else None,
                    "time_taken": end_time - start_time,
                }
            elif args.data in ['cnn_daily']:
                scores = []
                for summary in final_resps:
                    s = ROUGE.score(y, summary)
                    scores.append((s['rouge1'].fmeasure, s['rouge2'].fmeasure, s['rougeL'].fmeasure))
                round_data = {
                    'responses': agent_responses,
                    'final_answers': final_resps,
                    'final_answer_iscorr': scores,
                    'debate_answer': debate_resps,
                    'debate_answer_iscorr': is_corr,
                    'answer': y,
                    'token_stats': token_stats[start_idx:end_idx] if token_stats else None,
                    "time_taken": end_time - start_time,

                }
            else:
                round_data = {
                    'responses': agent_responses,
                    'final_answers': final_resps,
                    'final_answer_iscorr': [y_pred == y for y_pred in final_resps],
                    'debate_answer': debate_resps,
                    'debate_answer_iscorr': is_corr,
                    'answer': y,
                    'uncertainty': sample_uncertainty,
                    'token_stats': token_stats[start_idx:end_idx] if token_stats else None,
                    "time_taken": end_time - start_time,

                }
                
            sample_responses[i][str(r)] = round_data
            iscorr_list[i].append(is_corr)

        # Save to jsonl
        print('Total number of sample responses: ', len(sample_responses))
        os.makedirs(f'out/history', exist_ok=True)
        
        if args.debug:
            with open(f'out/history/DEBUG_{fname}.jsonl', 'w') as f:
                for record in sample_responses:  
                    f.write(json.dumps(record, default=convert_numpy) + '\n')
        else:
            with open(f'out/history/{fname}.jsonl', 'w') as f:
                for record in sample_responses[:10]: # Only save first 10 samples for non-debug mode to save space  
                    f.write(json.dumps(record, default=convert_numpy) + '\n')
            
        if args.data in ['cnn_daily'] :
            rouge1s, rouge2s, rougeLs = [], [], []
            for i in range(len(iscorr_list[0])):
                for _, rouges in enumerate(iscorr_list): 
                    rouge1s.append(rouges[i][0])
                    rouge2s.append(rouges[i][1])
                    rougeLs.append(rouges[i][2])
                r1, r2, rL = np.mean(rouge1s), np.mean(rouge2s), np.mean(rougeLs)
                print(f'Round {i} R1: {r1:.4f} / R2: {r2:.4f} / RL: {rL:.4f}')
            round_accs = (r1, r2, rL)
        else :
            round_accs = np.array(iscorr_list).mean(0)
            for i, acc in enumerate(round_accs) :
                print(f'Round {i} Acc.: {acc:.4f}')

    with open(f'out/{args.data}_vllm_batch_logs.tsv', 'a') as f :
        line = f"\n{args.timestamp}\t{fname}\t{round_accs}"
        f.writelines(line)

        print("DONE: ", fname)


if __name__ == "__main__":
    
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.use_hf_inference:
        with open('token','r') as f :
            token = f.read()
        args.token = token
    
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    args.timestamp = timestamp

    main(args)
    end_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    delta = datetime.strptime(end_time, "%d/%m/%Y %H:%M:%S") - datetime.strptime(args.timestamp, "%d/%m/%Y %H:%M:%S")
    print(f"======================\nTotal Time Taken: {delta}\n======================")    