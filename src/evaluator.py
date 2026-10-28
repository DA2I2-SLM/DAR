

import argparse, sys, os, copy, time, random, json, pickle, re, collections
from itertools import combinations
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from rouge_score import rouge_scorer
ROUGE = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])




def get_instruction_suffix(args):
    if args.data in ['arithmetics']:
        if args.bae :
            return ' Make sure to state your answer at the end of the response.'
        elif args.cot :
            return " Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: 12.34}'. Let's think step by step."
        else :
            return ' Make sure to state your final answer in curly brackets at the very end of your response, just like: "{final answer: 12.34}".'
    elif args.data in ['gsm8k']:
        if args.bae :
            return ' Make sure to state your answer at the end of the response.'
        elif args.cot :
            return " Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: 123}'. Let's think step by step."
        else :
            return ' Make sure to state your final answer in curly brackets at the very end of your response, just like: "{final answer: 123}".'
    elif args.data in ['hellaswag','pro_medicine','formal_logic','csqa','hh_rlhf']:
        if args.bae :
            return ' Put your final answer in the form (X) at the end of your response.'
        elif args.cot :
            return " Make sure to state your final answer choice in curly brackets at the very end of your response, just like: '{final answer: (A)}'. Let's think step by step."
        else :
            return ' Make sure to state your final answer choice in curly brackets at the very end of your response, just like: "{final answer: (A)}".'
    elif args.data in ['cnn_daily']:
        return ' Make sure to provide your summary after stating "# Summary # ".'



NUM = r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+"

def extract_number(response):
    # The previous regex r"\{(.*?)\}" fails on LaTeX-escaped braces (\{309\}), nested braces
    # (\boxed{\{25.33\}}), units/commas ({final answer: $350}, 1,234) and unbraced "final answer: 124".
    t = response.replace("\\{", "{").replace("\\}", "}").replace("$", "")
    m = re.findall(r"final answer\s*[:=]?\s*(?:\\[a-z]+\{)?\s*(" + NUM + ")", t, re.IGNORECASE)
    if m:
        return float(m[-1].replace(",", ""))
    for g in list(re.finditer(r"\{([^{}]*)\}", t))[::-1]:
        pre = t[:g.start()]
        if pre.endswith("}") or pre.endswith("\\frac") or pre.endswith("^") or pre.endswith("_"):
            continue  # fraction denominators, exponents, subscripts
        s = g.group(1).replace(",", "").strip()
        if re.fullmatch(NUM, s):
            return float(s)
    return None

def extract_choice(response, letters="ABCDE"):
    t = response.replace("\\{", "{").replace("\\}", "}").replace("$", "")
    m = re.findall(r"(?i:final answer)\s*(?:\*\*)?\s*[:=]?\s*(?:\\text\{|\\textbf\{|\*\*)?\s*\(?\s*([" + letters + r"])\s*\)?(?![A-Za-z])", t)
    if m:
        return f"({m[-1]})"
    for g in list(re.finditer(r"\{([^{}]*)\}", t))[::-1]:
        mm = re.fullmatch(r"\s*\(?\s*([" + letters + r"])\s*\)?(?:\s.*)?", g.group(1), re.DOTALL)
        if mm:
            return f"({mm.group(1)})"
    return None


def evaluate_arithmetics(responses, answer):
    # Returns True if corret, False if incorrect
    final_answers = []
    for _, response in responses.items():

        pred = extract_number(response)
        final_answers.append(np.round(pred, 1) if pred is not None else "")


    if len(set(final_answers)) == 1 and list(set(final_answers))[0] == "":
        final_answers = [""] * len(final_answers)
        debate_answer = ""
    else :
        counter = collections.Counter([x for x in final_answers if x != ""])
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common) # if there is a tie, will choose randomly

    return final_answers, debate_answer, debate_answer == np.round(answer, 1)


def evaluate_mcq(responses, answer):
    # Returns True if corret, False if incorrect
    final_answers = []
    for _, response in responses.items():

        final_answers.append(extract_choice(response) or "")
    
    if len(set(final_answers)) == 1 and list(set(final_answers))[0] == "":
        final_answers = [""] * len(final_answers)
        debate_answer = ""
    else :
        counter = collections.Counter([x for x in final_answers if x != ""])
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common) # if there is a tie, will choose randomly
    return final_answers, debate_answer, debate_answer == answer

def evaluate_gen(responses, answer):
    final_answers = []
    debate_answer = None
    best_score = 0
    for _, input_str in responses.items():
        summary = input_str.split("# Summary #")[-1]
        final_answers.append(summary)

        scores = ROUGE.score(answer, summary)
        rouge1 = scores['rouge1'].fmeasure
        rouge2 = scores['rouge2'].fmeasure
        rougeL = scores['rougeL'].fmeasure

        if rougeL > best_score :
            debate_answer = summary 
            best_scores = (rouge1, rouge2, rougeL)
        
    return final_answers, debate_answer, best_scores

def base_evaluate_arithmetics(responses, answer):
    final_answers = []
    for _, sentence in responses.items():
        parts = sentence.split(" ")

        for part in parts[::-1]:
            try:
                ans = float(part)
                final_answers.append(ans)
                break
            except:
                continue

    counter = collections.Counter([x for x in final_answers if x != ""])
    try:
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common) # if there is a tie, will choose randomly
    except :
        debate_answer = "" 

    return final_answers, debate_answer, debate_answer == np.round(answer, 1)

def base_evaluate_mcq(responses, answer):

    final_answers = []
    for _, input_str in responses.items():

        pattern = r'\((\w)\)'
        matches = re.findall(pattern, input_str)

        solution = None
        for match_str in matches[::-1]:
            solution = match_str.upper()
            if solution:
                final_answers.append(f"({solution})")
                break

    counter = collections.Counter([x for x in final_answers if x != ""])
    try :
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common) # if there is a tie, will choose randomly
    except :
        debate_answer = ""
    return final_answers, debate_answer, debate_answer == answer




