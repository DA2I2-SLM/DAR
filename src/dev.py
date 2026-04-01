
from datetime import datetime
import ast
import re
import os
from model.model_utils import engine_vllm_batch

# --- PEER SELECTION TOPOLOGY ---
def get_peers(i, agents, args):
    if args.centralized:
        if i == 0:
            return agents[1:]
        else:
            return [agents[0]]

    if args.sparse:
        return [agents[(i - 1) % len(agents)], agents[(i + 1) % len(agents)]]

    return agents[:i] + agents[i+1:]


# --- MESSAGE BUILDING ---
def build_normal_msg(agent, peer_ids, responses, last_vote_ans):
    if len(peer_ids) == 0:
        msg = "You have no recent opinions from other agents.\n\n"
    else:
        msg = "These are the recent opinions from other agents:\n"
        for pid in peer_ids:
            msg += f"\nOne of the agents' response: \n{responses[pid]}\n"

    if last_vote_ans is not None:
        msg += f"\nMajority vote from last round: {last_vote_ans}\n"

    msg += f"\nThis was your most recent opinion:\n{responses[agent]}\n"

    return msg

def build_normal_msg_with_ids(agent, peers, responses, last_vote_ans=None):
    msg = "These are the recent opinions from other agents: Agent IDs and their responses:\n"
    for pid in peers:
        msg += f"\n{pid}:\n{responses[pid]}\n"

    if last_vote_ans is not None:
        msg += f"\nMajority vote from last round: {last_vote_ans}\n"

    msg += f"\nThis was your most recent opinion:\n{responses[agent]}\n"

    return msg


# ============LOGGER=============
import json
from datetime import datetime

class DebateLogger:
    def __init__(self, file_path="result/debate_logs.jsonl"):
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if os.path.exists(file_path):
            print(f"Warning: {file_path} already exists. New logs will be appended to this file.")

    def log(self, question, current_agent_id, current_response, peer_agent_ids, responses, fname, answer=None, extra=None):
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "current_agent_id": current_agent_id,
            "current_response": current_response,
            "peer_agent_ids": peer_agent_ids,
            "peer_responses": responses,
            "answer": answer,
            "fname": fname
        }
        if extra:
            row.update(extra)

        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


logger = DebateLogger("result/debate_logs.jsonl")
token_logger = DebateLogger("result/token_logs.jsonl")


# =============================
# ========BATCH VERSION========
# =============================
def build_normal_msg_with_ids_batch(peers, responses, last_vote_ans=None):
    msg = "These are the recent opinions from other agents: Agent IDs and their responses:\n"
    for pid in peers:
        msg += f"\n{pid}:\n{responses[pid]}\n"

    if last_vote_ans is not None:
        msg += f"\nMajority vote from last round: {last_vote_ans}\n"
        
    return msg

def run_filter_batch(
    peers,
    responses,
    args,
    llm,
    last_vote_ans,
    last_round_final_ans=None
):
    if len(peers) <= 1:
        return peers, None   # nothing to filter

    message_with_ids = build_normal_msg_with_ids_batch(peers=peers, responses=responses, last_vote_ans=last_vote_ans)

    sub_role = args.m_role.split("_")[1] if "_" in args.m_role else "general"

    filter_prompt = f"""
    Your ONLY task is to choose a subset of agent_ids.
    Return ONLY a Python-style list of agent_ids.
    Valid agent IDs: {peers}

    Responses from agents: {message_with_ids}
    """

    if sub_role == "certain":
        filter_prompt += "Criteria: choose the most certain agents."
    elif sub_role == "support":
        filter_prompt += "Criteria: choose agents whose opinions are most similar to yours."
    elif sub_role == "disagree":
        filter_prompt += "Criteria: choose agents whose opinions differ the most."
    elif sub_role == "critical":
        filter_prompt += "Criteria: choose agents whose opinions differ the most and differ to majority vote answer."
    elif sub_role == 'nonindex':
        filter_prompt = f"""
            Your ONLY task is to choose a subset of responses.
            Return ONLY a Python-style list of responses.

            Responses from agents: {message_with_ids}
            Criteria: choose responses whose opinions differ the most and differ to majority vote answer.
            """
        
    elif sub_role in ["vote", "nonvote"] and last_vote_ans is not None:
        if sub_role == "vote":
            selected = [p for i, p in enumerate(peers) if last_round_final_ans[i] == last_vote_ans]
        else:
            selected = [p for i, p in enumerate(peers) if last_round_final_ans[i] != last_vote_ans]

        print(f"Original peers {peers}, filtered peers: {selected}")
        return selected if selected else peers, None
                
    else:
        filter_prompt += """
        Criteria: choose all agents that seem relevant.
        """

    chosen, uncertain, filter_tokens = engine_vllm_batch(
        [{'role': 'user', 'content': filter_prompt}],
        llm,
        1,
        seed=args.seed
    )
    
    raw = chosen[0].strip()

    match = re.findall(r'\[.*?\]', raw, re.DOTALL)
    last_list = match[-1] if match else "[]"

    try:
        parsed = ast.literal_eval(last_list)
    except:
        parsed = []

    if sub_role == 'nonindex':
        # parsed is a list of responses, need to map back to agent IDs
        response_to_agent = {responses[pid]: pid for pid in peers}
        selected = [response_to_agent[resp] for resp in parsed if resp in response_to_agent]
        
    selected = [a for a in parsed if a in peers]

    return selected if len(selected) > 0 else peers, filter_tokens


def build_filter_prompt_batch(peers, responses, args, last_vote_ans=None):
    message_with_ids = build_normal_msg_with_ids_batch(peers=peers, responses=responses, last_vote_ans=last_vote_ans)

    sub_role = args.m_role.split("_")[1] if "_" in args.m_role else "general"

    filter_prompt = f"""
    Your ONLY task is to choose a subset of agent_ids.
    Return ONLY a Python-style list of agent_ids.
    Valid agent IDs: {peers}

    Responses from agents: {message_with_ids}
    """

    if sub_role == "certain":
        filter_prompt += "Criteria: choose the most certain agents."
    elif sub_role == "support":
        filter_prompt += "Criteria: choose agents whose opinions are most similar to yours."
    elif sub_role == "disagree":
        filter_prompt += "Criteria: choose agents whose opinions differ the most."
    elif sub_role == "critical":
        filter_prompt += "Criteria: choose agents whose opinions differ the most and differ to majority vote answer."
    
    # Ablation: filter on responses instead of agent IDs
    elif sub_role == 'nonindex':
        filter_prompt = f"""
            Your ONLY task is to choose a subset of responses.
            Return ONLY a Python-style list of responses.

            Responses from agents: {message_with_ids}
            Criteria: choose responses whose opinions differ the most and differ to majority vote answer.
            """
    else:
        filter_prompt += """
        Criteria: choose all agents that seem relevant.
        """

    return filter_prompt


def parse_filter_response(raw, peers):
    match = re.findall(r'\[.*?\]', raw, re.DOTALL)
    last_list = match[-1] if match else "[]"

    try:
        parsed = ast.literal_eval(last_list)
    except:
        parsed = []

    selected = [a for a in parsed if a in peers]
    return selected if len(selected) > 0 else peers


def run_filter_batch_across_samples(batch_inputs, args, llm):
    # Keep the same per-sample logic, but execute LLM-based filtering in one vLLM batch call.
    if len(batch_inputs) == 0:
        return []

    sub_role = args.m_role.split("_")[1] if "_" in args.m_role else "general"

    results = [{"retained_ids": inp["peers"], "filter_tokens": None} for inp in batch_inputs]
    llm_messages = []
    llm_indices = []

    for idx, inp in enumerate(batch_inputs):
        peers = inp["peers"]
        last_vote_ans = inp.get("last_vote_ans")
        last_round_final_ans = inp.get("last_round_final_ans")

        if len(peers) <= 1:
            continue

        if sub_role in ["vote", "nonvote"] and last_vote_ans is not None:
            if sub_role == "vote":
                selected = [p for i, p in enumerate(peers) if last_round_final_ans[i] == last_vote_ans]
            else:
                selected = [p for i, p in enumerate(peers) if last_round_final_ans[i] != last_vote_ans]

            results[idx]["retained_ids"] = selected if selected else peers
            continue

        filter_prompt = build_filter_prompt_batch(
            peers=peers,
            responses=inp["responses"],
            args=args,
            last_vote_ans=last_vote_ans
        )
        llm_messages.append({'role': 'user', 'content': filter_prompt})
        llm_indices.append(idx)

    if len(llm_messages) > 0:
        chosen, uncertain, token_stats = engine_vllm_batch(
            llm_messages,
            llm,
            1,
            seed=args.seed
        )

        for local_i, global_i in enumerate(llm_indices):
            peers = batch_inputs[global_i]["peers"]
            selected = parse_filter_response(chosen[local_i].strip(), peers)
            results[global_i]["retained_ids"] = selected
            if token_stats and local_i < len(token_stats):
                results[global_i]["filter_tokens"] = token_stats[local_i]

    return results

def get_new_message_global(
    args,
    sample,
    responses,
    personas=None,
    suffix=None,
    llm=None,
    last_vote_ans=None, 
    last_round_final_ans=None,
    precomputed_retained_ids=None,
    precomputed_filter_tokens=None
):
    new_message = {}
    agents = list(responses.keys())

    multi_agent = len(agents) > 1

    # DAR
    filtered_peers_map = None
    if multi_agent and args.m_role.startswith("filter"):        
        if precomputed_retained_ids is None:
            # run ONCE
            retained_ids, filter_tokens = run_filter_batch(
                peers=agents,
                responses=responses,
                args=args,
                llm=llm,
                last_vote_ans=last_vote_ans,
                last_round_final_ans=last_round_final_ans
            )
        else:
            retained_ids = precomputed_retained_ids
            filter_tokens = precomputed_filter_tokens

        # build map: agent -> peers subset
        filtered_peers_map = {}
        for agent in agents:
            filtered_peers_map[agent] = [a for a in retained_ids if a != agent]
            
        token_logger.log(
            question=sample,
            current_agent_id="GLOBAL",
            current_response="N/A",
            peer_agent_ids=retained_ids,
            responses=f"Retained IDs: {retained_ids}, Filter Tokens: {filter_tokens}",
            fname=args.fname,
            extra={"filter_tokens": filter_tokens}
        )
        
    for i, agent in enumerate(agents):

        # ================= SINGLE AGENT =================
        if not multi_agent:
            msg = f"""This was your most recent opinion: {responses[agent]}\n\n
            Revise your recent opinion to give your updated final answer to the question:\n{sample}"""
        else:
            peers = get_peers(i, agents, args)

            original_resp = responses[agent]
            
            # ===== FILTER =====
            if multi_agent and args.m_role.startswith("filter") and filtered_peers_map is not None:
                filtered = filtered_peers_map.get(agent, peers)
                
                # Fall back
                peers = filtered if len(filtered) > 0 else peers
            
            # For Qualitative Analysis 
            if args.debug:
                for pid in peers:
                    resp = responses[pid]

                    print(f"Peer {pid} response:\n{resp}\n")

                    logger.log(
                        question=sample,
                        current_agent_id=agent,
                        current_response=original_resp,
                        peer_agent_ids=pid,
                        responses=resp,
                        fname=args.fname
                    )

            
            msg = build_normal_msg(agent, peers, responses, last_vote_ans)

            msg += f"""\n\nUse these opinions carefully as additional advice to revise your recent opinion to give your final answer to the question:
            {sample}"""

        if suffix:
            msg += suffix

        if personas:
            new_message[agent] = [
                {'role': 'system', 'content': personas[agent.split("__")[-2]]},
                {'role': 'user', 'content': msg}
            ]
        else:
            new_message[agent] = {'role': 'user', 'content': msg}

    return new_message



