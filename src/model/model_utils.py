

model_dirs = {
    'llama3.1-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama3.2-3b': 'meta-llama/Llama-3.2-3B-Instruct',
    'llama3.2-1b': 'meta-llama/Llama-3.2-1B-Instruct',
    "gemma3-1b": "google/gemma-3-1b-it",
    "gemma3-4b": "google/gemma-3-4b-it",
    "gemma2-2b": "google/gemma-2-2b-it",
    "gemma2-9b": "google/gemma-2-9b-it",
    "falcon3-1b": "tiiuae/falcon3-1b-instruct",
    "falcon3-7b": "tiiuae/falcon3-7b-instruct",
    'qwen2.5-0.5b': 'Qwen/Qwen2.5-0.5B-Instruct',
    'qwen2.5-1.5b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'qwen2.5-3b': 'Qwen/Qwen2.5-3B-Instruct',
    'qwen2.5-7b': 'Qwen/Qwen2.5-7B-Instruct',
    'qwen2.5-14b': 'Qwen/Qwen2.5-14B-Instruct',
    'qwen2.5-32b': 'Qwen/Qwen2.5-32B-Instruct'
}

import transformers
transformers.utils.logging.set_verbosity_error()
from vllm import SamplingParams
from model.vllm import vLLM


def engine_vllm_batch(messages, agent, num_agents=1, stop_sequences=None, top_k_uncertainty=None, uncertainty_metric='anll', uncertainty_prompt=None, seed=None):

    # 1. Format prompts
    if type(messages[0]) == list:
        prompts = [agent.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in messages]
    else:
        prompts = [msg['content'] for msg in messages] # observe better performance during debate
        # [agent.tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=True) for msg in messages]
    
    # 2. Setup vLLM SamplingParams
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.9,
        max_tokens=512,
        # seed=seed, # Remove seed to avoid redundant responses
        stop=stop_sequences if stop_sequences else ["<|im_end|>", "<|endoftext|>"],
        logprobs=1  
    )

    # 3. Generate using vLLM (Batch processing all messages at once)
    outputs = agent.llm.generate(prompts, sampling_params, use_tqdm=False)

    responses = []
    nll_scores = []
    token_stats = []

    # 4. Calculate NLL and ANLL natively from vLLM outputs
    for output in outputs:
        generated_sequence = output.outputs[0]
        text = generated_sequence.text.strip()
        token_ids = generated_sequence.token_ids
        logprobs_list = generated_sequence.logprobs
        
        nll = 0.0
        for i, token_id in enumerate(token_ids):
            token_logprob = logprobs_list[i][token_id].logprob
            nll -= token_logprob
            
        num_tokens = len(token_ids)
        anll = nll / num_tokens if num_tokens > 0 else 0.0
        
        responses.append(text)
        nll_scores.append((anll, nll))
        
        # Token stats
        n_tokens = len(token_ids)
        token_stats.append({
            "input_tokens": len(output.prompt_token_ids),
            "output_tokens": n_tokens,
            "total_tokens": len(output.prompt_token_ids) + n_tokens
        })
    
    # 5. Apply top-K or threshold-based uncertainty filter PER SAMPLE (Chunking)
    final_responses = []
    final_nll_scores = []
    
    # Process in chunks. Each chunk represents the agents for a single sample.
    chunk_size = num_agents
    for i in range(0, len(responses), chunk_size):
        chunk_resps = responses[i:i + chunk_size]
        chunk_nlls = nll_scores[i:i + chunk_size]
        
        if top_k_uncertainty is not None:
            if uncertainty_metric == 'nll':
                ranked = sorted(zip(chunk_resps, chunk_nlls), key=lambda x: x[1][1])
            elif uncertainty_metric == 'anll':
                ranked = sorted(zip(chunk_resps, chunk_nlls), key=lambda x: x[1][0])
            else:
                raise ValueError("Invalid uncertainty metric!")

            if top_k_uncertainty < 1:
                k = int(len(chunk_resps) * top_k_uncertainty)
                k = max(k, 1) 
            else:
                k = int(round(top_k_uncertainty))
                k = min(k, len(ranked)) 
                
            chunk_resps, chunk_nlls = zip(*ranked[:k])
            chunk_resps = list(chunk_resps)
            chunk_nlls = list(chunk_nlls)
            
        # 6. Append uncertainty prompt if required
        if uncertainty_prompt not in [None, 'None', False]:
            for j in range(len(chunk_resps)):
                chunk_resps[j] += f"\n\nUncertainty score (Average Negative Log Likelihood) for this response: {chunk_nlls[j][0]:.4f}"
                
        final_responses.extend(chunk_resps)
        final_nll_scores.extend(chunk_nlls)
    
    return final_responses, final_nll_scores, token_stats


def get_agents(args):

    agent = vLLM(args, model_dirs[args.model])
        
    # update pad token
    if agent.tokenizer.pad_token is None :
        agent.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Personas: taken from DyLAN: https://arxiv.org/pdf/2310.02170
    if args.multi_persona :
        personas = {
            "None": "",
            "Assistant": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans.",
            "Mathematician": "You are a mathematician. You are good at math games, arithmetic calculation, and long-term planning.",
            "Economist": "You are an economist. You are good at economics, finance, and business. You have experience on understanding charts while interpreting the macroeconomic environment prevailing across world economies.",
            "Psychologist": "You are a psychologist. You are good at psychology, sociology, and philosophy. You give people scientific suggestions that will make them feel better.",
            "Lawyer": "You are a lawyer. You are good at law, politics, and history.",
            "Doctor": "You are a doctor and come up with creative treatments for illnesses or diseases. You are able to recommend conventional medicines, herbal remedies and other natural alternatives. You also consider the patient’s age, lifestyle and medical history when providing your recommendations.",
            "Programmer": "You are a programmer. You are good at computer science, engineering, and physics. You have experience in designing and developing computer software and hardware.",
            "Historian": "You are a historian. You research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop theories about what happened during various periods of history.",
            "PythonAssistant": "You are a Python writing assistant, an AI that only responds with python code, NOT ENGLISH. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature).", # from https://github.com/composable-models/llm_multiagent_debate.git
            "AlgorithmDeveloper": "You are an algorithm developer. You are good at developing and utilizing algorithms to solve problems. You must respond with python code, no free-flowing text (unless in a comment). You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature).",
            "ComputerScientist": "You are a computer scientist. You are good at writing high performance code and recognizing corner cases while solve real problems. You must respond with python code, no free-flowing text (unless in a comment). You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature).",
            "CodingArtist": "You are a coding artist. You write Python code that is not only functional but also aesthetically pleasing and creative. Your goal is to make the code an art form while maintaining its utility. You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature).",
            "SoftwareArchitect": "You are a software architect, skilled in designing and structuring code for scalability, maintainability, and robustness. Your responses should focus on best practices in software design. You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature)."
        }
        if args.data in ['arithmetics','gsm8k']:
            personas = {
                "Assistant": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans.",
                "Mathematician": "You are a mathematician. You are good at math games, arithmetic calculation, and long-term planning.",
                "Lawyer": "You are a lawyer. You are good at law, politics, and history.",
                "Economist": "You are an economist. You are good at economics, finance, and business. You have experience on understanding charts while interpreting the macroeconomic environment prevailing across world economies.",
                "Programmer": "You are a programmer. You are good at computer science, engineering, and physics. You have experience in designing and developing computer software and hardware."
            }
        elif args.data in ['pro_medicine']:
            personas = {
                "Assistant": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans.",
                "Mathematician": "You are a mathematician. You are good at math games, arithmetic calculation, and long-term planning.",
                "Programmer": "You are a programmer. You are good at computer science, engineering, and physics. You have experience in designing and developing computer software and hardware.",
                "Psychologist": "You are a psychologist. You are good at psychology, sociology, and philosophy. You give people scientific suggestions that will make them feel better.",
                "Doctor": "You are a doctor and come up with creative treatments for illnesses or diseases. You are able to recommend conventional medicines, herbal remedies and other natural alternatives. You also consider the patient’s age, lifestyle and medical history when providing your recommendations."
            }

    else:
        personas = {"None": ""}

            
    return agent, personas
