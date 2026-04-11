# Arithmetics with qwen2.5-1.5b
python src/main.py --model qwen2.5-1.5b --num_agents 1 --data arithmetics --data_size 100 --debate_rounds 1 --seed 42

python src/main.py --model qwen2.5-1.5b --num_agents 4 --data arithmetics --data_size 100 --debate_rounds 2 --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data arithmetics --data_size 100 --debate_rounds 2 --top_k_uncertainty 0.5 --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data arithmetics --data_size 100 --debate_rounds 2 --uncertainty_prompt True --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data arithmetics --data_size 100 --debate_rounds 2 --vote_prompt True --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data arithmetics --data_size 100 --debate_rounds 2 --uncertainty_prompt True --vote_prompt True --m_role filter_critical --seed 42

# GSM8K with qwen2.5-1.5b
python src/main.py --model qwen2.5-1.5b --num_agents 1 --data gsm8k --data_size 300 --debate_rounds 1 --seed 42

python src/main.py --model qwen2.5-1.5b --num_agents 4 --data gsm8k --data_size 300 --debate_rounds 2 --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data gsm8k --data_size 300 --debate_rounds 2 --top_k_uncertainty 0.5 --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data gsm8k --data_size 300 --debate_rounds 2 --uncertainty_prompt True --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data gsm8k --data_size 300 --debate_rounds 2 --vote_prompt True --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data gsm8k --data_size 300 --debate_rounds 2 --uncertainty_prompt True --vote_prompt True --m_role filter_critical --seed 42

# HH-RLHF with qwen2.5-1.5b
python src/main.py --model qwen2.5-1.5b --num_agents 1 --data hh_rlhf --data_size 300 --debate_rounds 1 --seed 42

python src/main.py --model qwen2.5-1.5b --num_agents 4 --data hh_rlhf --data_size 300 --debate_rounds 2 --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data hh_rlhf --data_size 300 --debate_rounds 2 --top_k_uncertainty 0.5 --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data hh_rlhf --data_size 300 --debate_rounds 2 --uncertainty_prompt True --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data hh_rlhf --data_size 300 --debate_rounds 2 --vote_prompt True --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data hh_rlhf --data_size 300 --debate_rounds 2 --uncertainty_prompt True --vote_prompt True --m_role filter_critical --seed 42

# Form.Log. with qwen2.5-1.5b
python src/main.py --model qwen2.5-1.5b --num_agents 1 --data formal_logic --debate_rounds 1 --seed 42

python src/main.py --model qwen2.5-1.5b --num_agents 4 --data formal_logic --debate_rounds 2 --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data formal_logic --debate_rounds 2 --top_k_uncertainty 0.5 --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data formal_logic --debate_rounds 2 --uncertainty_prompt True --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data formal_logic --debate_rounds 2 --vote_prompt True --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data formal_logic --debate_rounds 2 --uncertainty_prompt True --vote_prompt True --m_role filter_critical --seed 42

# Prod.Med. with qwen2.5-1.5b
python src/main.py --model qwen2.5-1.5b --num_agents 1 --data pro_medicine --debate_rounds 1 --seed 42

python src/main.py --model qwen2.5-1.5b --num_agents 4 --data pro_medicine --debate_rounds 2 --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data pro_medicine --debate_rounds 2 --top_k_uncertainty 0.5 --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data pro_medicine --debate_rounds 2 --uncertainty_prompt True --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data pro_medicine --debate_rounds 2 --vote_prompt True --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data pro_medicine --debate_rounds 2 --uncertainty_prompt True --vote_prompt True --m_role filter_critical --seed 42

# CSQA with qwen2.5-1.5b
python src/main.py --model qwen2.5-1.5b --num_agents 1 --data csqa --data_size 300 --debate_rounds 1 --seed 42

python src/main.py --model qwen2.5-1.5b --num_agents 4 --data csqa --data_size 300 --debate_rounds 2 --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data csqa --data_size 300 --debate_rounds 2 --top_k_uncertainty 0.5 --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data csqa --data_size 300 --debate_rounds 2 --uncertainty_prompt True --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data csqa --data_size 300 --debate_rounds 2 --vote_prompt True --seed 42
python src/main.py --model qwen2.5-1.5b --num_agents 4 --data csqa --data_size 300 --debate_rounds 2 --uncertainty_prompt True --vote_prompt True --m_role filter_critical --seed 42