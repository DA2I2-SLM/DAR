from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class vLLM(object):
    def __init__(self, args, model_dir):
        super(vLLM, self).__init__()
        self.name = model_dir
        
        # Initialize tokenizer to handle chat templates natively
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        gpu_memory_utilization = getattr(args, 'gpu_memory_utilization', 0.9)

        # Initialize vLLM engine
        self.llm = LLM(
            model=model_dir, 
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=False, # Enable CUDA graphs for faster inference
            seed=args.seed,
            # dtype="bfloat16" # Use mixed precision for faster inference and reduced memory usage if your GPU supports it (e.g., NVIDIA Ampere or later)
        )