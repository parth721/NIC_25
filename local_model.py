from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_community.llms import HuggingFacePipeline
from configs.params import ModelParams

model_config = ModelParams()

def local_load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_path,
        trust_remote_code=True,
        padding_side="left",
        pad_token="<|endoftext|>",
        eos_token="<|endoftext|>",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=model_config.max_tokens,
        temperature=model_config.temperature,
        top_p=0.9,
        repetition_penalty=1.1,
        batch_size=1,
        do_sample=True,
        return_full_text=False
    )

    return HuggingFacePipeline(pipeline=pipe)

