qwen_template = {
    "system_format": "<|im_start|>system\n{content}<|im_end|>\n",
    "user_format": "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
    "assistant_format": "{content}<|im_end|>\n",
    "system": "You are a helpful assistant.",
}

gemma_template = {
    "system_format": "<bos>",
    "user_format": "<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n",
    "assistant_format": "{content}<|eot_id|>",
    "system": None,
}

llama3_template = {
    "system_format": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
    "user_format": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "assistant_format": "{content}<|eot_id|>",
    "system": None,
    "stop_word": "<|eot_id|>",
}

tinyllama_template = {
    "system_format": "<|system|>\n{content}\n<|endoftext|>\n",
    "user_format": "<|user|>\n{content}\n<|endoftext|>\n<|assistant|>\n",
    "assistant_format": "{content}<|endoftext|>\n",
    "system": "You are a helpful assistant.",
}

model2template = {
    "Qwen/Qwen1.5-0.5B": qwen_template,
    "Qwen/Qwen1.5-1.8B": qwen_template,
    "Qwen/Qwen1.5-7B": qwen_template,
    "Qwen/Qwen2.5-1.5B-Instruct": qwen_template,
    "google/gemma-2b": gemma_template,
    "google/gemma-7b": gemma_template,
    "meta-llama/Llama-3.2-1B": llama3_template,
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": tinyllama_template,
}
