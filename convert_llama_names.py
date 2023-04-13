import json


def convert_llama_names(model_name: str) -> None:
    """Convert LlamaForCausalLM to LlamaForCausalLM and 
        LLaMATokenizer to LlamaTokenizer"""
    with open(f"{model_name}/config.json", "r", encoding='utf-8') as f:
        data = f.read()

    config = json.loads(data)
    config["architectures"] = ["LlamaForCausalLM"]
    with open(f"{model_name}/config.json", "w", encoding='utf-8') as f:
        json.dump(config, f)


    with open(f"{model_name}/tokenizer_config.json", "r", encoding='utf-8') as f:
        data = f.read()

    config = json.loads(data)
    config["tokenizer_class"] = "LlamaTokenizer"

    with open(f"{model_name}/tokenizer_config.json", "w", encoding='utf-8') as f:
        json.dump(config, f)


if __name__ == "__main__":
    convert_llama_names("llama-7b-hf")
    convert_llama_names("llama-13b-hf")
