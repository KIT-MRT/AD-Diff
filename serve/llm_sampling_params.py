llm_sampling_params = {
    "qwen3": {
        # recommended parameters for Qwen3-0.6B from https://huggingface.co/Qwen/Qwen3-0.6B#best-practices
        "thinking": {
            "max_tokens": 32768,
            "temperature": 0.6, # tune
            "top_p": 0.95, # tune
            "top_k": 20, # tune
        },
        "non_thinking": {
            "max_tokens": 256,
            "temperature": 0.7, # tune
            "top_p": 0.8, # tune
            "top_k": 20, # tune
        },
    },
    "deepseek": {
        "thinking": None,  # not yet supported
        "non_thinking": { # parameters for r1 according to https://huggingface.co/RedHatAI/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic
            "max_tokens": 256,
            "temperature": 0.6,
            "top_p": None,
            "top_k": None,
        },
    },
    "gpt-oss" : {
        "thinking": {
            "max_tokens": None,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "reasoning_effort": None,
            },
        "non_thinking": {
            "max_tokens": None,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            },
        },
    "glm": {
        "thinking": {
            "max_tokens": 32768,
            "temperature": 1., # from https://huggingface.co/zai-org/GLM-4.6-FP8
            "top_p": 0.95,
            "top_k": 40,
        },
        "non_thinking": { # guessed parameters
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
        },
    },
}
