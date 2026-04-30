# file modified from original Apache-2.0 licensed code from https://github.com/Understanding-Visual-Datasets/VisDiff
# see LICENSE and NOTICE files in the root directory for details

import json
import logging
import os
from typing import List

import lmdb
import openai

from serve.global_vars import LLM_CACHE_FILE, VLLM_PORT, USE_CACHE
from serve.utils_general import get_from_cache, save_to_cache
from serve.llm_sampling_params import llm_sampling_params

logging.basicConfig(level=logging.INFO)

if not os.path.exists(LLM_CACHE_FILE):
    os.makedirs(LLM_CACHE_FILE)

if USE_CACHE:
    llm_cache = lmdb.open(LLM_CACHE_FILE, map_size=int(1e11))


def get_llm_output(prompt: str, model: str, vllm_hostname: str, think: bool = False, greedy = False, from_choices = None, system_prompt = None) -> tuple[str, str]:
    api_base = {
        "gpt-3.5-turbo": "https://api.openai.com/v1",
        "gpt-4": "https://api.openai.com/v1"
    }
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "42"),
        base_url=api_base.get(model, f"http://{vllm_hostname}:{VLLM_PORT}/v1"),
    )
    system_messages = [{"role": "system", "content": system_prompt}] if system_prompt is not None else []
    if model in ["gpt-3.5-turbo", "gpt-4"]:
        messages = [{"role": "system", "content": "You are a helpful assistant."}] + system_messages + [
            {"role": "user", "content": prompt},
        ]

    elif "vicuna" in model.lower():
        messages=system_messages + [
        {"role": "user", "content": prompt}
        ]
        messages = prompt
    elif "gpt-oss" in model.lower():
        messages=[]
        if think:
            assert llm_sampling_params['gpt-oss']['thinking']['reasoning_effort'] in ["low", "medium", "high", None], "Invalid reasoning effort level."
            if llm_sampling_params['gpt-oss']['thinking']['reasoning_effort'] is not None:
                messages.append({"role": "system", "content": f"Reasoning: {llm_sampling_params['gpt-oss']['thinking']['reasoning_effort']}."})
        messages.extend(system_messages)
        messages.append({"role": "user", "content": prompt})
    else:
        messages = system_messages + [
        {"role": "user", "content": prompt}
        ]

    key = json.dumps([model, messages])


    if USE_CACHE:
        cached_value = get_from_cache(key, llm_cache)
        if cached_value is not None:
            logging.debug(f"LLM Cache Hit")
            return cached_value, ""

    for _ in range(3):
        try:
            reasoning = ""
            if model in ["gpt-3.5-turbo", "gpt-4"]:
                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                )
                response = completion["choices"][0]["message"]["content"]
            elif model == "vicuna":
                completion = openai.Completion.create(
                    model="lmsys/vicuna-7b-v1.5",
                    prompt=prompt,
                    max_tokens=256,
                    temperature=0,  # TODO: greedy may not be optimal
                )
                response = completion["choices"][0]["text"]
            else:
                assert not ("thinking" in model.lower() and not think or "instruct" in model.lower() and think), "Instruct models cannot think, thinking models must think."
                if "qwen3" in model.lower():
                    llm_param_set = llm_sampling_params["qwen3"]
                elif "deepseek" in model.lower():
                    llm_param_set = llm_sampling_params["deepseek"]
                elif "gpt-oss" in model.lower():
                    llm_param_set = llm_sampling_params["gpt-oss"]
                elif "glm" in model.lower():
                    llm_param_set = llm_sampling_params["glm"]
                else:
                    raise ValueError(f"Model {model} not supported.")
                llm_param_set = llm_param_set["thinking" if think else "non_thinking"]
                if greedy and not think:
                    extra_args = dict(
                        max_tokens=llm_param_set["max_tokens"],
                        temperature=0.,
                        top_p=0.1,
                        extra_body={
                            "top_k": 1,
                            "chat_template_kwargs": {"enable_thinking": False},
                    },
                    )
                elif not greedy:
                    extra_args = dict(
                        max_tokens=llm_param_set["max_tokens"],
                        temperature=llm_param_set["temperature"],
                        top_p=llm_param_set["top_p"],
                        extra_body={
                            "top_k": llm_param_set["top_k"],
                            "chat_template_kwargs": {"enable_thinking": think},
                        },
                    )
                else:
                    raise ValueError("Invalid combination of greedy and think parameters, you shouldn't think greedily, it results in nonsense.")
                if from_choices is not None and len(from_choices) > 0:
                    if len(from_choices) == 1:
                        logging.warning("Only one choice provided, not much to choose from.")
                    extra_args["extra_body"]["guided_choice"] = from_choices
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **extra_args
                )

                response = completion.choices[0].message.content
                reasoning = completion.choices[0].message.reasoning_content
            if USE_CACHE:
                save_to_cache(key, response, llm_cache)
            return response, reasoning

        except Exception as e:
            logging.error(f"LLM Error: {e}")
            last_exception = e
            continue
    else:
        raise last_exception


def prompt_differences(captions1: List[str], captions2: List[str]) -> str:
    caption1_concat = "\n".join(
        [f"Image {i + 1}: {caption}" for i, caption in enumerate(captions1)]
    )
    caption2_concat = "\n".join(
        [f"Image {i + 1}: {caption}" for i, caption in enumerate(captions2)]
    )
    prompt = f"""Here are two groups of images:

Group 1:
```
{caption1_concat}
```

Group 2:
```
{caption2_concat}
```

What are the differences between the two groups of images?
Think carefully and summarize each difference in JSON format, such as:
```
{{"difference": several words, "rationale": group 1... while group 2...}}
```
Output JSON only. Do not include any other information.
"""
    return prompt


def get_differences(captions1: List[str], captions2: List[str], model: str, vllm_hostname: str) -> str:
    prompt = prompt_differences(captions1, captions2)
    differences, _ = get_llm_output(prompt, model, vllm_hostname)
    try:
        differences = json.loads(differences)
    except Exception as e:
        logging.error(f"Difference Error: {e}")
        raise e
    return differences
