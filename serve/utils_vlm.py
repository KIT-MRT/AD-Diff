# file modified from original Apache-2.0 licensed code from https://github.com/Understanding-Visual-Datasets/VisDiff
# see LICENSE and NOTICE files in the root directory for details

import json
import logging
import base64

logging.basicConfig(level=logging.INFO)

import os
from typing import Dict, List

import lmdb
import openai

from serve.global_vars import VLM_CACHE_FILE, USE_CACHE, VLLM_PORT
from serve.utils_general import get_from_cache, save_to_cache

if not os.path.exists(VLM_CACHE_FILE):
    os.makedirs(VLM_CACHE_FILE)

if USE_CACHE:
    vlm_cache = lmdb.open(VLM_CACHE_FILE, map_size=int(1e11))

def get_image_base64(image_path):
    with open(image_path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')

def get_vlm_output(images: list | str, prompt: str, model: str, hostname: str, get_reasoning: bool = False, images2: list | None = None, prompt2: str | None = None, system_prompt: str | None = None, prompt3: str | None = None) -> str:
    if isinstance(images, str):
        images = [images]
    system_messages = [{"role": "system", "content": system_prompt}] if system_prompt is not None else []
    key = json.dumps([model, *images, prompt])
    if USE_CACHE:
        cached_value = get_from_cache(key, vlm_cache)
        if cached_value is not None:
            logging.debug(f"VLM Cache Hit")
            return cached_value

    if "qwen3-vl" in model.lower():
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "42"),
            base_url=f"http://{hostname}:{VLLM_PORT}/v1",
        )
        messages = system_messages + [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ] + [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{get_image_base64(image)}"}} for image in images
                    ] + ([
                        {"type": "text", "text": prompt2}
                    ] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{get_image_base64(image)}"}} for image in images2
                    ] if images2 and prompt2 else []) + ([
                        {"type": "text", "text": prompt3}] if prompt3 else [])
                }
            ]

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2048
            )
            response = completion.choices[0].message.content
            reasoning = completion.choices[0].message.reasoning_content if hasattr(completion.choices[0].message, 'reasoning_content') else ""
            if USE_CACHE:
                save_to_cache(key, response, vlm_cache)
            if get_reasoning:
                return response, reasoning
            return response
        except Exception as e:
            logging.error(f"VLM Error: {e}")
            raise e
    else:
        raise NotImplementedError(f"VLM model {model} not implemented.")
