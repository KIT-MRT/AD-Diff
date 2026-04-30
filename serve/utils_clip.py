# file modified from original Apache-2.0 licensed code from https://github.com/Understanding-Visual-Datasets/VisDiff
# see LICENSE and NOTICE files in the root directory for details

import json
import logging
import os
from typing import List
import pathlib

import lmdb
import numpy as np
import openai
from PIL import Image

from serve.global_vars import CLIP_CACHE_FILE, CLIP_PORT, USE_CACHE
from serve.utils_general import get_from_cache, save_to_cache

if not os.path.exists(CLIP_CACHE_FILE):
    os.makedirs(CLIP_CACHE_FILE)

if USE_CACHE:
    clip_cache = lmdb.open(CLIP_CACHE_FILE, map_size=int(1e11))


def get_embeddings(inputs: List[str], model: str, modality: str, clip_hostname: str) -> np.ndarray:
    input_to_embeddings = {}
    for inp in inputs:
        key = json.dumps([inp, model])
        if USE_CACHE:
            cached_value = get_from_cache(key, clip_cache)
            if cached_value is not None:
                logging.debug(f"CLIP Cache Hit")
                input_to_embeddings[inp] = json.loads(cached_value)

    uncached_inputs = [inp for inp in inputs if inp not in input_to_embeddings]

    if len(uncached_inputs) > 0:
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "42"),
            base_url=f"http://{clip_hostname}:{CLIP_PORT}/v1",
        )
        try:
            for input in uncached_inputs: # vllm embed api only supports one input at a time
                siglip_mm_processor_kwargs = {"padding": "max_length", "truncation": True, "max_length": 64}
                clip_mm_processor_kwargs = {"padding": True}
                if modality == "image":
                    abs_path_input = str(pathlib.Path(input).resolve())
                answer = client.post(
                    "/embeddings",
                    cast_to = openai.types.create_embedding_response.CreateEmbeddingResponse,
                    body={
                        "model": model,
                        "encoding_format": "float",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": f'file://{abs_path_input}'}} if modality == "image" else {"type": "text", "text": input}
                                ],
                            }
                        ],
                        "mm_processor_kwargs": siglip_mm_processor_kwargs if "siglip" in model.lower() else clip_mm_processor_kwargs
                    },
                )
                embedding = answer.data[0].embedding
                input_to_embeddings[input] = embedding
                if USE_CACHE:
                    key = json.dumps([input, model])
                    save_to_cache(key, json.dumps(embedding), clip_cache)
        except Exception as e:
            logging.error(f"CLIP Error for input {input}: {e}")
            raise e

    input_embeddings = [input_to_embeddings[inp] for inp in inputs]
    return np.array(input_embeddings)

