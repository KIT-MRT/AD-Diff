# file modified from original Apache-2.0 licensed code from https://github.com/Understanding-Visual-Datasets/VisDiff
# see LICENSE and NOTICE files in the root directory for details

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union
from abc import ABC, abstractmethod

import pandas as pd
import wandb
from numpy import random
from PIL import Image

from serve.utils_general import get_context_prompt, save_data_diff_image
from serve.utils_llm import get_llm_output
from serve.utils_vlm import get_vlm_output

try:
    import torch
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
except ImportError:
    print(
        "Warning: torch or torchvision not found. VLMQwenFeatureProposer will not work without these dependencies."
    )
    torch_available = False
else:
    torch_available = True

try:
    from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration
except ImportError:
    print(
        "Warning: transformers not found. VLMQwenFeatureProposer will not work without this dependency."
    )
    transformers_available = False
else:
    transformers_available = True

class Proposer(ABC):
    """
    Proposer is responsible for proposing hypotheses about the differences between two datasets.
    Inherit from this abstract base class if you want to implement a proposer that does not follow the two-stage approach (propose hypotheses on subsample, then rank on full dataset). If you want to implement a two-stage proposer, inherit from SamplingProposer instead.
    """

    def __init__(self, args: Dict, rng: random.Generator):
        self.args = args
        self.rng = rng


    @abstractmethod
    def propose(
        self, dataset1: List[Dict], dataset2: List[Dict]
    ) -> Tuple[List[str], List[Dict], List[Dict]]:
        """
        Proposer is responsible for proposing hypotheses about the differences between two datasets.
        The visualize method can be used to return any images to be logged in wandb.

        Override this method to implement your proposal logic (if you don't want to use a two-stage approach, otherwise inherit from SamplingProposer). You can use the get_vlm_output and get_llm_output functions from serve/utils_vlm.py and serve/utils_llm.py to get outputs from VLMs and LLMs respectively.

        Args:
            dataset1: List of samples from dataset 1, where each sample is a dictionary with at least a "path" key pointing to an image file.
            dataset2: List of samples from dataset 2, where each sample is a dictionary with at least a "path" key pointing to an image file.

        Returns:
            hypotheses: A list of strings, where each string is a hypothesis about the differences between the two datasets.
            logs: A list of dictionaries, where each dictionary contains any relevant logs for the corresponding hypothesis. This can include prompts, model outputs, or any other information you want to log to wandb.
            images: A list of dictionaries, where each dictionary contains a key "images_group_1" mapping to a list of wandb.Image objects for dataset 1, and a key "images_group_2" mapping to a list of wandb.Image objects for dataset 2. This can be used to log images to wandb for visualization.
        """
        raise NotImplementedError

    def visualize(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
    ) -> Dict:
        """
        Given two sampled datasets, return a dictionary of images to log to wandb.
        The returned dictionary should have the format:
        {
            "images_group_1": [wandb.Image, wandb.Image, ...],
            "images_group_2": [wandb.Image, wandb.Image, ...],
        }
        where the lists of wandb.Image objects correspond to the images in the sampled datasets.

        Args:
            sampled_dataset1: List of samples from dataset 1, where each sample is a dictionary with at least a "path" key pointing to an image file.
            sampled_dataset2: List of samples from dataset 2, where each sample is a dictionary with at least a "path" key pointing to an image file.

        Returns:
            A dictionary with keys "images_group_1" and "images_group_2", where the values are lists of wandb.Image objects corresponding to the images in the sampled datasets.
        """

        images1 = [
            wandb.Image(
                Image.open(item["path"]).convert("RGB").resize((224, 224)),
                caption=item.get("caption", ""),
            )
            for item in sampled_dataset1
        ]
        images2 = [
            wandb.Image(
                Image.open(item["path"]).convert("RGB").resize((224, 224)),
                caption=item.get("caption", ""),
            )
            for item in sampled_dataset2
        ]
        images = {"images_group_1": images1, "images_group_2": images2}
        return images

class SamplingProposer(Proposer):
    """
    SamplingProposer is a Proposer that follows a two-stage approach: first it samples a subset of the datasets, then it proposes hypotheses based on the sampled subsets. The proposed hypotheses can then be ranked on the full datasets by a Ranker.
    """

    def __init__(self, args: Dict, rng: random.Generator):
        self.args = args
        self.vlm_hostname = args.get("vlm_hostname", "localhost")
        self.rng = rng
        print(f"Initialized proposer with VLM hostname: {self.vlm_hostname}")

    def propose(
        self, dataset1: List[Dict], dataset2: List[Dict]
    ) -> Tuple[List[str], List[Dict], List[Dict]]:
        """
        Given two datasets, return a list of hypotheses
        """
        all_hypotheses = []
        all_logs = []
        all_images = []
        for i in range(self.args["num_rounds"]):
            sample_size = min(self.args["num_samples"], len(dataset1), len(dataset2))
            if sample_size < self.args["num_samples"]:
                print(
                    f"Warning: sample size {sample_size} is less than requested {self.args['num_samples']} for datasets with first member {dataset1[0]} and {dataset2[0]}"
                )
            sampled_dataset1 = self.sample(dataset1, sample_size)
            sampled_dataset2 = self.sample(dataset2, sample_size)
            hypotheses, logs = self.get_hypotheses(sampled_dataset1, sampled_dataset2)
            images = self.visualize(sampled_dataset1, sampled_dataset2)
            all_hypotheses += hypotheses
            all_logs.append(logs)
            all_images.append(images)
        return all_hypotheses, all_logs, all_images

    @abstractmethod
    def get_hypotheses(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
    ) -> Tuple[List[str], Dict]:
        """
        Given two sampled datasets, return a list of hypotheses about the differences between the two datasets, and any relevant logs.
        The logs can include any information you want to log to wandb, such as prompts, model outputs, or any other relevant information.

        Override this method to implement your hypothesis generation logic. You can use the get_vlm_output and get_llm_output functions from serve/utils_vlm.py and serve/utils_llm.py to get outputs from VLMs and LLMs respectively.

        Args:
            sampled_dataset1: List of samples from dataset 1, where each sample is a dictionary with at least a "path" key pointing to an image file.
            sampled_dataset2: List of samples from dataset 2, where each sample is a dictionary with at least a "path" key pointing to an image file.

        Returns:
            hypotheses: A list of strings, where each string is a hypothesis about the differences between the two datasets.
            logs: A dictionary containing any relevant logs for the generated hypotheses. This can include prompts, model outputs, or any other information you want to log to wandb.
        """
        raise NotImplementedError

    def sample(self, dataset: List[Dict], n: int) -> List[Dict]:
        indices = self.rng.choice(len(dataset), n, replace=False)
        return [dataset[i] for i in indices]


class LLMProposer(SamplingProposer):
    def __init__(self, args: Dict, rng: random.Generator):
        super().__init__(args, rng=rng)
        self.prompt = get_context_prompt(args["prompt"], self.args["context_mode"])
        self.llm_hostname = args.get("proposer_llm_hostname", "localhost")
        print(f"Initialized proposer with LLM hostname: {self.llm_hostname}")

    def captioning(self, dataset: List[Dict]):
        for item in dataset:
            item["caption"] = get_vlm_output(
                item["path"],
                get_context_prompt(
                    self.args["captioner"]["prompt"], self.args["context_mode"]
                ),
                self.args["captioner"]["model"],
                self.vlm_hostname,
            )

    def get_hypotheses(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
    ) -> Tuple[List[str], Dict]:
        self.captioning(sampled_dataset1)
        self.captioning(sampled_dataset2)
        captions1 = [
            f"Group A: {item['caption']}".replace("\n", " ").strip()
            for item in sampled_dataset1
        ]
        captions2 = [
            f"Group B: {item['caption']}".replace("\n", " ").strip()
            for item in sampled_dataset2
        ]
        caption_concat = "\n".join(captions1 + captions2)
        prompt = self.prompt.format(text=caption_concat)
        output, _ = get_llm_output(
            prompt, self.args["model"], vllm_hostname=self.llm_hostname
        )
        hypotheses = [line.replace("* ", "") for line in output.splitlines()]
        logs = {"prompt": prompt, "output": output}
        return hypotheses, logs


class VLMProposer(SamplingProposer):
    """
    Concatenate images and ask VLM to find differences
    """

    def __init__(self, args: Dict, rng: random.Generator):
        super().__init__(args, rng=rng)
        self.prompt = get_context_prompt(args["prompt"], self.args["context_mode"])

    def get_hypotheses(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
    ) -> Tuple[List[str], Dict]:
        assert len(sampled_dataset1) == len(sampled_dataset2), (
            "Groups must be of equal size"
        )
        assert len(sampled_dataset1) <= 20, "Groups must be smaller than 20"
        filenames = [item["path"] for item in sampled_dataset1 + sampled_dataset2]
        save_name = hashlib.sha256(json.dumps(filenames).encode()).hexdigest()

        image_path = f"cache/images/{save_name}.png"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        save_data_diff_image(sampled_dataset1, sampled_dataset2, image_path)
        output = get_vlm_output(
            image_path, self.prompt, self.args["model"], self.vlm_hostname
        )
        output = output.replace("</s>", " ").strip()  # remove </s> token for llava
        hypotheses = [line.replace("* ", "") for line in output.splitlines()]
        logs = {"image": image_path, "prompt": self.prompt, "output": output}
        return hypotheses, logs


class VLMMultipleImageProposer(SamplingProposer):
    """
    Provide multiple images to VLM and ask to find differences
    """

    def __init__(self, args: Dict, rng: random.Generator):
        super().__init__(args, rng=rng)
        self.prompt1 = get_context_prompt(args["prompt1"], self.args["context_mode"])
        self.prompt2 = get_context_prompt(args["prompt2"], self.args["context_mode"])
        self.prompt3 = get_context_prompt(args["prompt3"], self.args["context_mode"])
        self.system_prompt = get_context_prompt(
            args["system_prompt"], self.args["context_mode"]
        )

    def get_hypotheses(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
    ) -> Tuple[List[str], Dict]:
        image_paths1 = [item["path"] for item in sampled_dataset1]
        image_paths2 = [item["path"] for item in sampled_dataset2]

        output = get_vlm_output(
            image_paths1,
            self.prompt1,
            self.args["model"],
            self.vlm_hostname,
            images2=image_paths2,
            prompt2=self.prompt2,
            system_prompt=self.system_prompt,
            prompt3=self.prompt3,
        )
        hypotheses = [line.replace("* ", "") for line in output.splitlines()]
        logs = {
            "images1": image_paths1,
            "images2": image_paths2,
            "prompt1": self.prompt1,
            "prompt2": self.prompt2,
            "output": output,
        }
        return hypotheses, logs


if torch_available and transformers_available:
    class VLMQwenFeatureProposer(SamplingProposer):
        def __init__(self, args: Dict, rng: random.Generator):
            super().__init__(args, rng=rng)
            print("Initializing proposer with Qwen features...")

            model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct"
            self.processor = self._make_qwen3vl_processor_openclip224(model_id)
            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",  # shard across GPUs
                torch_dtype="auto",  # pick fp16/bf16 if supported
                local_files_only=False,
            ).eval()

            # OpenCLIP-like image transform (no aug, inference, 224)
            self.openclip224_transform = T.Compose(
                [
                    # resize shortest edge to 224 (keeps aspect ratio)
                    T.Resize(224, interpolation=InterpolationMode.BICUBIC),
                    T.CenterCrop((224, 224)),
                ]
            )

            print("Initialized proposer with Qwen features")

        def _load_image(self, path: Union[str, Path]) -> Image.Image:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.load()  # force decode while file is open
                return self.openclip224_transform(im)

        @staticmethod
        def _make_qwen3vl_processor_openclip224(model_id: str):
            processor = AutoProcessor.from_pretrained(model_id, local_files_only=False)
            ip = processor.image_processor

            ip.do_resize = False  # we already resized/cropped to 224
            ip.do_convert_rgb = False  # we already convert to RGB
            ip.do_rescale = True
            ip.do_normalize = True
            ip.image_mean = (0.48145466, 0.4578275, 0.40821073)
            ip.image_std = (0.26862954, 0.26130258, 0.27577711)
            return processor

        @torch.inference_mode()
        def _get_avg_image_features(self, dataset: List[Dict]):
            images = [self._load_image(sample["path"]) for sample in dataset]
            batch = self.processor.image_processor(images=images, return_tensors="pt")

            # Device + dtype for the vision tower
            vision_device = next(self.model.model.visual.parameters()).device
            vision_dtype = next(self.model.model.visual.parameters()).dtype

            pixel_values = batch["pixel_values"].to(
                device=vision_device, dtype=vision_dtype
            )
            image_grid_thw = batch["image_grid_thw"].to(device=vision_device)

            img_out = self.model.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                return_dict=True,
            )
            features = img_out.pooler_output  # tuple of tensors (T, H) with length B
            features = torch.stack(list(features), dim=0)  # (B, T, H)

            if features.ndim != 3:
                raise ValueError(
                    f"Expected image features (B,T,H), got {tuple(features.shape)}"
                )

            return features.mean(dim=0)  # (T, H)

        @torch.inference_mode()
        def _get_text_features_and_masks(self, example_image_path: Union[str, Path]):
            prompt = self.args["prompt"]
            # Qwen3-VL chat format: messages with [{"type":"image"}, {"type":"text"}]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Build text with template; important to add generation prompt
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            embed_device = self.model.get_input_embeddings().weight.device
            embed_dtype = self.model.get_input_embeddings().weight.dtype

            example_img = self._load_image(example_image_path)
            batch = self.processor(text=[text], images=[example_img], return_tensors="pt")
            input_ids = batch["input_ids"].to(device=embed_device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device=embed_device)

            # Text token embeddings
            inputs_embeds = self.model.get_input_embeddings()(input_ids).to(
                device=embed_device, dtype=embed_dtype
            )

            # Replace image placeholder token positions with avg_img_tokens
            image_token_id = self.model.config.image_token_id
            image_mask = (
                (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            ).to(device=embed_device)

            return inputs_embeds, image_mask, attention_mask, input_ids

        @torch.inference_mode()
        def get_hypotheses(
            self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
        ) -> Tuple[List[str], Dict]:
            # Find average image features
            avg_feature_1 = self._get_avg_image_features(sampled_dataset1)
            avg_feature_2 = self._get_avg_image_features(sampled_dataset2)
            avg_feature_diff = avg_feature_1 - avg_feature_2

            # Get text embedding, image mask, attention_mask
            inputs_embeds, image_mask, attention_mask, input_ids = (
                self._get_text_features_and_masks(sampled_dataset1[0]["path"])
            )

            # Put model input together
            n_img_tokens = int(image_mask[..., 0].sum().item())
            T, H = avg_feature_diff.shape
            if n_img_tokens != T:
                raise ValueError(
                    f"Mismatch: prompt has {n_img_tokens} image tokens, avg_img_tokens has T={T}. "
                    f"Make sure preprocessing + processor settings match between feature extraction and templating."
                )
            if inputs_embeds.shape[-1] != H:
                raise ValueError(
                    f"Hidden size mismatch: embeds H={inputs_embeds.shape[-1]} vs avg H={H}"
                )

            img_feats = avg_feature_diff.reshape(-1, H).to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, img_feats)

            lm_device = next(self.model.model.language_model.parameters()).device

            answers = []
            for _ in range(self.args["num_hypotheses"]):
                # Generate
                out_ids = self.model.generate(
                    input_ids=input_ids.to(device=lm_device),
                    inputs_embeds=inputs_embeds.to(device=lm_device),
                    attention_mask=attention_mask.to(device=lm_device),
                    max_new_tokens=128,
                )

                # Extract top caption
                diff_caption = self.processor.tokenizer.decode(
                    out_ids[0], skip_special_tokens=True
                )

                # keep everything after the last "assistant" line
                marker = "\nassistant\n"
                if marker in diff_caption:
                    answers.append(diff_caption.split(marker)[-1].strip())

            logs = {"output": answers}
            return answers, logs
else:
    print("Skipping definition of VLMQwenFeatureProposer since torch or transformers is not available.")


class NullProposer(SamplingProposer):
    def __init__(self, args: Dict, rng: random.Generator):
        args["num_rounds"] = 0
        super().__init__(args, rng=rng)

    def get_hypotheses(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
    ) -> Tuple[List[str], Dict]:
        return ["No hypothesis"], {}

