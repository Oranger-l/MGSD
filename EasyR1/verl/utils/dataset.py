# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    video_fps: float,
    return_fps: bool = False,
    return_metadata: bool = False,
) -> Any:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    return fetch_video(vision_info, return_video_sample_fps=return_fps, return_video_metadata=return_metadata)


def resolve_prompt_image_paths(prompt_image_paths: Optional[Union[str, list[str], tuple[str, ...]]]) -> list[str]:
    if not prompt_image_paths:
        return []

    if isinstance(prompt_image_paths, str):
        raw_items = [item.strip() for item in prompt_image_paths.split(",") if item.strip()]
    else:
        raw_items = [str(item).strip() for item in prompt_image_paths if str(item).strip()]

    resolved_paths = []
    for raw_item in raw_items:
        candidate = os.path.expanduser(raw_item)
        if not os.path.isabs(candidate):
            candidate = os.path.abspath(candidate)
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Prompt image path not found: {raw_item}")
        resolved_paths.append(candidate)

    return resolved_paths


def maybe_load_frozenlake_text_map(image_path: Union[str, os.PathLike[str]]) -> Optional[str]:
    """Load the sibling FrozenLake text map for a target image path if present."""
    image_path = os.path.abspath(os.fspath(image_path))
    parent = os.path.dirname(image_path)
    stem, _ = os.path.splitext(os.path.basename(image_path))
    if os.path.basename(parent) != "image":
        return None

    text_path = os.path.join(os.path.dirname(parent), "text", f"{stem}.txt")
    if not os.path.exists(text_path):
        return None

    with open(text_path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    desc = []
    in_map = False
    for line in lines:
        if in_map:
            if line.strip():
                desc.append(line.strip())
            continue
        if line.strip() == "map:":
            in_map = True

    if not desc:
        return None
    return "\n".join(desc)


def maybe_load_frozenlake_task_text(image_path: Union[str, os.PathLike[str]]) -> Optional[str]:
    """Load the full sibling FrozenLake text task file for a target image path if present."""
    image_path = os.path.abspath(os.fspath(image_path))
    parent = os.path.dirname(image_path)
    stem, _ = os.path.splitext(os.path.basename(image_path))
    if os.path.basename(parent) != "image":
        return None

    text_path = os.path.join(os.path.dirname(parent), "text", f"{stem}.txt")
    if not os.path.exists(text_path):
        return None

    with open(text_path, encoding="utf-8") as f:
        return f.read()


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        teacher_image_key: Optional[str] = None,
        teacher_text_key: Optional[str] = None,
        return_raw_chat: bool = False,
        system_prompt: Optional[str] = None,
        prompt_image_paths: Optional[Union[str, list[str], tuple[str, ...]]] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.teacher_image_key = teacher_image_key
        self.teacher_text_key = teacher_text_key
        self.return_raw_chat = return_raw_chat
        self.system_prompt = None
        if system_prompt and os.path.isfile(system_prompt):
            with open(system_prompt, "r") as f:
                self.system_prompt = f.read().strip()
        elif system_prompt:
            self.system_prompt = system_prompt
        self.prompt_image_paths = resolve_prompt_image_paths(prompt_image_paths)

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    @staticmethod
    def _append_text_content(content_list: list[dict[str, Any]], text: str) -> None:
        if text:
            content_list.append({"type": "text", "text": text})

    def _build_prompt_image_content(
        self, prompt_str: str, fallback_image_count: int = 0
    ) -> list[dict[str, Any]]:
        prompt_parts = prompt_str.split("<image>")
        content_list = []

        if len(prompt_parts) == 1:
            self._append_text_content(content_list, prompt_parts[0])
            for _ in range(fallback_image_count):
                content_list.append({"type": "image"})
            return content_list

        for i, content in enumerate(prompt_parts):
            self._append_text_content(content_list, content)
            if i != len(prompt_parts) - 1:
                content_list.append({"type": "image"})

        return content_list

    def _uses_interleaved_system_prompt(self) -> bool:
        placeholder_tokens = ("<IMAGE-1>", "<IMAGE-2>", "<TEST-IMAGE>")
        return bool(self.system_prompt and any(token in self.system_prompt for token in placeholder_tokens))

    def _get_interleaved_prompt_image_paths(self) -> list[str]:
        if not self._uses_interleaved_system_prompt():
            return list(self.prompt_image_paths)

        interleaved_prompt_images: list[str] = []
        if "<IMAGE-1>" in self.system_prompt:
            if len(self.prompt_image_paths) < 1:
                raise ValueError("System prompt uses <IMAGE-1> but no prompt image path was provided.")
            interleaved_prompt_images.append(self.prompt_image_paths[0])
        if "<IMAGE-2>" in self.system_prompt:
            if len(self.prompt_image_paths) < 2:
                raise ValueError("System prompt uses <IMAGE-2> but fewer than 2 prompt image paths were provided.")
            interleaved_prompt_images.append(self.prompt_image_paths[1])
        return interleaved_prompt_images

    def _build_interleaved_system_image_content(self, example_image_count: int) -> Optional[list[dict[str, Any]]]:
        if not self._uses_interleaved_system_prompt():
            return None

        placeholder_tokens = ("<IMAGE-1>", "<IMAGE-2>", "<TEST-IMAGE>")
        content_list = []
        remaining_text = self.system_prompt

        while remaining_text:
            positions = [(remaining_text.find(token), token) for token in placeholder_tokens]
            positions = [(pos, token) for pos, token in positions if pos != -1]
            if not positions:
                self._append_text_content(content_list, remaining_text)
                break

            next_pos, next_token = min(positions, key=lambda item: item[0])
            self._append_text_content(content_list, remaining_text[:next_pos])

            if next_token == "<TEST-IMAGE>":
                for _ in range(example_image_count):
                    content_list.append({"type": "image"})
            else:
                content_list.append({"type": "image"})

            remaining_text = remaining_text[next_pos + len(next_token) :]

        return content_list

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        messages = []
        if self.image_key in example:
            interleaved_content = self._build_interleaved_system_image_content(len(example[self.image_key]))
            if interleaved_content is not None:
                messages.append({"role": "user", "content": interleaved_content})
                return messages

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for _ in self.prompt_image_paths:
                content_list.append({"type": "image"})
            content_list.extend(self._build_prompt_image_content(prompt_str))
            messages.append({"role": "user", "content": content_list})
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            messages.append({"role": "user", "content": content_list})
        else:
            messages.append({"role": "user", "content": prompt_str})

        return messages

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = list(example[self.image_key])
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]
            images = self._get_interleaved_prompt_image_paths() + images

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            raw_images = list(example[self.image_key])
            images = list(example.pop(self.image_key))
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]
            images = self._get_interleaved_prompt_image_paths() + images

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
            if self.image_dir is not None and len(raw_images) != 0 and isinstance(raw_images[0], str):
                target_image_path = os.path.join(self.image_dir, raw_images[0])
                teacher_text_map = maybe_load_frozenlake_text_map(target_image_path)
                if teacher_text_map is not None:
                    example["teacher_text_map"] = teacher_text_map
                frozenlake_task_text = maybe_load_frozenlake_task_text(target_image_path)
                if frozenlake_task_text is not None:
                    example["frozenlake_task_text"] = frozenlake_task_text
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)

        if self.return_raw_chat:
            example["raw_prompt"] = messages

        if self.teacher_image_key and self.teacher_image_key in example:
            teacher_images_raw = example.pop(self.teacher_image_key)
            if teacher_images_raw:
                example["teacher_extra_images"] = [
                    process_image(img, self.min_pixels, self.max_pixels) for img in teacher_images_raw
                ]

        if self.teacher_text_key and self.teacher_text_key in example:
            teacher_text = example.pop(self.teacher_text_key)
            if teacher_text is not None:
                example["teacher_text"] = str(teacher_text)

        return example
