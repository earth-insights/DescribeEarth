import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
# from torchcodec.decoders import VideoDecoder
import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
    grid_thw_focal: List = [],          # 新增
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    idx_img   = 0
    idx_vid   = 0
    idx_focal = 0
    input_ids, targets = [], []

    for source in sources:
        if roles.get(source[0]["from"]) != roles["human"]:
            source = source[1:]

        input_id = tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
        target   = [IGNORE_INDEX] * len(input_id)

        for conv in source:
            role    = roles.get(conv.get("role") or conv["from"])
            content = conv.get("content") or conv["value"]

            def replace(tok, grid_list, idx):
                nonlocal content
                if tok not in content:
                    return idx
                parts = content.split(tok)
                new_parts = []
                for p in parts[:-1]:
                    new_parts.append(p)
                    rep = "<|vision_start|>" + "<|image_pad|>" * grid_list[idx] + "<|vision_end|>"
                    new_parts.append(rep)
                    idx += 1
                new_parts.append(parts[-1])
                content = "".join(new_parts)
                return idx

            idx_img   = replace("<image>",     grid_thw_image, idx_img)
            idx_vid   = replace("<video>",     grid_thw_video, idx_vid)
            idx_focal = replace("<focal_crop>", grid_thw_focal, idx_focal)

            enc = tokenizer.apply_chat_template([{"role": role, "content": content}])
            input_id += enc
            target   += [IGNORE_INDEX] * len(enc) if role in ["user", "system"] else enc

        assert len(input_id) == len(target)
        input_ids.append(input_id)
        targets.append(target)

    return dict(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        labels=torch.tensor(targets, dtype=torch.long),
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data["data_path"]
                else:
                    ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def process_video(self, video_file):
        decord_video = None
        decord_attempts = 0
        max_decord_attempts = 3
        while decord_attempts < max_decord_attempts:
            try:
                decord_video = self.video_decord(video_file)
                return decord_video
                if decord_video:
                    break
            except Exception as e:
                print(f"Decord attempt {decord_attempts + 1} failed: {e}")
                decord_attempts += 1

        torchcodec_video = None
        try:
            torchcodec_video = self.video_torchcodec(video_file)
            return torchcodec_video
        except Exception as e:
            print(f"torchcodec attempt failed: {e}")

    def video_decord(self, video_file):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
            return None
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def video_torchcodec(self, video_file):
        device = "cpu"  # or e.g. "cuda"
        decoder = VideoDecoder(video_file, device=device)
        total_frames = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
        video = frame_batch.data.cpu().numpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def process_video_frames(self, video, frame_idx, video_length):
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def get_data(self, sources) -> Dict[str, torch.Tensor]:
        grid_thw_img   = None
        grid_thw_vid   = None
        grid_thw_focal = None
        img_tensors    = None
        vid_tensors    = None
        focal_tensor   = None
        second_per_grid_ts = None

        # process image
        if "image" in sources[0]:
            img_files = sources[0]["image"]
            img_folder = sources[0]["data_path"]
            if not isinstance(img_files, list):
                img_files = [img_files]
            img_files = [os.path.join(img_folder, f) for f in img_files]
            tensors, grids = zip(*[self.process_image_unified(f) for f in img_files])
            img_tensors = torch.cat(tensors, dim=0)
            grid_thw_img = torch.stack(grids, dim=0)

        # process video
        if "video" in sources[0]:
            vid_files = sources[0]["video"]
            vid_folder = sources[0]["data_path"]
            if not isinstance(vid_files, list):
                vid_files = [vid_files]
            vid_files = [os.path.join(vid_folder, f) for f in vid_files]
            tensors, grids, ts = zip(*[self.process_video(f) for f in vid_files])
            vid_tensors = torch.cat(tensors, dim=0)
            grid_thw_vid = torch.stack(grids, dim=0)
            second_per_grid_ts = list(itertools.chain(*ts))

        # process focal_crop
        if "focal_crop" in sources[0]:
            focal_file = os.path.join(sources[0]["data_path"], sources[0]["focal_crop"])
            focal_tensor, focal_grid = self.process_image_unified(focal_file)
            grid_thw_focal = focal_grid.unsqueeze(0)

        # build grid_thw_* list
        grid_img_list   = [g.prod() // self.data_args.image_processor.merge_size**2
                        for g in (grid_thw_img  if grid_thw_img  is not None else [])]
        grid_vid_list   = [g.prod() // self.data_args.image_processor.merge_size**2
                        for g in (grid_thw_vid  if grid_thw_vid  is not None else [])]
        grid_focal_list = [g.prod() // self.data_args.image_processor.merge_size**2
                        for g in (grid_thw_focal if grid_thw_focal is not None else [])]

        # pre-process conversations
        chat_sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_img_list,
            grid_thw_video=grid_vid_list,
            grid_thw_focal=grid_focal_list,
        )

        pos_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=grid_thw_img,
            video_grid_thw=grid_thw_vid,
            second_per_grid_ts=second_per_grid_ts,
        )

        data_dict.update({
            "position_ids": pos_ids,
            "attention_mask": [data_dict["input_ids"].size(1)],
            "pixel_values": img_tensors,
            "image_grid_thw": grid_thw_img,
            "pixel_values_videos": vid_tensors,
            "video_grid_thw": grid_thw_vid,
            "focal_crop": focal_tensor.unsqueeze(0) if focal_tensor is not None else None,
            "focal_crop_grid_thw": grid_thw_focal,
        })
        return {k: v for k, v in data_dict.items() if v is not None}

    def _get_item(self, i) -> Dict[str, torch.Tensor]:

        sources = self.list_data_dict[i]

        if isinstance(sources, dict):
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            return self.get_data(sources)

        if isinstance(sources, list):
            data_list = []
            new_data_dict = {}
            for source in sources:
                if isinstance(i, int):
                    source = [source]
                assert (
                    len(source) == 1
                ), "Don't know why it is wrapped to a list"  # FIXME
                data_list.append(self.get_data(source))

            input_ids = torch.cat([d["input_ids"] for d in data_list], dim=1)
            labels = torch.cat([d["labels"] for d in data_list], dim=1)
            position_ids = torch.cat([d["position_ids"] for d in data_list], dim=2)
            attention_mask = [
                d["attention_mask"][0] for d in data_list if "attention_mask" in d
            ]
            new_data_dict = {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "attention_mask": attention_mask if attention_mask else None
            }
            
            if any("pixel_values" in d for d in data_list):
                new_data_dict.update({
                    "pixel_values": torch.cat([d["pixel_values"] for d in data_list if "pixel_values" in d], dim=0),
                    "image_grid_thw": torch.cat([d["image_grid_thw"] for d in data_list if "image_grid_thw" in d], dim=0)
                })
            
            if any("pixel_values_videos" in d for d in data_list):
                new_data_dict.update({
                    "pixel_values_videos": torch.cat([d["pixel_values_videos"] for d in data_list if "pixel_values_videos" in d], dim=0),
                    "video_grid_thw": torch.cat([d["video_grid_thw"] for d in data_list if "video_grid_thw" in d], dim=0)
                })
                
            if any("focal_crop" in d for d in data_list):
                new_data_dict.update({
                    "focal_crop": torch.cat(
                        [d["focal_crop"] for d in data_list if "focal_crop" in d],
                        dim=0
                    ),
                    "focal_crop_grid_thw": torch.cat(
                        [d["focal_crop_grid_thw"] for d in data_list if "focal_crop_grid_thw" in d],
                        dim=0
                    )
                })

            return new_data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class PackedDataCollatorForSupervisedDataset(object):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    input_ids, labels, position_ids, attention_mask = tuple(
        [instance[key] for instance in instances]
        for key in ("input_ids", "labels", "position_ids", "attention_mask")
    )
    attention_mask = list(
        itertools.chain(
            *(
                instance["attention_mask"]
                for instance in instances
                if "attention_mask" in instance
            )
        )
    )
    seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
    cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)

    input_ids = torch.cat(input_ids, dim=1)
    labels = torch.cat(labels, dim=1)
    position_ids = torch.cat(position_ids, dim=2)

    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=cumsum_seq_lens,
        position_ids=position_ids,
    )

    # collect images
    images = [
        instance["pixel_values"]
        for instance in instances
        if "pixel_values" in instance
    ]
    if images:
        batch["pixel_values"] = torch.cat(images, dim=0)
        batch["image_grid_thw"] = torch.cat(
            [inst["image_grid_thw"] for inst in instances if "image_grid_thw" in inst],
            dim=0,
        )
    else:
        batch["pixel_values"] = None
        batch["image_grid_thw"] = None

    # collect videos
    videos = [
        instance["pixel_values_videos"]
        for instance in instances
        if "pixel_values_videos" in instance
    ]
    if videos:
        batch["pixel_values_videos"] = torch.cat(videos, dim=0)
        batch["video_grid_thw"] = torch.cat(
            [inst["video_grid_thw"] for inst in instances if "video_grid_thw" in inst],
            dim=0,
        )
    else:
        batch["pixel_values_videos"] = None
        batch["video_grid_thw"] = None

    # collect focal crops
    focal_crops = [
        instance["focal_crop"]
        for instance in instances
        if "focal_crop" in instance
    ]
    if focal_crops:
        batch["focal_crop"] = torch.cat(focal_crops, dim=0)
        batch["focal_crop_grid_thw"] = torch.cat(
            [inst["focal_crop_grid_thw"] for inst in instances if "focal_crop_grid_thw" in inst],
            dim=0,
        )
    else:
        batch["focal_crop"] = None
        batch["focal_crop_grid_thw"] = None

    return batch


def make_supervised_data_module_packed(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = PackedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
