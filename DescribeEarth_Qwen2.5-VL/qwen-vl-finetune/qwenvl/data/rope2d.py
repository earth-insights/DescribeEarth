import os
import copy
import json
import random
import logging
import re
import time
import math
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
import transformers


def get_rope_index_25(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    focal_crop_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image_token_id = 151655
    video_token_id = 151656
    focal_token_id = 151657          # <focal_crop>
    vision_start_token_id = 151652
    mrope_position_deltas = []

    if input_ids is not None and (
        image_grid_thw is not None
        or video_grid_thw is not None
        or focal_crop_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)

        position_ids = torch.ones(
            3,
            total_input_ids.shape[0],
            total_input_ids.shape[1],
            dtype=total_input_ids.dtype,
            device=total_input_ids.device,
        )

        image_idx = video_idx = focal_idx = 0
        attention_mask = attention_mask.to(total_input_ids.device)

        for i, ids in enumerate(total_input_ids):
            ids = ids[attention_mask[i] == 1]
            tokens = ids.tolist()

            vision_starts = (
                (ids == vision_start_token_id).nonzero(as_tuple=True)[0]
            )
            vision_toks = ids[vision_starts + 1]

            img_num = (vision_toks == image_token_id).sum().item()
            vid_num = (vision_toks == video_token_id).sum().item()
            foc_num = (vision_toks == focal_token_id).sum().item()

            llm_pos_parts = []
            st = 0
            remain = [img_num, vid_num, foc_num]

            for _ in range(img_num + vid_num + foc_num):
                ed_img = tokens.index(image_token_id, st) if remain[0] else len(tokens) + 1
                ed_vid = tokens.index(video_token_id, st) if remain[1] else len(tokens) + 1
                ed_foc = tokens.index(focal_token_id, st) if remain[2] else len(tokens) + 1

                # 取最小索引
                min_ed = min(ed_img, ed_vid, ed_foc)
                if min_ed == ed_img:
                    t, h, w = image_grid_thw[image_idx]
                    second_per_grid_t = 0
                    image_idx += 1
                    remain[0] -= 1
                    ed = ed_img
                elif min_ed == ed_vid:
                    t, h, w = video_grid_thw[video_idx]
                    second_per_grid_t = second_per_grid_ts[video_idx] if second_per_grid_ts is not None else 1.0
                    video_idx += 1
                    remain[1] -= 1
                    ed = ed_vid
                else:
                    t, h, w = focal_crop_grid_thw[focal_idx]
                    second_per_grid_t = 0
                    focal_idx += 1
                    remain[2] -= 1
                    ed = ed_foc

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st
                st_idx = llm_pos_parts[-1].max() + 1 if llm_pos_parts else 0

                llm_pos_parts.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                t_idx = (
                    torch.arange(llm_grid_t)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    .flatten()
                    * second_per_grid_t
                    * 25
                ).long()
                h_idx = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_idx = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_parts.append(
                    torch.stack([t_idx, h_idx, w_idx]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(tokens):
                st_idx = llm_pos_parts[-1].max() + 1 if llm_pos_parts else 0
                text_len = len(tokens) - st
                llm_pos_parts.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_parts, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max().item() + 1 - len(total_input_ids[i])
            )

        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=total_input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            )
            max_pos = position_ids.max(0)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_pos + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            
        return position_ids, mrope_position_deltas
    

def get_rope_index_2(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    focal_crop_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image_token_id = 151655
    video_token_id = 151656
    focal_token_id = 151657          # <focal_crop>
    vision_start_token_id = 151652
    mrope_position_deltas = []

    if input_ids is not None and (
        image_grid_thw is not None
        or video_grid_thw is not None
        or focal_crop_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)

        position_ids = torch.ones(
            3,
            total_input_ids.shape[0],
            total_input_ids.shape[1],
            dtype=total_input_ids.dtype,
            device=total_input_ids.device,
        )

        image_idx = video_idx = focal_idx = 0
        attention_mask = attention_mask.to(total_input_ids.device)

        for i, ids in enumerate(total_input_ids):
            ids = ids[attention_mask[i] == 1]
            tokens = ids.tolist()

            vision_starts = (
                (ids == vision_start_token_id).nonzero(as_tuple=True)[0]
            )
            vision_toks = ids[vision_starts + 1]

            img_num = (vision_toks == image_token_id).sum().item()
            vid_num = (vision_toks == video_token_id).sum().item()
            foc_num = (vision_toks == focal_token_id).sum().item()

            llm_pos_parts = []
            st = 0
            remain = [img_num, vid_num, foc_num]

            for _ in range(img_num + vid_num + foc_num):
                ed_img = tokens.index(image_token_id, st) if remain[0] else len(tokens) + 1
                ed_vid = tokens.index(video_token_id, st) if remain[1] else len(tokens) + 1
                ed_foc = tokens.index(focal_token_id, st) if remain[2] else len(tokens) + 1

                min_ed = min(ed_img, ed_vid, ed_foc)
                if min_ed == ed_img:
                    t, h, w = image_grid_thw[image_idx]
                    image_idx += 1
                    remain[0] -= 1
                    ed = ed_img
                elif min_ed == ed_vid:
                    t, h, w = video_grid_thw[video_idx]
                    video_idx += 1
                    remain[1] -= 1
                    ed = ed_vid
                else:
                    t, h, w = focal_crop_grid_thw[focal_idx]
                    focal_idx += 1
                    remain[2] -= 1
                    ed = ed_foc

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st
                st_idx = llm_pos_parts[-1].max() + 1 if llm_pos_parts else 0

                llm_pos_parts.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                t_idx = (
                    torch.arange(llm_grid_t)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    .flatten()
                )
                h_idx = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_idx = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_parts.append(
                    torch.stack([t_idx, h_idx, w_idx]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(tokens):
                st_idx = llm_pos_parts[-1].max() + 1 if llm_pos_parts else 0
                text_len = len(tokens) - st
                llm_pos_parts.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_parts, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max().item() + 1 - len(total_input_ids[i])
            )

        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=total_input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            )
            max_pos = position_ids.max(0)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_pos + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            
        return position_ids, mrope_position_deltas
    