#!/usr/bin/env python3
from __future__ import annotations
import os
import sys

import argparse
import numpy as np
from pathlib import Path

import torch
from transformers import AutoProcessor

from PIL import Image
from vision_process import process_vision_info_dam

import open_clip

sys.path.append('../DescribeEarth_Qwen2.5-VL/qwen-vl-finetune/qwenvl/train')
from DescribeEarth import (
    DescribeEarthForConditionalGeneration,
    DescribeEarthProcessor,
)


def crop_aabb_bbox(
    image: Image.Image,
    pts: np.ndarray,
    T_large: int = 224,
    T_small: int = 112,
) -> Image.Image:
    """Crop bbox region from image and resize to 224x224."""
    W, H = image.size
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    bw, bh = x_max - x_min, y_max - y_min

    if bw >= T_large or bh >= T_large:
        crop = image.crop((x_min, y_min, x_max, y_max))
    elif bw < T_small and bh < T_small:
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        half = T_small / 2
        x0, y0 = max(cx - half, 0), max(cy - half, 0)
        x0, y0 = min(x0, W - T_small), min(y0, H - T_small)
        crop = image.crop((x0, y0, x0 + T_small, y0 + T_small))
    else:
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        half = T_large / 2
        x0, y0 = max(cx - half, 0), max(cy - half, 0)
        x0, y0 = min(x0, W - T_large), min(y0, H - T_large)
        crop = image.crop((x0, y0, x0 + T_large, y0 + T_large))

    return crop.resize((224, 224), Image.LANCZOS)


def load_model_processor(model_dir: str):
    model = DescribeEarthForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoProcessor.from_pretrained(model_dir)
    processor = DescribeEarthProcessor.from_pretrained(model_dir)
    return model, tokenizer, processor


def load_remoteclip(clip_ckpt: Path, device: str = "cuda"):
    clip_name = "RN50"
    model, _, preprocess = open_clip.create_model_and_transforms(clip_name)
    tokenizer = open_clip.get_tokenizer(clip_name)

    category = [
        "airplane", "airport", "runway", "helipad",
        "ship", "harbor", "bridge", "overpass",
        "railway", "railway_station", "train",
        "vehicle", "car", "truck", "bus", "van",
        "motorcycle", "bicycle", "roundabout", "parking_lot",
        "container_crane", "dock", "port", "toll_station", "service_area",
        "stadium", "soccer_field", "baseball_field", "basketball_court",
        "tennis_court", "ground_track_field", "swimming_pool", "golf_course",
        "playground", "amusement_park", "race_track",
        "storage_tank", "oil_tank", "chimney", "windmill",
        "powerplant", "solar_farm", "industrial_area", "oil_refinery",
        "factory", "warehouse", "construction_site",
        "residential_area", "apartment", "villa", "commercial_area",
        "building", "skyscraper", "church", "temple",
        "school", "hospital", "office_building", "stadium_roof",
        "shopping_mall", "market", "theater", "museum",
        "government_building", "prison",
        "military_base", "radar_station", "missile_launcher",
        "bunker", "air_defense_system", "warship", "submarine",
        "forest", "farmland", "meadow", "desert",
        "mountain", "hill", "river", "lake",
        "pond", "reservoir", "dam", "wetland",
        "beach", "coastline", "island", "peninsula",
        "glacier", "volcano",
        "highway", "street", "intersection", "crosswalk",
        "bridge_tunnel", "pipeline", "powerline", "communication_tower",
        "satellite_dish", "water_tower", "lighthouse", "fence",
        "wind_turbine_farm", "solar_panel_array",
        "billboard", "statue", "fountain",
        "monument", "cemetery", "greenhouse",
        "container", "crane", "excavator",
        "airship", "balloon", "kite",
    ]
    text = tokenizer(category).to(device)

    if clip_ckpt.exists():
        ckpt = torch.load(clip_ckpt, map_location=device)
        _ = model.load_state_dict(ckpt)

    model.eval().to(device)
    return model, preprocess, category, text


def classify_clip(
    model,
    preprocess,
    category,
    text,
    focal: Image.Image,
    device="cuda",
) -> tuple[str, float]:
    clip_input = preprocess(focal).unsqueeze(0).to(device)
    with torch.no_grad(), torch.amp.autocast(device):
        img_feat = model.encode_image(clip_input)
        txt_feat = model.encode_text(text)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        probs = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1).cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    return category[top_idx], float(probs[top_idx])  # 0~1


def run_inference(
    model,
    tokenizer,
    processor,
    image: Image.Image,
    focal: Image.Image,
    pts: np.ndarray,
    category: str | None = None,
    max_tok: int = 8192,
) -> str:
    category_text = f"of category {category} " if category else ""
    prompt = (
        f"Please describe the object {category_text}in the bounding box in the original image <image>, "
        f"where the bounding box is defined by the coordinates: "
        f"(x_left_top: {pts[0][0]:.1f}, y_left_top: {pts[0][1]:.1f}, "
        f"x_right_top: {pts[1][0]:.1f}, y_right_top: {pts[1][1]:.1f}, "
        f"x_right_bottom: {pts[2][0]:.1f}, y_right_bottom: {pts[2][1]:.1f}, "
        f"x_left_bottom: {pts[3][0]:.1f}, y_left_bottom: {pts[3][1]:.1f}). "
        f"The corresponding cropped region is shown in the focal image <focal_crop>."
    )
    print(prompt)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "focal_crop", "focal_crop": focal},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, focal_inputs = process_vision_info_dam(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        focal_crop=focal_inputs,
        is_decode=False,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        gen_ids = model.generate(**inputs, max_new_tokens=max_tok, use_cache=False)

    gen_ids_trim = [oid[len(iid):] for iid, oid in zip(inputs.input_ids, gen_ids)]
    return processor.batch_decode(
        gen_ids_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to Qwen2.5-VL model")
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--bbox", type=float, nargs="+", required=True,
                        help="4 numbers (x1 y1 x2 y2) or 8 numbers (4 points)")
    parser.add_argument("--clip_ckpt", type=Path,
                        default=None,
                        help="Path to RemoteCLIP checkpoint")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")

    if len(args.bbox) == 4:
        x1, y1, x2, y2 = args.bbox
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    elif len(args.bbox) == 8:
        pts = np.array(args.bbox, dtype=float).reshape(-1, 2)
    else:
        raise ValueError("bbox must be 4 or 8 numbers")

    focal = crop_aabb_bbox(image, pts)

    clip_category = None
    if args.clip_ckpt:
        clip_model, clip_preprocess, categories, clip_text = load_remoteclip(args.clip_ckpt)
        cat, conf = classify_clip(clip_model, clip_preprocess, categories, clip_text, focal)
        if conf >= 0.5:  # confidence threshold 50%
            clip_category = cat

    model, tokenizer, processor = load_model_processor(args.model_dir)
    answer = run_inference(model, tokenizer, processor, image, focal, pts, category=clip_category)
    print(answer)


if __name__ == "__main__":
    main()
