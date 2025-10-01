#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from pathlib import Path
from PIL import Image
import torch
import gradio as gr

from ..environments.DescribeEarth import (
    DescribeEarthForConditionalGeneration,
    DescribeEarthProcessor,
)
from transformers import AutoProcessor
import open_clip
from vision_process import process_vision_info_dam

# ------------------  Utils  ------------------
def crop_aabb_bbox(image: Image.Image, pts: np.ndarray, T_large: int = 224, T_small: int = 112) -> Image.Image:
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
        "airplane", "airport", "runway", "helipad", "ship", "harbor", "bridge", "overpass",
        "railway", "railway_station", "train", "vehicle", "car", "truck", "bus", "van",
        "motorcycle", "bicycle", "roundabout", "parking_lot", "container_crane", "dock",
        "port", "toll_station", "service_area", "stadium", "soccer_field", "baseball_field",
        "basketball_court", "tennis_court", "ground_track_field", "swimming_pool", "golf_course",
        "playground", "amusement_park", "race_track", "storage_tank", "oil_tank", "chimney",
        "windmill", "powerplant", "solar_farm", "industrial_area", "oil_refinery", "factory",
        "warehouse", "construction_site", "residential_area", "apartment", "villa", "commercial_area",
        "building", "skyscraper", "church", "temple", "school", "hospital", "office_building",
        "stadium_roof", "shopping_mall", "market", "theater", "museum", "government_building",
        "prison", "military_base", "radar_station", "missile_launcher", "bunker", "air_defense_system",
        "warship", "submarine", "forest", "farmland", "meadow", "desert", "mountain", "hill",
        "river", "lake", "pond", "reservoir", "dam", "wetland", "beach", "coastline", "island",
        "peninsula", "glacier", "volcano", "highway", "street", "intersection", "crosswalk",
        "bridge_tunnel", "pipeline", "powerline", "communication_tower", "satellite_dish",
        "water_tower", "lighthouse", "fence", "wind_turbine_farm", "solar_panel_array",
        "billboard", "statue", "fountain", "monument", "cemetery", "greenhouse", "container",
        "crane", "excavator", "airship", "balloon", "kite",
    ]
    text = tokenizer(category).to(device)
    if clip_ckpt.exists():
        ckpt = torch.load(clip_ckpt, map_location=device)
        model.load_state_dict(ckpt)
    model.eval().to(device)
    return model, preprocess, category, text

def classify_clip(model, preprocess, category, text, focal: Image.Image, device="cuda") -> tuple[str, float]:
    clip_input = preprocess(focal).unsqueeze(0).to(device)
    with torch.no_grad(), torch.amp.autocast(device):
        img_feat = model.encode_image(clip_input)
        txt_feat = model.encode_text(text)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        probs = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1).cpu().numpy()[0]
    top_idx = int(np.argmax(probs))
    return category[top_idx], float(probs[top_idx])

def run_inference(model, tokenizer, processor,
                  image: Image.Image,
                  focal: Image.Image,
                  pts: np.ndarray,
                  category: str | None = None,
                  max_tok: int = 8192) -> str:
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
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "focal_crop", "focal_crop": focal},
        {"type": "text", "text": prompt},
    ]}]
    print(prompt)
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

# ------------------  Gradio UI ------------------
DEFAULT_MODEL_DIR = "../outputs/dam_cat_448_224_froz"
DEFAULT_CLIP_CKPT = "../RemoteCLIP/RemoteCLIP-RN50.pt"

click_history = []
def reset_clicks():
    click_history.clear()

def click_handler(img, evt: gr.SelectData,
                  model_dir: str,
                  clip_ckpt: str):
    if img is None:
        return None, "Please upload an image and click 4 points."
    click_history.append((evt.index[0], evt.index[1]))
    if len(click_history) < 4:
        return None, f"Clicked {len(click_history)}/4 points."

    pts = np.array(click_history, dtype=np.float32)
    image = Image.fromarray(img).convert("RGB")
    focal = crop_aabb_bbox(image, pts)

    # CLIP
    clip_model, clip_preprocess, categories, clip_text = load_remoteclip(Path(clip_ckpt))
    cat, conf = classify_clip(clip_model, clip_preprocess, categories, clip_text, focal)
    clip_category = cat if conf >= 0.8 else None

    # Qwen2.5-VL
    model, tokenizer, processor = load_model_processor(model_dir)
    answer = run_inference(model, tokenizer, processor, image, focal, pts, category=clip_category)

    reset_clicks()
    return focal, answer

with gr.Blocks(title="DescribeEarth 🌍 | Four-Click Object Description") as demo:
    gr.Markdown("# DescribeEarth 🌍  |  Four-Click Object Description")
    gr.Markdown("Upload an image → click **four points** in any order → get instant crop & description.")

    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(type="numpy", label="Input Image (click 4 points)", height=420)
            with gr.Group():
                model_dir = gr.Textbox(value=DEFAULT_MODEL_DIR, label="Qwen2.5-VL Model Path")
                clip_path = gr.Textbox(value=DEFAULT_CLIP_CKPT, label="RemoteCLIP Checkpoint Path")
        with gr.Column(scale=1):
            gr.Markdown("### 📌 How to use\n1. Upload an image  \n2. Click **4 points** (any order)  \n3. View crop & description instantly")
            focal_out = gr.Image(type="pil", label="Cropped Region (224×224)", height=280)
    ans_out = gr.Textbox(label="Description", lines=6, max_lines=12)

    img_in.select(click_handler,
                  inputs=[img_in, model_dir, clip_path],
                  outputs=[focal_out, ans_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False)
