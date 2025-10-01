#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from openai import OpenAI
from tqdm import tqdm

# --- Dependencies for Local Model ---
import torch
from transformers import AutoProcessor

sys.path.append('../DescribeEarth_Qwen2.5-VL/qwen-vl-finetune/qwenvl/train')
from DescribeEarth import (
    DescribeEarthForConditionalGeneration,
    DescribeEarthProcessor,
)
from vision_process import process_vision_info_dam as pvd


# ==============================================================================
# ==                           DESCRIPTION GENERATORS                         ==
# ==============================================================================

class DescriptionGenerator(ABC):
    """Abstract base class for all description generators."""

    @abstractmethod
    def generate(self, sample_data: Dict[str, Any], folder_path: Path) -> str:
        """
        Generates a description for a given sample.

        Args:
            sample_data: The dictionary loaded from 'sample.json'.
            folder_path: The path to the directory containing the sample assets.

        Returns:
            The generated description as a string.
        """
        pass


class APIGenerator(DescriptionGenerator):
    """
    Generates descriptions using a remote OpenAI-compatible API.
    """
    def __init__(self, api_key: str, model_name: str, api_url: str,
                 max_retries: int = 3, retry_delay: float = 2.0):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = api_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @staticmethod
    def _encode_image_base64(filepath: Union[str, Path]) -> str:
        """Reads a local image and encodes it into a base64 string."""
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _clean_prompt(prompt: str) -> str:
        """Removes specific phrases from the prompt for the vision model."""
        # Remove the focal crop phrase
        cleaned = re.sub(
            r"The corresponding cropped region is shown in the focal image.*?<focal_crop>\s*",
            "", prompt, flags=re.IGNORECASE | re.DOTALL
        )
        # Remove the word 'original' which might confuse the model
        cleaned = re.sub(r"\boriginal\b", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def generate(self, sample_data: Dict[str, Any], folder_path: Path) -> str:
        """
        Calls the vision API to generate an image description.
        This implementation handles one image.
        """
        image_path = folder_path / sample_data["image"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found in {folder_path}")

        prompt = sample_data["conversations"][0]["value"]
        cleaned_prompt = self._clean_prompt(prompt)
        img_base64 = self._encode_image_base64(image_path)

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": cleaned_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                        },
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(self.api_url, headers=headers, json=payload, timeout=90)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"⚠️ API call failed ({attempt}/{self.max_retries}): {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    return "" # Return empty string on final failure
        return ""


class LocalModelGenerator(DescriptionGenerator):
    """
    Generates descriptions using a local fine-tuned Qwen-VL model.
    This class is a singleton based on the model directory path.
    """
    _instances: Dict[str, "LocalModelGenerator"] = {}
    MAX_TOKENS = 8192

    def __new__(cls, *args, **kwargs) -> "LocalModelGenerator":
        """Singleton implementation based on model_dir."""
        model_dir = kwargs.get("model_dir", "")
        if not model_dir:
            raise ValueError("model_dir must be provided for LocalModelGenerator")

        model_dir_str = str(Path(model_dir).expanduser().resolve())
        if model_dir_str not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[model_dir_str] = instance
        return cls._instances[model_dir_str]

    def __init__(self, *, model_dir: str):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.model_dir = str(Path(model_dir).expanduser().resolve())
        if not Path(self.model_dir).is_dir():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        print(f"Loading local model from {self.model_dir}...")
        self.model = DescribeEarthForConditionalGeneration.from_pretrained(
            self.model_dir, torch_dtype="auto", device_map="auto"
        )
        self.tokenizer = AutoProcessor.from_pretrained(self.model_dir)
        self.processor = DescribeEarthProcessor.from_pretrained(self.model_dir)
        print("✅ Local model loaded successfully.")

        self._initialized = True

    def generate(self, sample_data: Dict[str, Any], folder_path: Path) -> str:
        """
        Generates a description using the local model, handling both a main
        image and a focal crop.
        """
        image_path = folder_path / sample_data["image"]
        crop_path = folder_path / sample_data["focal_crop"]

        if not image_path.exists() or not crop_path.exists():
            raise FileNotFoundError(f"Missing image or focal_crop in {folder_path}")

        prompt = sample_data["conversations"][0]["value"]
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "focal_crop", "focal_crop": str(crop_path)},
                {"type": "text", "text": prompt},
            ],
        }]

        # Process visual and text inputs for the model
        image_inputs, video_inputs, focal_inputs = pvd(messages)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            focal_crop=focal_inputs,
            is_decode=False,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate text description
        with torch.inference_mode():
            gen_ids = self.model.generate(
                **inputs, max_new_tokens=self.MAX_TOKENS, use_cache=False
            )

        # Decode the generated token IDs
        gen_ids_trim = [gid[len(iid):] for iid, gid in zip(inputs.input_ids, gen_ids)]
        result = self.processor.batch_decode(
            gen_ids_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return result


# ==============================================================================
# ==                                GPT SCORER                                ==
# ==============================================================================

class GPTSCorer:
    """
    Scores a given description against a QA list using a GPT model.
    """
    SYSTEM_PROMPT = (
        "You are a senior expert in remote-sensing image interpretation.\n\n"
        "INPUT FORMAT:\n"
        "Description: <description>\n"
        "QA list:\n"
        "1. <question> → <choice1>(score1) / <choice2>(score2) / ...\n"
        "...\n\n"
        "TASK:\n"
        "For every question, select the single choice whose label best matches the description. "
        "Return ONLY a Python list of the chosen scores in the same order, e.g. [0.5, 1]. "
        "No explanations, no extra characters."
    )

    def __init__(self, api_key: str, model_name: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def _build_prompt(self, description: str, qa_list: List[Dict[str, Any]]) -> str:
        """Constructs the full prompt for the GPT scorer."""
        lines = [f"Description: {description}\n", "QA list:"]
        for i, qa in enumerate(qa_list, 1):
            choices_str = " / ".join(f"{c[0]}({c[1]})" for c in qa["choices"])
            lines.append(f"{i}. {qa['question']} → {choices_str}")
        return "\n".join(lines)

    def _parse_scores(self, output: str, num_questions: int) -> List[Union[float, None]]:
        """Extracts the list of scores from the model's raw output."""
        match = re.search(r"\[.*?\]", output, re.DOTALL)
        if not match:
            print(f"⚠️  Could not parse scores from output: {output}")
            return [None] * num_questions
        try:
            scores = json.loads(match.group(0))
            if isinstance(scores, list) and len(scores) == num_questions:
                # Ensure all elements are numbers or None
                return [s if isinstance(s, (int, float)) else None for s in scores]
            else:
                print(f"⚠️  Parsed list has incorrect length or type: {scores}")
        except json.JSONDecodeError:
            print(f"⚠️  JSON decoding failed for: {match.group(0)}")

        return [None] * num_questions

    def score(self, description: str, qa_list: List[Dict[str, Any]]) -> List[Optional[float]]:
        """
        Generates and parses scores for a description.

        Args:
            description: The model-generated description.
            qa_list: The list of questions and choices.

        Returns:
            A list of scores, with None for any parsing failures.
        """
        if not description:
            print("⚠️  Description is empty, cannot score.")
            return [None] * len(qa_list)
        
        num_questions = len(qa_list)
        prompt = self._build_prompt(description, qa_list)
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
            raw_output = completion.choices[0].message.content.strip()
            return self._parse_scores(raw_output, num_questions)
        except Exception as e:
            print(f"❌ Error during GPT scoring API call: {e}")
            return [None] * num_questions


# ==============================================================================
# ==                                MAIN WORKFLOW                             ==
# ==============================================================================

def load_qa(qa_file: str) -> Dict[str, List[Dict[str, Any]]]:
    """Loads the QA JSON file."""
    return json.loads(Path(qa_file).read_text(encoding="utf-8"))


def main():
    """Main function to run the evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Unified script to evaluate remote sensing descriptions.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- Core Arguments ---
    parser.add_argument("qa_file", help="Path to the QA JSON file.")
    parser.add_argument("root", help="Root directory containing <id>/sample.json folders.")
    parser.add_argument("-o", "--output_file", default="scores.json", help="Path for the output JSON file.")
    parser.add_argument(
        "--generator",
        required=True,
        choices=['api', 'local'],
        help="Choose the description generator type:\n"
             "  'api': Use a remote vision API (e.g., GPT-4o).\n"
             "  'local': Use a local model (e.g., Qwen-VL)."
    )

    # --- API Generator Arguments ---
    api_group = parser.add_argument_group('API Generator Options (for --generator=api)')
    api_group.add_argument("--api-key", default=os.environ.get("DESC_API_KEY"), help="API key for description generation. Reads from DESC_API_KEY env var.")
    api_group.add_argument("--api-model", default="gpt-4o", help="Model name for description generation.")
    api_group.add_argument("--api-url", default="https://api.openai.com/v1/chat/completions", help="API endpoint URL for description generation.")

    # --- Local Generator Arguments ---
    local_group = parser.add_argument_group('Local Generator Options (for --generator=local)')
    local_group.add_argument("--model-dir", help="Directory with tokenizer and model weights for the local generator.")

    # --- Scorer Arguments ---
    scorer_group = parser.add_argument_group('GPT Scorer Options')
    scorer_group.add_argument("--scorer-api-key", default=os.environ.get("SCORER_API_KEY"), help="API key for the GPT scorer. Reads from SCORER_API_KEY env var.")
    scorer_group.add_argument("--scorer-model", default="gpt-4.1", help="Model name for the GPT scorer.")
    scorer_group.add_argument("--scorer-base-url", default="https://api.openai.com/v1", help="Base URL for the GPT scorer API.")
    
    args = parser.parse_args()

    # --- Initialize Generator ---
    generator: DescriptionGenerator
    if args.generator == 'api':
        if not args.api_key:
            parser.error("API key is required for the 'api' generator. Provide --api-key or set DESC_API_KEY.")
        generator = APIGenerator(
            api_key=args.api_key,
            model_name=args.api_model,
            api_url=args.api_url
        )
    elif args.generator == 'local':
        if not args.model_dir:
            parser.error("Model directory is required for the 'local' generator. Provide --model-dir.")
        generator = LocalModelGenerator(model_dir=args.model_dir)
    else:
        # This case should not be reached due to argparse `choices`
        raise ValueError(f"Invalid generator type: {args.generator}")

    # --- Initialize Scorer ---
    if not args.scorer_api_key:
        parser.error("API key is required for the scorer. Provide --scorer-api-key or set SCORER_API_KEY.")
    scorer = GPTSCorer(
        api_key=args.scorer_api_key,
        model_name=args.scorer_model,
        base_url=args.scorer_base_url
    )

    # --- Run Evaluation Loop ---
    qa_data = load_qa(args.qa_file)
    root_dir = Path(args.root).expanduser().resolve()
    all_scores = {}

    for path_key, qa_list in tqdm(qa_data.items(), desc="Evaluating Samples"):
        folder = root_dir / path_key
        try:
            # Step 1: Generate description
            sample_content = json.loads((folder / "sample.json").read_text(encoding="utf-8"))
            description = generator.generate(
                sample_data=sample_content,
                folder_path=folder
            )
            
            # Step 2: Score the description
            score_list = scorer.score(description, qa_list)
            all_scores[path_key] = score_list

        except FileNotFoundError as e:
            print(f"\nWarning: Skipping {path_key} due to missing file: {e}")
            all_scores[path_key] = None
        except Exception as e:
            print(f"\nError: An unexpected error occurred while processing {path_key}: {e}")
            all_scores[path_key] = None

    # --- Save Results ---
    output_path = Path(args.output_file)
    output_path.write_text(json.dumps(all_scores, ensure_ascii=False, indent=2))
    print(f"\n✅ Done! Scores saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()