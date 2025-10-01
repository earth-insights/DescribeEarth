# -*- coding: utf-8 -*-
"""
This script uses the Qwen2.5-VL model to generate detailed descriptions for objects
detected in remote sensing images. It reads object information (class and bounding boxes)
from XML annotation files, processes the corresponding images in batches,
and writes the generated descriptions back into new XML files.

Workflow:
1.  Parse command-line arguments for model path, data folders, and processing options.
2.  Load the Qwen2.5-VL model and its associated processor.
3.  Iterate through images in the specified folder.
4.  For each image, read its corresponding XML annotation file to get object data.
5.  Construct prompts for each object, combining the image, object class, and polygon coordinates.
6.  Process objects in batches to generate descriptions using the model.
7.  Update the XML tree with the new descriptions.
8.  Save the updated XML to an output folder.
"""

# ==============================================================================
# 1. Imports and Setup
# ==============================================================================
import argparse
import logging
import os
import sys
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple

import torch
from modelscope import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm

# Add the project's utility directory to the system path to import custom modules.
sys.path.append('../DescribeEarth_Qwen2.5-VL')
from qwen_vl_utils import process_vision_info

# ==============================================================================
# 2. Global Configuration and Prompts
# ==============================================================================

# Define the system prompt that instructs the model on how to behave.
# This prompt sets the persona (professional remote sensing analyst) and defines
# the expected output format and style.
SYSTEM_PROMPT = """
You are a professional remote sensing analyst. Your task is to examine objects marked by polygon bounding boxes in high-resolution remote sensing images.

Write a clear, coherent paragraph that objectively describes each object’s visible characteristics. Avoid lists, guesses, or speculative language.

You must:
- Use direct and confident language based strictly on visual evidence;
- Describe the object's visual appearance, structure, orientation, and position;
- Mention its surrounding environment and any signs of activity;
- Identify its precise type or subcategory only if it can be determined visually with certainty.

Do not use uncertain phrases like "might be", "possibly", or "could be". Be concise but detailed, and only include facts that can be visually verified.
""".strip()


def setup_logging() -> None:
    """Configures the root logger for the script."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )


# ==============================================================================
# 3. Data Handling Functions
# ==============================================================================

def parse_and_get_objects(xml_path: str) -> Tuple[List[Dict[str, Any]], ET.ElementTree]:
    """
    Parses an XML annotation file to extract object information.

    Args:
        xml_path (str): The path to the XML file.

    Returns:
        Tuple[List[Dict[str, Any]], ET.ElementTree]:
            A tuple containing:
            - A list of dictionaries, where each dictionary represents an object
              with its class, polygon coordinates, and the original XML element.
            - The parsed XML tree object.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []

    for obj_elem in root.findall('object'):
        name = obj_elem.findtext('name', default='')
        robndbox = obj_elem.find('robndbox')

        if robndbox is not None:
            # Extract the four corner points of the rotated bounding box.
            points = [
                (float(robndbox.findtext('x_left_top')), float(robndbox.findtext('y_left_top'))),
                (float(robndbox.findtext('x_right_top')), float(robndbox.findtext('y_right_top'))),
                (float(robndbox.findtext('x_right_bottom')), float(robndbox.findtext('y_right_bottom'))),
                (float(robndbox.findtext('x_left_bottom')), float(robndbox.findtext('y_left_bottom')))
            ]
            objects.append({
                'class': name,
                'polygon': points,
                'xml_elem': obj_elem  # Keep a reference to the XML element for later updates.
            })

    return objects, tree


def build_messages(objects: List[Dict[str, Any]], image_path: str) -> List[List[Dict[str, Any]]]:
    """
    Constructs a batch of messages for the model based on object data.

    Args:
        objects (List[Dict[str, Any]]): A list of object dictionaries from `parse_and_get_objects`.
        image_path (str): Path to the corresponding image file.

    Returns:
        List[List[Dict[str, Any]]]: A list of message lists, one for each object.
    """
    messages_batch = []
    for obj in objects:
        # Format the polygon coordinates into a readable string.
        polygon_str = ', '.join([f'({int(x)}, {int(y)})' for x, y in obj['polygon']])
        user_prompt = f"Category: {obj['class']}\nPolygon bounding box: {polygon_str}"

        # Create the message structure required by the Qwen-VL model.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_prompt}
            ]}
        ]
        messages_batch.append(messages)
    return messages_batch


# ==============================================================================
# 4. Core Model Inference Functions
# ==============================================================================

def describe_objects(
    image_path: str,
    xml_path: str,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    device: str,
    max_batch_size: int = 10
) -> ET.ElementTree:
    """
    Generates descriptions for all objects in an image and updates the XML tree.

    Args:
        image_path (str): Path to the image file.
        xml_path (str): Path to the corresponding XML annotation file.
        model (Qwen2_5_VLForConditionalGeneration): The loaded Qwen-VL model.
        processor (AutoProcessor): The model's processor for handling inputs.
        device (str): The device to run inference on ('cuda' or 'cpu').
        max_batch_size (int): The maximum number of objects to process in a single batch.

    Returns:
        ET.ElementTree: The updated XML tree with new <description> tags.
    """
    objects, tree = parse_and_get_objects(xml_path)
    all_messages = build_messages(objects, image_path)

    try:
        # Process all objects in batches to manage memory usage.
        for i in range(0, len(all_messages), max_batch_size):
            batch_messages = all_messages[i:i + max_batch_size]

            # 1. Apply the chat template to format text inputs.
            text_inputs = [
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in batch_messages
            ]

            # 2. Extract image and video information using the custom utility.
            imgs, vids = process_vision_info(batch_messages)

            # 3. Use the processor to create a combined input tensor.
            inputs = processor(
                text=text_inputs,
                images=imgs,
                videos=vids,
                padding=True,
                return_tensors='pt'
            ).to(device)

            # 4. Generate text descriptions with the model.
            gen = model.generate(**inputs, max_new_tokens=1024)

            # 5. Trim the input tokens from the generated output to get only the response.
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen)]
            output_texts = processor.batch_decode(trimmed, skip_special_tokens=True)

            # 6. Append the generated descriptions to their corresponding XML elements.
            for obj, out_text in zip(objects[i:i + max_batch_size], output_texts):
                description_elem = ET.Element('description')
                description_elem.text = out_text.strip()
                obj['xml_elem'].append(description_elem)

    except Exception as e:
        # If an error occurs during processing, log it and add an error message
        # to the XML for all objects in the current image.
        logging.error(f"Failed to process {os.path.basename(image_path)}: {e}")
        for obj in objects:
            description_elem = ET.Element('description')
            description_elem.text = 'Description failed due to error.'
            obj['xml_elem'].append(description_elem)

    return tree


def process_dataset(
    image_folder: str,
    xml_folder: str,
    output_folder: str,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    device: str,
    max_batch_size: int
) -> None:
    """
    Processes an entire dataset of images and annotations.

    Args:
        image_folder (str): Path to the folder containing images.
        xml_folder (str): Path to the folder containing XML annotations.
        output_folder (str): Path to the folder where updated XML files will be saved.
        model: The loaded model.
        processor: The model's processor.
        device (str): The device for inference.
        max_batch_size (int): The batch size for processing objects.
    """
    os.makedirs(output_folder, exist_ok=True)
    img_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.tif'))])

    for img_file in tqdm(img_files, desc="Processing images"):
        base_name = os.path.splitext(img_file)[0]
        image_path = os.path.join(image_folder, img_file)
        xml_path = os.path.join(xml_folder, base_name + '.xml')
        output_path = os.path.join(output_folder, base_name + '.xml')

        # Skip if the annotation file is missing.
        if not os.path.exists(xml_path):
            logging.warning(f"Missing XML annotation for image {img_file}, skipping.")
            continue

        # Skip if the output file already exists to avoid re-processing.
        if os.path.exists(output_path):
            logging.info(f"Skipping {img_file}, output already exists.")
            continue

        logging.info(f"Processing {img_file}")
        updated_tree = describe_objects(image_path, xml_path, model, processor, device, max_batch_size)
        updated_tree.write(output_path, encoding='utf-8', xml_declaration=True)
        logging.info(f"Saved updated XML to {output_path}")


# ==============================================================================
# 5. Main Execution Block
# ==============================================================================

def main() -> None:
    """
    Main function to set up, parse arguments, and start the dataset processing.
    """
    setup_logging()
    parser = argparse.ArgumentParser(description="Generate object descriptions for remote sensing images using Qwen-VL.")
    parser.add_argument('--model_path', type=str, default='../Qwen/Qwen2.5-VL-32B-Instruct', help="Path to the pre-trained model directory.")
    parser.add_argument('--image_folder', type=str, default='./data/DIOR/JPEGImages-trainval', help="Path to the image folder.")
    parser.add_argument('--xml_folder', type=str, default='./data/DIOR/Annotations/Oriented_Bounding_Boxes', help="Path to the XML annotation folder.")
    parser.add_argument('--output_folder', type=str, default='./data/DIOR/Qwen_Description_test', help="Path to save the updated XML files.")
    parser.add_argument('--max_new_tokens', type=int, default=8192, help="Maximum number of new tokens to generate.")
    parser.add_argument('--max_batch_size', type=int, default=24, help="Maximum number of objects to process in one batch.")
    args = parser.parse_args()

    # Determine the computation device.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # Load the model with appropriate settings.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance on compatible GPUs.
        device_map='auto',           # Automatically distribute model across available GPUs.
    )

    # Configure the processor with pixel limits for handling high-resolution images.
    min_pixels = 768 * 28 * 28
    max_pixels = 1024 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        use_fast=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )

    # Set padding side to 'left' for batch generation. This is crucial as it ensures
    # that padding tokens are added to the left, which is required for many
    # autoregressive models during batched inference.
    processor.tokenizer.padding_side = 'left'

    # Start the main processing loop.
    process_dataset(
        image_folder=args.image_folder,
        xml_folder=args.xml_folder,
        output_folder=args.output_folder,
        model=model,
        processor=processor,
        device=device,
        max_batch_size=args.max_batch_size
    )

    logging.info("Processing complete.")


if __name__ == '__main__':
    main()