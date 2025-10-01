import os
import json
import xml.etree.ElementTree as ET
import argparse
import shutil
import yaml
import numpy as np
from PIL import Image, ImageFile

# Set environment to handle potentially truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ========== Constants (Module Level, Fixed Parameters for DescribeEarth Standard) ==========
# Target output size for the focal crop
FOCAL_SIZE = 224
# Threshold size differentiating small objects (DescribeEarth standard uses 112)
SMALL_THRESHOLD = 112

# Pre-calculated half sizes
HALF_FOCAL = FOCAL_SIZE / 2
HALF_SMALL = SMALL_THRESHOLD / 2

# ========== 1. Argument Parsing ==========
def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate DescribeEarth dataset with 224x224 focal crops.")
    parser.add_argument('--image_folder', type=str, required=True,
                        help="Path to the folder containing original images.")
    parser.add_argument('--description_folder', type=str, required=True,
                        help="Path to the folder containing XML annotation files.")
    parser.add_argument('--output_folder', type=str, required=True,
                        help="Path where the resulting dataset structure will be saved.")
    parser.add_argument('--dataset_name', type=str, default='MyDescribeEarthDataset',
                        help="Name of the dataset (used in default.yaml).")
    return parser.parse_args()

# ========== 2. Focal Cropping Logic ==========
def crop_aabb_bbox(image: Image.Image, pts: np.ndarray) -> Image.Image:
    """
    Crops the image based on the Axis-Aligned Bounding Box (AABB) derived from 
    the RBBX points (pts), according to DescribeEarth focal crop rules.

    The resulting crop is resized to FOCAL_SIZE x FOCAL_SIZE (224x224) without padding.
    """
    W, H = image.size
    
    # Calculate AABB from RBBX points
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

    bw = x_max - x_min
    bh = y_max - y_min

    crop_area = None

    # Case 1: Large object (bw >= FOCAL_SIZE or bh >= FOCAL_SIZE)
    if bw >= FOCAL_SIZE or bh >= FOCAL_SIZE:
        # Crop exactly to the AABB
        crop_area = (x_min, y_min, x_max, y_max)
        
    # Case 2: Small object (bw < SMALL_THRESHOLD and bh < SMALL_THRESHOLD)
    elif bw < SMALL_THRESHOLD and bh < SMALL_THRESHOLD:
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        
        crop_size = SMALL_THRESHOLD
        half_crop = HALF_SMALL
        
        # Calculate top-left coordinate (x0, y0), clamped
        x0 = max(cx - half_crop, 0)
        y0 = max(cy - half_crop, 0)
        
        # Adjust to ensure crop doesn't exceed image bounds
        x0 = min(x0, W - crop_size)
        y0 = min(y0, H - crop_size)
        
        crop_area = (x0, y0, x0 + crop_size, y0 + crop_size)
        
    # Case 3: Intermediate object (Default 224x224 centered crop)
    else:
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        
        crop_size = FOCAL_SIZE
        half_crop = HALF_FOCAL
        
        # Calculate top-left coordinate (x0, y0), clamped
        x0 = max(cx - half_crop, 0)
        y0 = max(cy - half_crop, 0)
        
        # Adjust to ensure crop doesn't exceed image bounds
        x0 = min(x0, W - crop_size)
        y0 = min(y0, H - crop_size)
        
        crop_area = (x0, y0, x0 + crop_size, y0 + crop_size)

    crop = image.crop(crop_area)

    # Resize the resulting crop to FOCAL_SIZE x FOCAL_SIZE
    return crop.resize((FOCAL_SIZE, FOCAL_SIZE), Image.LANCZOS)

# ========== 3. Main Processing Logic ==========
def main():
    args = parse_arguments()
    image_dir = args.image_folder
    annotation_dir = args.description_folder
    output_dir = args.output_folder
    dataset_name = args.dataset_name

    # Setup output directories
    output_images_dir = os.path.join(output_dir, 'images')
    output_focal_dir = os.path.join(output_dir, 'focal_crop')
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_focal_dir, exist_ok=True)

    dataset_entries = []

    # Iterate through all XML annotation files
    for annotation_filename in os.listdir(annotation_dir):
        if not annotation_filename.endswith('.xml'):
            continue
            
        xml_path = os.path.join(annotation_dir, annotation_filename)
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError:
            print(f"[ERROR] Failed to parse XML file: {xml_path}. Skipping.")
            continue

        image_filename_element = root.find('filename')
        if image_filename_element is None or not image_filename_element.text:
            print(f"[WARNING] XML {annotation_filename} missing 'filename' tag or value. Skipping.")
            continue
            
        image_filename = image_filename_element.text
        image_path = os.path.join(image_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"[WARNING] Image file {image_path} not found. Skipping {annotation_filename}.")
            continue

        # Copy original image
        output_image_path = os.path.join(output_images_dir, image_filename)
        shutil.copy(image_path, output_image_path)
        
        try:
            # Open and convert to RGB before processing
            original_image = Image.open(image_path).convert("RGB")
        except IOError as e:
            print(f"[ERROR] Failed to open image {image_path}: {e}. Skipping.")
            continue

        # Process all objects defined in the XML
        for obj_id, obj in enumerate(root.findall('.//object')):
            robndbox = obj.find('robndbox')
            description_element = obj.find('description')

            if robndbox is None or description_element is None or description_element.text is None:
                continue

            # Extract 4-point coordinates (RBBX format)
            try:
                pts = np.array([
                    [float(robndbox.find('x_left_top').text), float(robndbox.find('y_left_top').text)],
                    [float(robndbox.find('x_right_top').text), float(robndbox.find('y_right_top').text)],
                    [float(robndbox.find('x_right_bottom').text), float(robndbox.find('y_right_bottom').text)],
                    [float(robndbox.find('x_left_bottom').text), float(robndbox.find('y_left_bottom').text)],
                ], dtype=np.float32)
            except (AttributeError, ValueError):
                print(f"[WARNING] Coordinate error in {annotation_filename} object {obj_id}. Skipping.")
                continue

            description = description_element.text.strip()
            
            # Generate the focal crop
            crop_img = crop_aabb_bbox(original_image, pts)

            # Save focal crop with a unique identifier
            base_image_name = os.path.splitext(image_filename)[0]
            focal_filename = f"{base_image_name}_{obj_id:03d}.jpg"
            focal_crop_path = os.path.join(output_focal_dir, focal_filename)
            crop_img.save(focal_crop_path)
            
            # Format RBBX coordinates for the conversation prompt (for reproducibility)
            coords_prompt = (
                f"(x_left_top: {pts[0][0]:.1f}, y_left_top: {pts[0][1]:.1f}, "
                f"x_right_top: {pts[1][0]:.1f}, y_right_top: {pts[1][1]:.1f}, "
                f"x_right_bottom: {pts[2][0]:.1f}, y_right_bottom: {pts[2][1]:.1f}, "
                f"x_left_bottom: {pts[3][0]:.1f}, y_left_bottom: {pts[3][1]:.1f})"
            )
            
            # Construct the conversation entry
            entry = {
                "image": f"images/{image_filename}",
                "focal_crop": f"focal_crop/{focal_filename}",
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            f"Please describe the object in the bounding box in the original image <image>, "
                            f"where the bounding box is defined by the coordinates: {coords_prompt}. "
                            f"The corresponding cropped region is shown in the focal image <focal_crop>"
                        )
                    },
                    {
                        "from": "gpt",
                        "value": description
                    }
                ]
            }
            dataset_entries.append(entry)

    # --- Output Generation ---
    
    # 1. Save the final dataset JSON file
    dataset_json_path = os.path.join(output_dir, 'dataset.json')
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset_entries, f, indent=4)

    # 2. Generate default.yaml configuration file
    yaml_content = {
        dataset_name: {
            "_target_": "llava.data.LLaVADataset",
            "data_path": dataset_json_path,
            "media_dir": output_dir
        }
    }
    yaml_path = os.path.join(output_dir, 'default.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"\n[DONE] Dataset created successfully at {output_dir}")
    print(f"Total samples generated: {len(dataset_entries)}")

if __name__ == "__main__":
    main()