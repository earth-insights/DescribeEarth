# -*- coding: utf-8 -*-
"""
This script analyzes model prediction results against a dataset.

It processes multiple JSON files, each containing evaluation scores from a
different model. It aggregates these scores based on three criteria derived
from the dataset's metadata:
1. Category: The general class of the data sample.
2. Difficulty: The difficulty level of the data sample.
3. Question Type: The specific type of each question within a sample.

The script calculates both the total scores and the score rates (score / max_score)
for each model across all criteria. The final results are compiled into an
Excel file with two separate sheets: "Total Scores" and "Score Rates".
"""

import os
import json
from collections import defaultdict
from typing import Dict, Any, List, Tuple

import pandas as pd

# --- Constants ---
# Directory containing model prediction results (JSON files).
PREDICT_JSON_DIR = ''
# Root directory of the dataset, containing metadata for each sample.
DATASET_ROOT_DIR = ''
# Path for the output Excel file.
EXCEL_OUT = ''

# --- Type Aliases for clarity ---
StatsDict = Dict[str, Dict[str, int]]
GroupedStats = Dict[str, StatsDict]


def load_json_file(file_path: str) -> Any:
    """
    Loads data from a JSON file with UTF-8 encoding.

    Args:
        file_path: The path to the JSON file.

    Returns:
        The loaded JSON data. Returns None if the file is not found or
        cannot be decoded.
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found at {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Error reading or decoding JSON from {file_path}: {e}")
        return None


def analyze_model_results(result_json_path: str, question_map: Dict[str, List[str]]) -> GroupedStats:
    """
    Analyzes a single model's result file and computes statistics.

    Aggregates scores by category, difficulty, and question type.

    Args:
        result_json_path: Path to the model's prediction result JSON file.
        question_map: A dictionary mapping instance IDs to a list of question types.

    Returns:
        A dictionary containing aggregated statistics for "Category",
        "Difficulty", and "QuestionType".
    """
    # Initialize dictionaries to hold statistics for different groups.
    # Structure: {group_key: {"score_sum": X, "sample_count": Y, "max_score_sum": Z}}
    cat_stats: StatsDict = defaultdict(lambda: {"score_sum": 0, "sample_count": 0, "max_score_sum": 0})
    diff_stats: StatsDict = defaultdict(lambda: {"score_sum": 0, "sample_count": 0, "max_score_sum": 0})
    ques_stats: StatsDict = defaultdict(lambda: {"score_sum": 0, "sample_count": 0, "max_score_sum": 0})

    # Load the model's prediction scores.
    model_data = load_json_file(result_json_path)
    if not model_data:
        return {}

    for instance_id, scores_list in model_data.items():
        # Construct the path to the sample's metadata file.
        sample_meta_path = os.path.join(DATASET_ROOT_DIR, str(instance_id), 'sample.json')
        
        # Load sample metadata.
        meta = load_json_file(sample_meta_path)
        if not meta:
            print(f"Warning: Skipping {instance_id} due to missing or invalid metadata file.")
            continue

        total_score = sum(scores_list)
        # Each question has a maximum score of 2.
        max_possible_score = len(scores_list) * 2

        # 1. Aggregate by Category
        # The category might be stored under 'category' or 'class_name'.
        category = meta.get('category') or meta.get('class_name')
        if category:
            cat_stats[category]["score_sum"] += total_score
            cat_stats[category]["max_score_sum"] += max_possible_score
            cat_stats[category]["sample_count"] += 1

        # 2. Aggregate by Difficulty
        if (difficulty := meta.get('difficulty')):
            diff_stats[difficulty]["score_sum"] += total_score
            diff_stats[difficulty]["max_score_sum"] += max_possible_score
            diff_stats[difficulty]["sample_count"] += 1

        # 3. Aggregate by Question Type
        if instance_id in question_map:
            question_types = question_map[instance_id]
            # Ensure the number of types matches the number of scores.
            if len(question_types) == len(scores_list):
                for q_type, q_score in zip(question_types, scores_list):
                    ques_stats[q_type]["score_sum"] += q_score
                    # Max score for a single question is 2.
                    ques_stats[q_type]["max_score_sum"] += 2
                    # The count here refers to individual questions, not samples.
                    ques_stats[q_type]["sample_count"] += 1
            else:
                print(f"Warning: Mismatch between number of scores and question types for {instance_id}.")

    return {"Category": cat_stats, "Difficulty": diff_stats, "QuestionType": ques_stats}


def main():
    """
    Main function to orchestrate the analysis and report generation.
    """
    # Load the global mapping of instance IDs to question types.
    question_map = load_json_file(os.path.join(DATASET_ROOT_DIR, 'question_class.json'))
    if not question_map:
        print("Error: Could not load question type map. Aborting.")
        return

    # Find all model result JSON files in the specified directory.
    result_files = sorted([f for f in os.listdir(PREDICT_JSON_DIR) if f.endswith('.json')])
    if not result_files:
        print(f"No JSON files found in {PREDICT_JSON_DIR}. Exiting.")
        return
        
    print(f"Found {len(result_files)} result files to process.")

    # Dictionaries to store flattened data for DataFrame creation.
    # The keys will be model names, and values will be dictionaries
    # mapping (Group, Key) tuples to a score or rate.
    table_scores = {}
    table_rates = {}

    # Process each result file.
    for result_filename in result_files:
        model_name = os.path.splitext(result_filename)[0]
        print(f"Processing {model_name}...")
        
        full_path = os.path.join(PREDICT_JSON_DIR, result_filename)
        stats = analyze_model_results(full_path, question_map)

        score_row = {}
        rate_row = {}

        # Flatten the nested statistics into rows for the final tables.
        for group_name, group_data in stats.items():
            for key, values in group_data.items():
                score = values["score_sum"]
                max_score = values["max_score_sum"]
                
                # Multi-level index for the DataFrame.
                multi_index_key = (group_name, key)
                
                score_row[multi_index_key] = score
                
                # Calculate the score rate, avoiding division by zero.
                rate = score / max_score if max_score > 0 else 0
                rate_row[multi_index_key] = rate

        table_scores[model_name] = score_row
        table_rates[model_name] = rate_row

    # Create DataFrames from the collected data.
    # 'orient=columns' is suitable when dict keys are column names.
    df_score = pd.DataFrame.from_dict(table_scores, orient='columns').fillna(0)
    df_score.index = pd.MultiIndex.from_tuples(df_score.index, names=['Group', 'Key'])

    df_rate = pd.DataFrame.from_dict(table_rates, orient='columns').fillna(0)
    df_rate.index = pd.MultiIndex.from_tuples(df_rate.index, names=['Group', 'Key'])

    # Write both DataFrames to an Excel file.
    try:
        with pd.ExcelWriter(EXCEL_OUT) as writer:
            df_score.to_excel(writer, sheet_name="Total Scores")
            df_rate.to_excel(writer, sheet_name="Score Rates")
        print(f"Successfully generated summary report: {EXCEL_OUT}")
    except Exception as e:
        print(f"Error writing to Excel file {EXCEL_OUT}: {e}")


if __name__ == "__main__":
    main()