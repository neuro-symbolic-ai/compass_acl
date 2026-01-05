import json
import csv
import os
from typing import Optional, Dict, Any, List, Union
from pathlib import Path


def json_to_csv(
        input_path: str,
        output_path: str,
        tokenizer=None,
        include_tokenizer_indices: bool = True
) -> None:
    """
    Convert intervention JSON file to CSV format with specified headers.

    Args:
        input_path: Path to input JSON file (intervention results)
        output_path: Path to output CSV file
        tokenizer: Tokenizer for computing token indices (optional)
        include_tokenizer_indices: Whether to include correct_idx/incorrect_idx columns
    """

    # Load JSON data
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Handle different JSON structures
    if "corpus" in data:
        # Standard intervention JSON format
        results = data["corpus"]
    elif isinstance(data, list):
        # Direct list of results
        results = data
    else:
        raise ValueError(f"Unrecognized JSON format in {input_path}")

    if not results:
        print(f"No results found in {input_path}")
        return

    # Create output directory
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Define CSV headers
    headers = [
        "clean", "corrupted", "intervention_type", "target_role",
        "target_word", "replacement_word", "position", "sentence_idx",
        "semantic_frame"
    ]

    if include_tokenizer_indices:
        headers.extend(["correct_idx", "incorrect_idx"])

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for result in results:
            # Extract basic fields
            row = {
                "clean": result.get("original_sentence", ""),
                "corrupted": result.get("modified_sentence", ""),
                "intervention_type": result.get("intervention_type", ""),
                "target_role": result.get("target_role", ""),
                "target_word": result.get("target_word", ""),
                "replacement_word": result.get("replacement_word", ""),
                "position": result.get("position", ""),
                "sentence_idx": _extract_sentence_idx(result),
                "semantic_frame": _extract_semantic_frame(result)
            }

            # Add tokenizer indices if requested
            if include_tokenizer_indices:
                if "correct_token_idx" in result and "incorrect_token_idx" in result:
                    # Use pre-computed indices from JSON
                    row["correct_idx"] = result["correct_token_idx"]
                    row["incorrect_idx"] = result["incorrect_token_idx"]
                elif tokenizer:
                    # Compute indices using tokenizer
                    correct_idx, incorrect_idx = _compute_token_indices(
                        result, tokenizer
                    )
                    row["correct_idx"] = correct_idx
                    row["incorrect_idx"] = incorrect_idx
                else:
                    # No tokenizer available
                    row["correct_idx"] = ""
                    row["incorrect_idx"] = ""

            writer.writerow(row)

    print(f"Converted {len(results)} intervention results to {output_path}")


def _extract_sentence_idx(result: Dict[str, Any]) -> str:
    """Extract sentence index from various possible locations."""
    # Check metadata first
    if "metadata" in result and isinstance(result["metadata"], dict):
        if "sentence_idx" in result["metadata"]:
            return str(result["metadata"]["sentence_idx"])

    # Check direct field
    if "sentence_idx" in result:
        return str(result["sentence_idx"])

    return ""


def _extract_semantic_frame(result: Dict[str, Any]) -> str:
    """Extract semantic frame from various possible locations."""
    # Check metadata first
    if "metadata" in result and isinstance(result["metadata"], dict):
        if "semantic_frame" in result["metadata"]:
            return str(result["metadata"]["semantic_frame"])

    # Check direct field
    if "semantic_frame" in result:
        return str(result["semantic_frame"])

    # Check original metadata
    if "original_metadata" in result and isinstance(result["original_metadata"], dict):
        metadata = result["original_metadata"].get("metadata", {})
        if isinstance(metadata, dict) and "frame" in metadata:
            return str(metadata["frame"])

    return ""


def _compute_token_indices(result: Dict[str, Any], tokenizer) -> tuple:
    """Compute token indices using tokenizer."""
    target_word = result.get("target_word", "")
    replacement_word = result.get("replacement_word", "")

    if not target_word:
        return "", ""

    try:
        # Get token ID for target word
        correct_idx = tokenizer.encode(target_word, add_special_tokens=False)[0]

        # Handle elimination case (no replacement word)
        if replacement_word is None or replacement_word == "":
            incorrect_idx = -1
        else:
            incorrect_idx = tokenizer.encode(replacement_word, add_special_tokens=False)[0]

        return correct_idx, incorrect_idx

    except Exception as e:
        print(f"Error computing token indices for '{target_word}' -> '{replacement_word}': {e}")
        return "", ""


def batch_convert_json_to_csv(
        input_dir: str,
        output_dir: str,
        tokenizer=None,
        file_pattern: str = "*.json"
) -> None:
    """
    Convert multiple JSON files to CSV format.

    Args:
        input_dir: Directory containing JSON files
        output_dir: Directory to save CSV files
        tokenizer: Tokenizer for computing token indices
        file_pattern: Pattern for matching JSON files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Find JSON files
    json_files = list(input_path.glob(file_pattern))

    if not json_files:
        print(f"No JSON files found in {input_dir} matching pattern {file_pattern}")
        return

    print(f"Converting {len(json_files)} JSON files...")

    for json_file in json_files:
        # Create corresponding CSV filename
        csv_filename = json_file.stem + ".csv"
        csv_path = output_path / csv_filename

        try:
            json_to_csv(
                input_path=str(json_file),
                output_path=str(csv_path),
                tokenizer=tokenizer
            )
        except Exception as e:
            print(f"Error converting {json_file}: {e}")

    print(f"Batch conversion complete. CSV files saved to {output_dir}")


def load_and_convert(
        json_data: Union[str, Dict, List],
        output_path: str,
        tokenizer=None
) -> None:
    """
    Convert loaded JSON data (from file path, dict, or list) to CSV.

    Args:
        json_data: Either file path string, dict, or list of intervention results
        output_path: Path to output CSV file
        tokenizer: Tokenizer for computing token indices
    """

    if isinstance(json_data, str):
        # It's a file path
        json_to_csv(json_data, output_path, tokenizer)

    elif isinstance(json_data, dict):
        # It's a loaded JSON dict
        if "results" in json_data:
            results = json_data["results"]
        else:
            results = [json_data]  # Single result

        _write_results_to_csv(results, output_path, tokenizer)

    elif isinstance(json_data, list):
        # It's a list of results
        _write_results_to_csv(json_data, output_path, tokenizer)

    else:
        raise ValueError(f"Unsupported json_data type: {type(json_data)}")


def _write_results_to_csv(results: List[Dict], output_path: str, tokenizer=None):
    """Helper function to write results list to CSV."""
    # Create temporary JSON structure for consistency
    temp_data = {"results": results}

    # Write to temporary JSON file, then convert
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(temp_data, tmp, indent=2, default=str)
        tmp_path = tmp.name

    try:
        json_to_csv(tmp_path, output_path, tokenizer)
    finally:
        os.unlink(tmp_path)