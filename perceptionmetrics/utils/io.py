from glob import glob
import json
import os
import re
from typing import List

from PIL import Image, UnidentifiedImageError
import yaml


def _ensure_existing_file(fname: str):
    if not os.path.exists(fname):
        raise FileNotFoundError(f"File not found: {fname}")
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"Expected a file path but got: {fname}")


def _ensure_parent_directory(fname: str):
    parent_dir = os.path.dirname(os.path.abspath(fname))
    if not os.path.exists(parent_dir):
        raise NotADirectoryError(f"Parent directory not found: {parent_dir}")
    if not os.path.isdir(parent_dir):
        raise NotADirectoryError(f"Parent path is not a directory: {parent_dir}")


def read_txt(fname: str) -> List[str]:
    """Read a .txt file line by line

    :param fname: .txt filename
    :type fname: str
    :return: List of lines found in the .txt file
    :rtype: List[str]
    """
    _ensure_existing_file(fname)
    with open(fname, "r") as f:
        data = f.read().split("\n")
    return [line for line in data if line]


def read_yaml(fname: str) -> dict:
    """Read a YAML file

    :param fname: YAML filename
    :type fname: str
    :return: Dictionary containing YAML file data
    :rtype: dict
    """
    _ensure_existing_file(fname)
    try:
        with open(fname, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML file '{fname}': {exc}") from exc
    return data


def read_json(fname: str) -> dict:
    """Read a JSON file

    :param fname: JSON filename
    :type fname: str
    :return: Dictionary containing JSON file data
    :rtype: dict
    """
    _ensure_existing_file(fname)
    try:
        with open(fname, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON file '{fname}': {exc}") from exc
    return data


def write_json(fname: str, data: dict):
    """Write a JSON file properly indented

    :param fname: Target JSON filename
    :type fname: str
    :param data: Dictionary containing data to be dumped as a JSON file
    :type data: dict
    """
    _ensure_parent_directory(fname)
    if os.path.exists(fname) and os.path.isdir(fname):
        raise NotADirectoryError(f"Target path is a directory: {fname}")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def get_image_mode(fname: str) -> str:
    """Given an image retrieve its color mode using PIL

    :param fname: Input image
    :type fname: str
    :return: PIL color image mode
    :rtype: str
    """
    _ensure_existing_file(fname)
    try:
        with Image.open(fname) as img:
            return img.mode
    except UnidentifiedImageError as exc:
        raise ValueError(f"Failed to identify image file '{fname}': {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to read image mode from '{fname}': {exc}") from exc


def extract_wildcard_matches(pattern: str) -> list:
    """Given a pattern with wildcards, extract the matches

    :param pattern: 'Globable' pattern with wildcards
    :type pattern: str
    :return: Matches found in the pattern
    :rtype: list
    """
    regex_pattern = re.escape(pattern).replace(r"\*", "(.*?)")
    regex = re.compile(regex_pattern)
    files = glob(pattern)
    matches = [regex.match(file).groups() for file in files if regex.match(file)]
    return matches
