import os
import yaml
from pathlib import Path
from typing import Any, Dict
from src.logger import logger
from src.exception import CustomException
import sys
import json
from ensure import ensure_annotations
import joblib

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> Dict:
    """Reads a YAML file and returns its contents as a dictionary.
    
    Args:
        path_to_yaml (Path): Path to the YAML file.
    Returns:
        Dict: Contents of the YAML file as a dictionary.
    Raises:
        CustomException: If there is an error reading the file.
    """
    try:
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
        return content
    except Exception as e:
        raise CustomException(e, sys) from e

@ensure_annotations
def write_yaml(path_to_yaml: Path, content: Dict) -> None:
    """Writes a dictionary to a YAML file.
    
    Args:
        path_to_yaml (Path): Path to the YAML file.
        content (Dict): Dictionary to write to the file.
    Raises:
        CustomException: If there is an error writing the file.
    """
    try:
        with open(path_to_yaml, 'w') as yaml_file:
            yaml.safe_dump(content, yaml_file)
    except Exception as e:
        raise CustomException(e, sys) from e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True) -> None:
    """Creates directories if they do not exist.
    
    Args:
        path_to_directories (list): List of directory paths to create.
        verbose (bool): If True, logs the creation of directories.
    Raises:
        CustomException: If there is an error creating the directories.
    """
    try:
        for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"Directory created at: {path}")
    except Exception as e:
        raise CustomException(e, sys) from e
    
@ensure_annotations
def save_json(path: Path, data: Dict) -> None:
    """Saves a dictionary to a JSON file.
    
    Args:
        path (Path): Path to the JSON file.
        data (Dict): Dictionary to save.
    Raises:
        CustomException: If there is an error saving the file.
    """
    try:
        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    except Exception as e:
        raise CustomException(e, sys) from e
    
@ensure_annotations
def load_json(path: Path) -> Dict:
    """Loads a dictionary from a JSON file.
    
    Args:
        path (Path): Path to the JSON file.
    Returns:
        Dict: Dictionary loaded from the file.
    Raises:
        CustomException: If there is an error loading the file.
    """
    try:
        with open(path, 'r') as json_file:
            data = json.load(json_file)
        return data
    except Exception as e:
        raise CustomException(e, sys) from e

@ensure_annotations
def save_object(path: Path, obj: Any) -> None:
    """Saves a Python object to a file using joblib.
    
    Args:
        path (Path): Path to the file where the object will be saved.
        obj (Any): Python object to save.
    Raises:
        CustomException: If there is an error saving the object.
    """
    try:
        joblib.dump(obj, path)
        logger.info(f"Object saved at: {path}")
    except Exception as e:
        raise CustomException(e, sys) from e

@ensure_annotations
def load_object(path: Path) -> Any:
    """Loads a Python object from a file using joblib.
    
    Args:
        path (Path): Path to the file from which the object will be loaded.
    Returns:
        Any: Loaded Python object.
    Raises:
        CustomException: If there is an error loading the object.
    """
    try:
        obj = joblib.load(path)
        logger.info(f"Object loaded from: {path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys) from e

@ensure_annotations
def get_size(path: Path) -> str:
    """Returns the size of a file in KB.
    
    Args:
        path (Path): Path to the file.
    Returns:
        str: Size of the file in KB.
    Raises:
        CustomException: If there is an error getting the file size.
    """
    try:
        size_in_kb = round(os.path.getsize(path) / 1024, 2)
        return f"{size_in_kb} KB"
    except Exception as e:
        raise CustomException(e, sys) from e
    
@ensure_annotations
def copy_file(source: Path, destination: Path) -> None:
    """Copies a file from source to destination.
    
    Args:
        source (Path): Path to the source file.
        destination (Path): Path to the destination file.
    Raises:
        CustomException: If there is an error copying the file.
    """
    try:
        from shutil import copy2
        copy2(source, destination)
        logger.info(f"File copied from {source} to {destination}")
    except Exception as e:
        raise CustomException(e, sys) from e

@ensure_annotations
def list_files(directory: Path, extension: str = None) -> list:
    """Lists all files in a directory, optionally filtering by extension.
    
    Args:
        directory (Path): Path to the directory.
        extension (str, optional): File extension to filter by. Defaults to None.
    Returns:
        list: List of file paths.
    Raises:
        CustomException: If there is an error listing the files.
    """
    try:
        files = []
        for item in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, item)):
                if extension:
                    if item.endswith(extension):
                        files.append(os.path.join(directory, item))
                else:
                    files.append(os.path.join(directory, item))
        return files
    except Exception as e:
        raise CustomException(e, sys) from e
    
@ensure_annotations
def read_file_lines(path: Path) -> list:
    """Reads all lines from a text file and returns them as a list.
    
    Args:
        path (Path): Path to the text file.
    Returns:
        list: List of lines from the file.
    Raises:
        CustomException: If there is an error reading the file.
    """
    try:
        with open(path, 'r') as file:
            lines = file.readlines()
        return [line.strip() for line in lines]
    except Exception as e:
        raise CustomException(e, sys) from e
    
@ensure_annotations
def write_file_lines(path: Path, lines: list) -> None:
    """Writes a list of lines to a text file.
    
    Args:
        path (Path): Path to the text file.
        lines (list): List of lines to write to the file.
    Raises:
        CustomException: If there is an error writing the file.
    """
    try:
        with open(path, 'w') as file:
            for line in lines:
                file.write(f"{line}\n")
        logger.info(f"Lines written to file at: {path}")
    except Exception as e:
        raise CustomException(e, sys) from e

@ensure_annotations
def append_file_lines(path: Path, lines: list) -> None:
    """Appends a list of lines to a text file.
    
    Args:
        path (Path): Path to the text file.
        lines (list): List of lines to append to the file.
    Raises:
        CustomException: If there is an error appending to the file.
    """
    try:
        with open(path, 'a') as file:
            for line in lines:
                file.write(f"{line}\n")
        logger.info(f"Lines appended to file at: {path}")
    except Exception as e:
        raise CustomException(e, sys) from e

@ensure_annotations
def file_exists(path: Path) -> bool:
    """Checks if a file exists at the given path.
    
    Args:
        path (Path): Path to the file.
    Returns:
        bool: True if the file exists, False otherwise.
    Raises:
        CustomException: If there is an error checking the file.
    """
    try:
        return os.path.isfile(path)
    except Exception as e:
        raise CustomException(e, sys) from e