import logging

from subgen.config import use_path_mapping, path_mapping_from, path_mapping_to


def path_mapping(fullpath: str) -> str:
    if use_path_mapping:
        logging.debug("Updated path: " + fullpath.replace(path_mapping_from, path_mapping_to))
        return fullpath.replace(path_mapping_from, path_mapping_to)
    return fullpath
