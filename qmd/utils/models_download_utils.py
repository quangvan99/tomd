# Copyright (c) Opendatalab. All rights reserved.
import os

# Root of the tomd project; model paths are like "models/Layout/PP-DocLayoutV2"
# so root must be the parent of "models/", i.e. the tomd directory itself.
_LOCAL_MODELS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def auto_download_and_get_model_root_path(relative_path: str, repo_mode='pipeline') -> str:
    """
    Returns the local models root path.
    Original function downloaded from HuggingFace/ModelScope; replaced with local path.
    """
    return _LOCAL_MODELS_ROOT
