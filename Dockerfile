FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        poppler-utils \
        fonts-dejavu-core \
        procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        "fastapi" "uvicorn[standard]" "pydantic" \
        "pillow" "opencv-python-headless" "albumentations" \
        "numpy" "scipy" "pandas" "scikit-image" "shapely" "pyclipper" "sympy" \
        "onnxruntime-gpu" \
        "beautifulsoup4" "lxml" "ftfy" "loguru" "magika" "fast_langdetect" \
        "openpyxl" "python-docx" "python-pptx" "mammoth" "reportlab" \
        "pdfminer.six" "pdftext" "pypdf" "pypdfium2" "pylatexenc" \
        "tokenizers<0.22" "transformers<5" "tqdm" "pyyaml" "six" "omegaconf"

WORKDIR /app
