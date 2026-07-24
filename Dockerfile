# FunASR API Server Dockerfile
# 基于 vLLM 官方镜像，复用预装的 CUDA/torch/vllm/triton 运行时环境。
# 单一镜像，同时支持 CPU 和 GPU 模式，通过环境变量在运行时切换。
# 默认 CPU 模式；GPU 模式设置 FUNASR_DEVICE=cuda FUNASR_ENGINE=vllm FUNASR_MODEL=fun-asr-nano。
# 镜像特性：
#   - 基础镜像 vllm/vllm-openai:v0.19.0，预装 CUDA/torch/vllm/triton
#   - uv 从 Gitee 定制版安装，PyPI 使用阿里云镜像源
#   - apt 使用阿里云镜像源
#   - 时区 Asia/Shanghai
#   - 模型缓存目录 MODELSCOPE_CACHE 可通过环境变量自定义

FROM vllm/vllm-openai:v0.19.0

# Switch apt source to Aliyun mirror (China acceleration)
# vLLM image is Ubuntu-based; handle both classic and DEB822 formats
RUN sed -i 's|archive.ubuntu.com|mirrors.aliyun.com|g; s|security.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null \
    || true
RUN sed -i 's|archive.ubuntu.com|mirrors.aliyun.com|g; s|security.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null \
    || true

# Install additional system dependencies (CUDA/torch/triton already in base image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set timezone
ENV TZ=Asia/Shanghai

# Install uv from Gitee custom mirror (works in China without GitHub access)
# UV_INDEX_URL: use Aliyun PyPI mirror for faster downloads
ENV UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
RUN curl -LsSf https://gitee.com/wangnov/uv-custom/releases/download/latest/uv-installer-custom.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY app/ ./app/

# Install project dependencies into system Python
# Pre-installed packages (torch, vllm, triton, numpy) are kept; only missing deps are installed
RUN uv pip install --system ".[vllm]"

# Ensure source code takes precedence (supports docker-compose volume mount for development)
ENV PYTHONPATH=/app

# Default environment (CPU mode; override at runtime for GPU via .env)
# GPU mode: FUNASR_DEVICE=cuda, FUNASR_ENGINE=vllm, FUNASR_MODEL=fun-asr-nano
ENV FUNASR_HOST=0.0.0.0 \
    FUNASR_PORT=8000 \
    FUNASR_DEVICE=cpu \
    FUNASR_ENGINE=auto \
    FUNASR_MODEL=sensevoice \
    MODELSCOPE_CACHE=/root/.cache/modelscope

EXPOSE ${FUNASR_PORT:-8000}

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request, os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"FUNASR_PORT\", \"8000\")}/health')"

# Clear vLLM image's default ENTRYPOINT (vllm serve) so our CMD runs directly
ENTRYPOINT []
CMD ["funasr-api"]
