# FunASR API Server Dockerfile
# 单一镜像，同时支持 CPU 和 GPU 模式，通过环境变量在运行时切换。
# 默认 CPU 模式；GPU 模式设置 FUNASR_DEVICE=cuda FUNASR_ENGINE=vllm FUNASR_MODEL=fun-asr-nano。
# 镜像特性：
#   - uv 从 Gitee 定制版安装，PyPI 使用阿里云镜像源
#   - apt 使用阿里云镜像源
#   - 时区 Asia/Shanghai
#   - 模型缓存目录 MODELSCOPE_CACHE 可通过环境变量自定义
#   - 包含 vLLM 可选依赖，GPU 模式可直接启用

FROM python:3.10-slim

# Switch apt source to Aliyun mirror (China acceleration)
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null \
    || sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null \
    || true

# Install system dependencies
# gcc: required by Triton (vLLM CUDA kernel compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    tzdata \
    gcc \
    gnupg \
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
COPY pyproject.toml uv.lock README.md ./
COPY app/ ./app/

# Install all dependencies (including vLLM extra for GPU support)
RUN uv sync --frozen --no-dev --extra vllm --no-install-project

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

CMD ["uv", "run", "funasr-api"]
