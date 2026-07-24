# FunASR Unified API Server

基于 FunASR + FastAPI 的统一语音识别 API 服务，支持 vLLM 和 PyTorch AutoModel 双引擎，提供离线转写和实时流式识别接口。

## 特性

- **双引擎架构**：vLLM 高性能推理 (RTFx 340+) + PyTorch AutoModel 轻量部署
- **离线转写**：HTTP REST、OpenAI Whisper 兼容 API、WebSocket
- **实时流式**：WebSocket 流式识别，partial 实时预览
- **uv 项目管理**：快速依赖安装与锁定
- **Docker 部署**：基于 vLLM 官方镜像，通过 .env 环境变量切换 CPU/GPU 模式
- **国内镜像加速**：apt/PyPI 使用阿里云源，uv 从 Gitee 定制版安装

## 快速开始

### 安装依赖

```bash
# CPU 模式（默认）
uv sync

# GPU 模式（含 vLLM）
uv sync --extra vllm
```

### 启动服务

```bash
# CPU 模式 - SenseVoice
uv run funasr-api --model sensevoice --device cpu --port 8000

# GPU 模式 - vLLM + Fun-ASR-Nano
uv run funasr-api --engine vllm --model fun-asr-nano --device cuda --port 8000
```

服务启动后访问 http://localhost:8000/docs 查看交互式 API 文档。

## API 接口

### 1. POST /asr — FunASR 原生接口

```bash
curl -X POST http://localhost:8000/asr \
  -F "file=@audio.wav" -F "language=中文" -F "spk=true"
```

### 2. POST /v1/audio/transcriptions — OpenAI 兼容接口

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
result = client.audio.transcriptions.create(
    model="sensevoice",
    file=open("audio.wav", "rb"),
    response_format="verbose_json",
)
print(result.text)
```

### 3. WebSocket /ws — 离线 WebSocket 转写

发送完整音频后获取识别结果。

### 4. WebSocket /ws/realtime — 实时流式识别

逐帧发送音频，获取 partial 实时预览和最终结果。

## 配置

通过命令行参数或环境变量配置，所有环境变量详见 `.env.example`。
主要配置项：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `FUNASR_ENGINE` | auto | 引擎：auto (PyTorch) / vllm (GPU 高性能) |
| `FUNASR_MODEL` | sensevoice | 模型名称 |
| `FUNASR_DEVICE` | cpu | 设备：cpu / cuda |
| `FUNASR_PORT` | 8000 | 服务端口 |
| `MODELSCOPE_CACHE` | ./model_cache | 模型缓存目录（映射到容器） |
| `TZ` | Asia/Shanghai | 容器时区 |
| `FUNASR_ENABLE_SPK` | false | 是否启用说话人分离 |

## Docker 部署

使用单一镜像，通过 `.env` 文件控制运行模式，默认启用 GPU 透传。

```bash
# 1. 准备配置文件
cp .env.example .env

# 2. 按需修改 .env 中的配置
#    CPU 模式：FUNASR_DEVICE=cpu, FUNASR_ENGINE=auto, FUNASR_MODEL=sensevoice
#    GPU 模式：FUNASR_DEVICE=cuda, FUNASR_ENGINE=vllm, FUNASR_MODEL=fun-asr-nano

# 3. 构建并启动
docker compose up -d --build

# 4. 查看日志
docker compose logs -f

# 5. 健康检查
docker compose exec funasr-api python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"
```

### Docker 镜像说明

- 基础镜像：`vllm/vllm-openai:v0.19.0`（预装 CUDA/torch/vllm/triton 运行时环境）
- 依赖安装：`uv pip install --system`（安装到系统 Python，复用镜像预装包）
- 入口点：清空 vLLM 镜像默认 ENTRYPOINT，直接运行 `funasr-api` 控制台脚本
- PYTHONPATH：设置 `/app`，支持 docker-compose 卷挂载热更新
- uv 安装：从 Gitee 定制版安装（`gitee.com/wangnov/uv-custom`）
- 镜像源：apt 和 PyPI 均使用阿里云源
- 时区：`Asia/Shanghai`
- 模型缓存：通过 `MODELSCOPE_CACHE` 环境变量指定宿主机目录，映射到容器 `/root/.cache/modelscope`
- GPU 透传：通过 `deploy.resources` 配置 NVIDIA GPU，需安装 NVIDIA Container Toolkit

## 项目结构

```
app/
├── main.py              # FastAPI 入口
├── config.py            # 配置管理
├── schemas.py           # 请求/响应模型
├── audio_utils.py       # 音频处理工具
├── ws_session.py        # WebSocket 会话管理
├── engines/             # ASR 引擎抽象层
│   ├── base.py          # 抽象基类
│   ├── automodel_engine.py  # PyTorch AutoModel 引擎
│   └── vllm_engine.py   # vLLM 引擎
└── routers/             # API 路由
    ├── offline.py       # 离线转写
    ├── realtime.py     # 实时流式
    └── system.py        # 系统接口
```