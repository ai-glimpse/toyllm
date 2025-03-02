# Toy LLM

## 快速开始

### 环境配置

- 推荐使用 UV 安装依赖
  - 创建虚拟环境：`uv venv -p 3.12`
  - 激活虚拟环境：`source .venv/bin/activate`
  - 安装`toyllm`：`uv pip install toyllm`

### 拉取项目 & 下载模型文件

- 拉取本项目到本地
  - `git clone https://github.com/ai-glimpse/toyllm.git`
- 安装 LFS: [https://git-lfs.com](https://git-lfs.com)
- `git lfs install`
- 下载模型文件
  - 在`toyllm`的根目录下执行`git clone https://huggingface.co/MathewShen/toyllm-gpt2 models`
  - 或者直接从[https://huggingface.co/MathewShen/toyllm-gpt2/tree/main](https://huggingface.co/MathewShen/toyllm-gpt2/tree/main)下载模型文件，并放到`toyllm/models`目录下

### 使用


- GPT2 运行：`python toyllm/cli/run_gpt2.py`
  - `python toyllm/cli/run_gpt2.py --help` 查看参数信息

- Speculative Sampling GPT2: `python toyllm/cli/run_speculative_sampling_gpt2.py`
  - `python toyllm/cli/run_speculative_sampling_gpt2.py --help` 查看参数信息

## Acknowledgements

The project is highly inspired by the following projects:

- [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- [neelnanda-io/TransformerLens](https://github.com/neelnanda-io/TransformerLens)
