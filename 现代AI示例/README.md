# 现代 AI 示例

《深入理解AI和Agent：从线性代数开始的自学笔记》的配套代码。所有模块共用本目录内唯一的 `.venv` 虚拟环境。

## 环境与运行

~~~powershell
# 在现代AI示例目录执行，不向系统 Python 安装依赖。
uv sync --frozen --all-extras --python 3.12
.\.venv\Scripts\python.exe -B -m modern_ai math
.\.venv\Scripts\python.exe -B -m modern_ai learning
.\.venv\Scripts\python.exe -B -m modern_ai networks
.\.venv\Scripts\python.exe -B -m modern_ai language
.\.venv\Scripts\python.exe -B -m modern_ai retrieval
~~~

[`pyproject.toml`](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/pyproject.toml)（代码包内：`现代AI示例/pyproject.toml`）声明依赖，[`uv.lock`](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/uv.lock)（代码包内：`现代AI示例/uv.lock`）锁定版本。源码直接从当前目录读取运行。`-B` 参数用于关闭字节码缓存；测试数据库与故障实验产生的临时目录会在单次实验结束后自动清理。

## 模块与篇章

| 篇章 | 主要源码 | 内容 |
| --- | --- | --- |
| 第一篇 | [mathlab.py](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/modern_ai/mathlab.py)（代码包内：`现代AI示例/modern_ai/mathlab.py`） | 向量、变换、投影、压缩与误差 |
| 第二篇 | [learning.py](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/modern_ai/learning.py)（代码包内：`现代AI示例/modern_ai/learning.py`） | 回归、分类、划分与评价 |
| 第三篇 | [networks.py](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/modern_ai/networks.py)（代码包内：`现代AI示例/modern_ai/networks.py`） | 梯度、卷积、循环网络与注意力 |
| 第四篇 | [language_foundations.py](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/modern_ai/language_foundations.py)（代码包内：`现代AI示例/modern_ai/language_foundations.py`）、[language.py](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/modern_ai/language.py)（代码包内：`现代AI示例/modern_ai/language.py`） | 分词、注意力、训练、采样与缓存 |
| 第五篇 | [posttraining.py](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/modern_ai/posttraining.py)（代码包内：`现代AI示例/modern_ai/posttraining.py`） | SFT、DPO、PPO、LoRA 与量化 |
| 第六篇 | [visual_audio.py](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/modern_ai/visual_audio.py)（代码包内：`现代AI示例/modern_ai/visual_audio.py`）、[diffusion_lab.py](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/modern_ai/diffusion_lab.py)（代码包内：`现代AI示例/modern_ai/diffusion_lab.py`） | 图文表示、扩散、声学与语音 |
| 第七篇 | [retrieval.py](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/modern_ai/retrieval.py)（代码包内：`现代AI示例/modern_ai/retrieval.py`）、[agent.py](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/modern_ai/agent.py)（代码包内：`现代AI示例/modern_ai/agent.py`）、[rag_agent_lab.py](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/modern_ai/rag_agent_lab.py)（代码包内：`现代AI示例/modern_ai/rag_agent_lab.py`）、[transport.py](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/modern_ai/transport.py)（代码包内：`现代AI示例/modern_ai/transport.py`） | 检索、LLM Wiki 修订核验、工具校验、循环与 HTTP |
| 第八篇 | [mcp_stdio.py](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/modern_ai/mcp_stdio.py)（代码包内：`现代AI示例/modern_ai/mcp_stdio.py`）、[ecosystem_lab.py](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/modern_ai/ecosystem_lab.py)（代码包内：`现代AI示例/modern_ai/ecosystem_lab.py`）、[cooperation_lab.py](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/modern_ai/cooperation_lab.py)（代码包内：`现代AI示例/modern_ai/cooperation_lab.py`） | 协议、扩展、框架与多智能体 |
| 第九篇 | [persistence_lab.py](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/modern_ai/persistence_lab.py)（代码包内：`现代AI示例/modern_ai/persistence_lab.py`）、[runtime.py](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/modern_ai/runtime.py)（代码包内：`现代AI示例/modern_ai/runtime.py`）、[frameworks.py](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/modern_ai/frameworks.py)（代码包内：`现代AI示例/modern_ai/frameworks.py`） | 状态机、事务、租约与检查点 |
| 第十篇 | [harness_lab.py](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/modern_ai/harness_lab.py)（代码包内：`现代AI示例/modern_ai/harness_lab.py`）、[harness.py](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/modern_ai/harness.py)（代码包内：`现代AI示例/modern_ai/harness.py`） | 上下文、执行安全、追踪与评测 |
| 练习 | [solutions.py](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/modern_ai/solutions.py)（代码包内：`现代AI示例/modern_ai/solutions.py`） | 原数字题号与 PT、MM、RA、PE、RT、HA 题号入口 |

~~~powershell
# 查看单题的中间计算结果；仍使用本目录的唯一环境。
.\.venv\Scripts\python.exe -B -m modern_ai.solutions RT-01
~~~

正文中的文件路径以整个代码包的根目录为准；例如这里的 [`modern_ai/agent.py`](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/modern_ai/agent.py)（代码包内：`现代AI示例/modern_ai/agent.py`）对应代码包根目录下的 `现代AI示例/modern_ai/agent.py`。

内置检索数据见 [notes.json](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/data/notes.json)（代码包内：`现代AI示例/data/notes.json`）。教学插件清单为 [plugin.json](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/extensions/reading-plugin/.claude-plugin/plugin.json)（代码包内：`现代AI示例/extensions/reading-plugin/.claude-plugin/plugin.json`），技能说明为 [SKILL.md](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/extensions/reading-plugin/skills/organize-notes/SKILL.md)（代码包内：`现代AI示例/extensions/reading-plugin/skills/organize-notes/SKILL.md`）。这些均为离线实验材料，无需安装第三方桌面软件或后台服务。

代码采用 [MIT 许可](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/LICENSE)（代码包内：`现代AI示例/LICENSE`），各依赖库保留原有许可。