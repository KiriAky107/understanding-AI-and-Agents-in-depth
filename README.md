# 深入理解AI和Agent

**——从线性代数开始的自学笔记**

这是一套拒绝“黑盒调包”与“悬空背公式”的现代 AI 与智能体工程自学笔记。
全书从最具体的资料检索与预测困难出发，先用向量、矩阵与微积分把数据表示、特征变换和误差反传一笔一划算清楚；随后沿着自回归语言模型（Transformer）、后训练对齐（SFT / DPO / PPO / LoRA / 量化）、视觉与语音多模态（ViT / CLIP / VLM / DDPM / CTC / RVC），一路推进到外挂知识库（RAG 混合检索与 LLM Wiki）、工具闭环、MCP 协议、多智能体独立窗口协作，最终落脚于跨越进程崩溃的 **Agent Runtime（SQLite 状态机、原子事务、Outbox 与租约防护令牌）** 与 **Agent Harness（上下文预算、安全沙箱、首次偏差追踪与常驻型 Super-agent）**，并在终章回到历史唯物主义视角审视技术与人类实践的关系。

每一个核心公式均配有可纸笔手算的微型样本、精确到中间张量的矢量过程图，以及在普通笔记本 CPU 上即可离线跑通的验证代码。

## 写作与制作说明

本书的底稿与手稿由作者独立撰写，所有微型样本的中间矩阵与误差数值由作者亲手计算核对，配套 Python 实验程序与测试套件由作者自行编写。写作过程中使用 AI 辅助纠错与行文润色；经润色后的文段均由作者结合代码与数学推导重新逐字审阅定稿。

正文中的原创 SVG 架构与计算过程图由 AI 辅助生成，插图则由GPT-image2.1生成，并严格依据正文中的实际张量形状、浮点数值与状态机分支逐项核对修正。引用的第三方漫画原画与角色插图均另行注明原始出处与许可协议。

## 全书阅读目录

| 篇章 | 章节标题与在线阅读入口 | 核心内容概览 |
| --- | --- | --- |
| **序言** | [深入理解AI和Agent：序言与学习路线](分章/00-序言与学习路线.md) | 系统四层分工、十篇掌握基线、三条阅读路线与三遍阅读法 |
| **第一篇** | [一、数据表示与数学基础](分章/01-数据表示与数学基础.md) | 向量内积与余弦、矩阵几何变换、行列式、正交投影、PCA/SVD 与交叉熵 |
| **第二篇** | [二、从预测到机器学习](分章/02-从预测到机器学习.md) | 线性/逻辑回归梯度下降、正则化与反求约束、数据泄漏防范与混淆矩阵 |
| **第三篇** | [三、神经网络、深度学习与上下文表示](分章/03-神经网络深度学习与上下文表示.md) | 多层折线与计算图反传、CNN 局部卷积、RNN/LSTM 时间展开与 Attention 寻址 |
| **第四篇** | [四、大语言模型的架构、训练与生成](分章/04-大语言模型的架构训练与生成.md) | UTF-8/BPE 分词、多头注意力三路反传、RoPE、因果掩码、Pre-LN 块与 KV Cache |
| **第五篇** | [五、后训练、微调与部署](分章/05-后训练微调与部署.md) | SFT 掩码与提示梯度、DPO 偏好与 KL 推导、PPO 双向截断、CoT 与口癖、LoRA 与量化 |
| **第六篇** | [六、视觉、语音与多模态](分章/06-视觉语音与多模态.md) | ViT 切块重排、CLIP 对比与归一化导数、VLM 拼接、检测分割、DDPM 扩散、STFT、CTC 与 RVC |
| **第七篇** | [七、从模型调用、RAG 检索到工具闭环](分章/07-模型调用RAG与工具闭环.md) | HTTP 边界、`deepcopy` 快照、前置过滤、BM25/RRF 混合检索、LLM Wiki、十道校验与 Agent 循环 |
| **第八篇** | [八、协议、扩展、框架与多智能体](分章/08-协议扩展框架与多智能体.md) | MCP `stdio` 握手与错误分层、Skill 三级按需加载与 Plugin、三角色多智能体账单、LangChain/LangGraph 映射 |
| **第九篇** | [九、Agent Runtime：状态、事务与恢复](分章/09-AgentRuntime状态与恢复.md) | 四类身份与七张表、FSM 与乐观锁、四表原子事务、Outbox 去重、租约与防护令牌、LangGraph 检查点与重入 |
| **第十篇** | [十、Agent Harness：上下文、边界与长程任务](分章/10-AgentHarness上下文与长程任务.md) | 192 字节预算账单、长期记忆四记录、尝试与业务分离、路径/进程沙箱、`gate` 确认绑定、追踪、评测与 Super-agent |
| **第十一篇** | [十一、技术回顾与自主意识生成的哲学思考](分章/11-技术回顾与主观意识的思考.md) | 第一人称幻觉、逻辑与思维倒置批判、自主需要、对象化实践与中文房间、死劳动价值转移与真实之爱 |
| **后记** | [后记](分章/12-后记.md) | 全书十一个台阶回顾、进阶阅读方向与致谢 |

## 拓展阅读

拓展阅读书目与版本见[后记](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E5%88%86%E7%AB%A0/12-%E5%90%8E%E8%AE%B0.md)（代码包内：`分章/12-后记.md`）。请按书目自行获取相应中文版本阅读；可下载的电子书可放入本地 `拓展阅读/` 目录，教材 PDF 不随仓库或代码包提供。

## 仓库目录结构与路径约定

本公开仓库完整保留分章正文、SVG 矢量配图与全部可运行 Python 示例工程，目录结构如下：

~~~text
├─ 分章/                     序言、第一至十一篇正文与后记 Markdown
├─ assets/                   当前正文引用的全部 SVG 矢量图与插图
├─ 现代AI示例/               全书唯一共用的 Python 实验工程目录
│  ├─ modern_ai/             按职责拆分的算法、模型、协议、Runtime 与 Harness 源码模块
│  ├─ data/                  内置自编学习笔记语料与实验数据（离线可用）
│  ├─ extensions/            教学用 Plugin 清单与 Agent Skill 三级加载示例
│  ├─ pyproject.toml         项目依赖声明
│  └─ uv.lock                精确锁定的跨平台依赖版本锁文件
├─ LICENSE                   代码与文档分项许可说明
└─ LICENSES/                 完整许可证文本
~~~

**关于正文中“代码包内”路径的查找约定**：
正文中所有标注为 `代码包内：...` 的相对路径，一律以克隆或解压后包含上述文件夹的**项目根目录**为起点计算，与读者将导出的 PDF 存放在哪个目录下无关。例如，`代码包内：现代AI示例/modern_ai/agent.py` 即指代根目录下的 `现代AI示例/modern_ai/agent.py`。

点击 GitHub 页面右上角的 **Code → Download ZIP** 即可获取全部公开文件。作者另行提供的完整离线代码包中还可包含 `docs/`（83 道练习题完整题解与各篇 JSON 核验记录）及 `tools/`（本地自动化回归检查脚本）；合篇排印稿、PDF、旧版归档与本地构建工具不纳入公开仓库。

## 快速运行指南

本项目使用 **Python 3.12** 与现代包管理器 **[uv](https://docs.astral.sh/uv/)**。全书十篇的所有实验共用 `现代AI示例/` 目录下的**唯一一个 `.venv` 虚拟环境**，默认配置轻量级 CPU 版 PyTorch，无需独立显卡，无需下载数 GB 的外部模型权重，也无需配置任何商业 API 密钥。

在克隆或解压后的目录中打开终端，执行以下命令即可一键复现：

~~~powershell
cd 现代AI示例

# 1. 严格读取 uv.lock 锁文件，一键同步创建唯一虚拟环境 .venv
uv sync --frozen --all-extras --python 3.12

# 2. 运行前五篇与第七篇基础核验入口
.\.venv\Scripts\python.exe -B -m modern_ai math
.\.venv\Scripts\python.exe -B -m modern_ai learning
.\.venv\Scripts\python.exe -B -m modern_ai networks
.\.venv\Scripts\python.exe -B -m modern_ai language
.\.venv\Scripts\python.exe -B -m modern_ai retrieval

# 3. 运行后训练、多模态、RAG/Agent、MCP/多智能体、Runtime 事务与 Harness 专项实验
.\.venv\Scripts\python.exe -B -m modern_ai.posttraining
.\.venv\Scripts\python.exe -B -m modern_ai.visual_audio
.\.venv\Scripts\python.exe -B -m modern_ai.rag_agent_lab
.\.venv\Scripts\python.exe -B -m modern_ai.ecosystem_lab all
.\.venv\Scripts\python.exe -B -m modern_ai.persistence_lab all
.\.venv\Scripts\python.exe -B -m modern_ai.harness_lab context

# 4. 按稳定题号单独查看任意练习题的中间推导状态（支持全部 83 道题号，如 04-06、PT-02、MM-05、RA-04、PE-04、RT-07、HA-08）
.\.venv\Scripts\python.exe -B -m modern_ai.solutions RT-07
~~~

各源码模块的详细职责划分与完整命令清单，请参阅 [`现代AI示例/README.md`](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/%E7%8E%B0%E4%BB%A3AI%E7%A4%BA%E4%BE%8B/README.md)（代码包内：`现代AI示例/README.md`）。后续运行任何章节实验时，请继续直接指定 `.\.venv\Scripts\python.exe`，切勿向系统全局或其他 Python 环境散装依赖。

## 开源许可与第三方素材说明

本项目采取代码与图文分项许可策略：

* **配套源码（`现代AI示例/` 下的代码文件）**：采用 **[MIT License](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/LICENSES/MIT.txt)**（代码包内：`LICENSES/MIT.txt`）许可；
* **原创书稿正文与原创 SVG 配图**：采用 **[CC BY-NC-SA 4.0（知识共享 署名-非商业性使用-相同方式共享 4.0 国际）](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/LICENSES/CC-BY-NC-SA-4.0.txt)**（代码包内：`LICENSES/CC-BY-NC-SA-4.0.txt`）许可。

具体适用范围与条款全文见根目录下的 [LICENSE](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/LICENSE)（代码包内：`LICENSE`）。

书中引用的第三方作品、XKCD 漫画原画、角色设定及同人插图等素材，其著作权与许可归原权利人所有。详细来源、作者署名与原始许可条款均逐一列明于 [LICENSE](https://github.com/KiriAky107/understanding-AI-and-Agents-in-depth/blob/main/LICENSE)（代码包内：`LICENSE`）的“第三方内容”部分，不适用本项目上述开源许可。

## 勘误与反馈

欢迎在仓库 **Issues** 中提交勘误或讨论。提交时请尽量注明所在的**篇章、小节标题、具体句子或公式步骤**，并附上你的复现输入与终端输出结果；若涉及数值计算或框架行为差异，请一并说明操作系统、Python 小版本及依赖环境信息，以便在完全一致的实验条件下快速定位核对。
