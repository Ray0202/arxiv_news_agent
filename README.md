# paper-news-agent

每天抓取 arXiv 新论文 → 三级漏斗筛选 → LLM 精读并写成中英双份摘要 → 生成静态网站。

完整设计与后续阶段见 [PLAN.md](PLAN.md)。当前实现的是 **Phase 1**（端到端最小闭环）。

## 安装

```bash
python3.13 -m venv .venv
.venv/bin/pip install -e .
cp .env.example .env    # 填入 DEEPSEEK_API_KEY
```

## 使用

```bash
.venv/bin/pna run                      # 全流程，日期取最近一个工作日
.venv/bin/pna run --date 2026-07-30    # 指定日期
.venv/bin/pna stats                    # 看每天的漏斗数字
open docs/index.html
```

各级可以单独跑，也可以重复跑（幂等，按 `arxiv_id` 记状态）：

```bash
.venv/bin/pna ingest    --date 2026-07-30   # OAI-PMH 抓元数据
.venv/bin/pna filter    --date 2026-07-30   # 一级：分类 + 关键词
.venv/bin/pna triage    --date 2026-07-30   # 三级：LLM 打分
.venv/bin/pna summarize --date 2026-07-30 --limit 1   # 精读（--limit 便于试成本）
.venv/bin/pna build-site --all
```

只有 `summarize` 花钱。`--force` 才会重算已完成的阶段，所以重跑站点构建不会重复付费。

## 配置

只需要改 [config/interests.yaml](config/interests.yaml)：领域关键词、`avoid` 否定描述、
阈值、每日篇数与成本闸门。提示词在 [config/prompts/](config/prompts/)。

## 后端（OpenAI / DeepSeek / Anthropic）

默认走 `gpt-5.4-mini`。换后端只要改 `models:` 里的 model 名字——每一级独立按 model id
解析 provider，可以混用。能力差异由 [src/pna/providers.py](src/pna/providers.py) 处理：

| | OpenAI | Anthropic | DeepSeek |
|---|---|---|---|
| 协议 | Chat Completions | Messages | Messages（兼容端点） |
| schema 约束 | `json_schema` + `strict:true`，**服务端强制** | 原生 structured outputs | **不支持**，改用强制单次工具调用 |
| 输出上限参数 | `max_completion_tokens`（`max_tokens` 直接报错） | `max_tokens` | `max_tokens` |
| 推理档位 | `reasoning_effort`：none/low/medium/high/xhigh（**无 minimal**） | `output_config.effort` | 同左 |
| prompt caching | 自动，需共享前缀 ≥1024 tokens | 显式 `cache_control` | 自动 |
| 峰值加价 | 无 | 无 | 北京 9-12、14-18 点双倍 |

**OpenAI 侧最需要防的坑**：`max_completion_tokens` 同时覆盖 reasoning 和可见输出。实测
`xhigh` 能把 6000 的预算全部花在 reasoning 上，返回 `finish_reason="length"` 且
`content=""`——**空字符串而不是半截 JSON**，`json.loads` 会在离病因三层远的地方炸。
代码里显式检查 `finish_reason` 并报出实际花在 reasoning 上的 token 数。

两个坑写在代码注释里，这里也记一下：

1. DeepSeek **接受并静默忽略** `output_config.format`。发错了会拿到散文而不是 JSON，而且
   要到三个阶段之后才报 JSON 解析错。所以 provider 能力是一张显式表，不做特性嗅探。
2. 无论哪家的"保证"都不足信：每次输出都在客户端按 schema 校验，失败时把校验器的报错喂回
   模型重试（最多 3 次，重试也计入成本）。DeepSeek 官方文档也提示 JSON 响应偶发为空。

GitHub Actions 的 cron 定在 `0 5 * * 1-6`（UTC 05:00 = 北京 13:00），落在 DeepSeek 的
非峰值窗口。**改这一行前先重算一遍时区**——UTC 03:00 是北京 11:00，正好双倍价。

### 成本（实测，同一天 881→127→15 篇）

| 后端 | 打分 127 篇 | 精读+速览 15 篇 | 合计/天 | 折月 |
|---|---|---|---|---|
| **gpt-5.4-mini**（当前） | $0.143 | $0.342 | **$0.485** | ~$10.7 |
| deepseek v4-pro/flash | $0.008 | $0.055 | $0.062 | ~$1.4 |
| claude-opus-5 深读 | — | — | ~$1.3 | ~$27 |

gpt-5.4-mini 比 DeepSeek 贵约 8×，换来的是：**0/98 个数字未核到**（DeepSeek 4/98 里有
一处真幻觉）、**0 次 schema 重试**、tldr 字数守住 40 字上限（DeepSeek 长期 55-73 字）。

打分那一级没有命中缓存：triage 的 system prompt 只有约 600 tokens，**够不到 OpenAI 自动
缓存的 1024 token 门槛**；精读那一级的 prompt 约 1671 tokens，实测缓存读 28160 tokens。
如果要压打分成本，把 triage 换成 `deepseek-v4-flash` 或 `gpt-5.4-nano` 即可（一行配置）。

## 渲染

**公式**：模型被要求把所有数学包进 `$...$`，页面用 KaTeX 渲染（CDN + 校验过的 SRI 哈希）。
首次实测模型一个都没包，摘要里全是 `S_{i-1}`、`γ^{j-i}` 这类裸片段，直接显示成乱码，所以
除了改 prompt 还加了构建期归一 [`site/mathfix.py`](src/pna/site/mathfix.py) 兜底。

兜底刻意保守，只处理带**花括号**下标/上标的片段（`S_{i-1}`）和反斜杠命令。两类不管：

- 散文里的 Unicode 符号（`抽取→存储→检索` 是句子，`8×H800` 是规格），包起来只会更糟。
- 无花括号的纯 Unicode 表达式（`λ·Σ||M_m||_F²`）。只包住能识别的那一小段会得到
  `λ·Σ||$M_m$||_F²`——半截进数学模式比整体不进更难读。这类靠 prompt 解决。

SRI 哈希必须用 `openssl dgst -sha384` 从真实文件算，**不能凭记忆写**：哈希不匹配浏览器会
直接拒载脚本，唯一症状是全页公式都不渲染。升版本时三个都要重算。

**图片**：从 arXiv HTML 的 `<figure>` 抽出编号、caption 和绝对 URL（`src` 是相对路径且已
含版本目录，用最终响应 URL 做 base 拼接）。精读时把图片清单喂给模型，它按**编号**挑至多 3 张
并说明"这张图给出正文说不清的什么"，站点据编号精确匹配。对不上的编号直接丢弃，不渲染坏图。

目前是**热链 arxiv.org 原图**（你说的"先试着放原图"）。若日后要自托管，改
`_extract_figures` 下载到 `site/static/figures/` 即可。

## 数据

- `data/papers/YYYY-MM-DD.jsonl` —— 唯一真相源，一行一篇，含各级产物。
- `data/runs/YYYY-MM-DD.json` —— 每次运行的漏斗数字与 token/成本。
- `cache/` 与 `data/papers/` 不入 git（前者可重建，后者会让仓库无限膨胀）。
- `docs/` 是渲染出来的站点，**要**入 git —— GitHub Pages 从这个目录发布。

## 当前状态

| 阶段 | 状态 |
|---|---|
| ingest (OAI-PMH) | ✅ 真实数据验证：971 条 → 881 篇新投稿（正确剔除 90 篇改元数据的旧论文） |
| filter 一级 | ✅ 881 → 127，词边界匹配无 `reagent` 类误报 |
| fulltext (HTML→MD) | ✅ 公式保留为 LaTeX，摘要去重，超长按章节裁剪 |
| provider 抽象 | ✅ 单测锁定路由：DeepSeek 不发 structured outputs、fallback 只发给 Opus 5、峰值时段算价 |
| filter 三级 | ✅ 127 篇 → 50 篇过线，0 失败，45s，$0.008（非峰值） |
| summarize | ✅ 5 深读 + 10 浅读，0 失败 0 重试，4m14s，$0.055（非峰值） |
| 幻觉防线 | ✅ 深读 80 个数字核对，78 通过；2 个告警是模型自己算出来、论文没写的数（Σ C(20,k)=21699 ≈ "~21700"）——告警正确 |
| 数值核对 | ✅ 单测覆盖（含前导零/千分位/derived delta 的边界） |
| 静态站 | ✅ fixture 渲染验证，含 HTML 转义与空日期 |
| enrich（引用/机构/代码验活） | Phase 2，见 PLAN.md |

`pytest` 68 项全绿，不需要任何 API key。

### 实测发现（都已修掉或记录在代码注释里）

1. **思考模式与强制 `tool_choice` 不兼容**：DeepSeek 对 `{"type":"tool","name":...}` 报
   `Thinking mode does not support this tool_choice`，但 `{"type":"any"}` 与思考并存。
   因为只挂一个工具，`any` 的约束力等价。
2. **`score` 被返回成字符串 `"8"`**：100% 的 triage 调用如此，靠重试纠正等于白花一倍钱。
   加了无损类型归一后，重试 83 → 5，成本 -62%，耗时 -37%。
3. **作者/机构块被我的抽取丢掉了**：LaTeXML 放在 `ltx_authors` 里，我没有对应分支，
   导致 `institutions` 恒为空——看起来像模型能力问题，其实是我的 bug。
4. **`why_it_matters_to_me` 无解**：精读 prompt 里没注入兴趣画像，模型只能回答
   "读者研究方向未明确提供"。
5. **字数超标改不动**：见 `config/interests.yaml` 的注释，最后是调上限而不是加压 prompt。

## 定时部署

`deploy/com.paper-news-agent.daily.plist` 是 launchd 任务的副本，纳入版本控制是因为
它是这套部署的一部分，而 `~/Library/LaunchAgents/` 不在任何备份链里。安装：

```bash
cp deploy/com.paper-news-agent.daily.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.paper-news-agent.daily.plist
```

周一到周五 12:00 起，每小时一次共 5 次机会（本地时间，自动跟随夏令时）。
成功后当天剩下的触发在毫秒级退出。周末不触发：arXiv 周末不公告。
