# paper-news-agent 设计与实施计划

> 目标：每天自动抓取 arXiv 新论文 → 按我的领域关键词筛选 → 用 LLM 精读并写成兼顾技术性与概括性的短文 → 附带元信息（收录情况 / 引用 / 机构 / 代码）→ 发布到一个每天更新的静态网站。

---

## 0. 环境与技术选型（已确认的前提）

| 项 | 结论 |
|---|---|
| Python | `/Users/ray/miniconda3/bin/python3.13`（系统自带 3.9.6 太老，不用） |
| Node | 未安装 → **前端不引入 Node 工具链**，用 Python + Jinja2 生成纯静态站 |
| LLM | **DeepSeek**（默认）走 Anthropic 兼容端点，同一个 SDK；精读 `deepseek-v4-pro`，粗筛 `deepseek-v4-flash`。可一行配置切回 Anthropic |
| 凭据 | 需要 `DEEPSEEK_API_KEY`；用哪个 key 由 `models:` 推导，未用到的后端不需要 key |
| 依赖管理 | `pyproject.toml` + `pip`（或后续装 `uv`） |
| 存储 | 每日 JSONL 为唯一真相源（git 友好）+ 派生 SQLite 作查询索引 |
| 部署 | GitHub Actions 定时任务 + GitHub Pages |

---

## 1. 总体架构

```
                  ┌─────────────────────────────────────────────┐
  arXiv OAI-PMH ─▶│ ① ingest      每日增量元数据 → raw JSONL      │
  arXiv API      │└─────────────────────────────────────────────┘
                          │
                  ┌───────▼─────────────────────────────────────┐
                  │ ② filter  三级漏斗（关键词→向量→LLM 打分）    │
                  └───────┬─────────────────────────────────────┘
                          │  ~200 篇/天 → ~30 篇
                  ┌───────▼──────────┐   ┌──────────────────────┐
                  │ ③ enrich          │◀─▶│ S2 / OpenAlex / DBLP │
                  │ 收录·引用·机构·代码 │   │ GitHub / ROR / HF    │
                  └───────┬──────────┘   └──────────────────────┘
                          │
                  ┌───────▼─────────────────────────────────────┐
                  │ ④ summarize  取全文 → LLM 结构化精读          │
                  └───────┬─────────────────────────────────────┘
                          │
                  ┌───────▼──────────┐   ┌──────────────────────┐
                  │ ⑤ store  JSONL+DB │──▶│ ⑥ site  静态站/RSS   │
                  └──────────────────┘   └──────────────────────┘
```

设计原则：

1. **每级独立、幂等、可重跑**。以 `arxiv_id@version` 为主键，每级把结果写回记录并打 `stage_done` 标记，`--force` 才重算。任一级挂掉不影响已完成的部分。
2. **贵的操作放最后**。LLM 精读是唯一昂贵环节，前面用免费手段把 200 篇砍到 10 篇。
3. **元信息与摘要解耦**。引用数会变，摘要不会；enrich 可以每周重跑刷新引用曲线，不用重新烧 token。

---

## 2. ① 数据抓取（ingest）

### 2.1 主通道：OAI-PMH 增量收割

```
http://export.arxiv.org/oai2?verb=ListRecords
    &from=2026-08-03&until=2026-08-04
    &metadataPrefix=arXiv&set=cs
```

- `metadataPrefix=arXiv` 给的是完整元数据：标题、作者、摘要、**全部 categories**、`comments`、`journal-ref`、`doi`、版本历史。`comments` 和 `journal-ref` 是判断"是否被收录"的第一手证据，比事后查数据库准。
- 分页用 `resumptionToken`；遇到 `503` 必须读 `Retry-After` 退避（arXiv 用 503 做流控，不是错误）。
- `set` 只能到大类粒度（`cs`、`math`、`stat`、`eess` 等），细分类在 metadata 的 `<categories>` 里自己过滤。

### 2.2 辅助通道

| 用途 | 接口 |
|---|---|
| 回填 / 按条件补抓 | Query API `export.arxiv.org/api/query?search_query=cat:cs.LG+AND+submittedDate:[...]`，**限速 1 req / 3s**，单次 `max_results ≤ 2000` |
| 当日速览、交叉验证 | 分类 RSS `https://rss.arxiv.org/rss/cs.LG` |
| 社区热度信号 | HF Daily Papers `https://huggingface.co/api/daily_papers?date=YYYY-MM-DD`（upvote 数，选配） |

### 2.3 时序注意点

arXiv 在美东 20:00（≈ UTC 次日 00:00–01:00）发布当日新公告，**周末不发布**。定时任务设 **UTC 05:00**（= 北京 13:00），既在 OAI feed 稳定之后，也落在 DeepSeek 的非峰值窗口——UTC 03:00 是北京 11:00，正好撞在双倍价时段。周末自动空跑。跨版本更新（v2/v3）也会出现在 OAI 增量里——按 `arxiv_id@version` 去重，只有当新版本改动了标题/摘要时才重新走后续流程。

---

## 3. ② 筛选（filter）—— 三级漏斗

配置文件 [`config/interests.yaml`](config/interests.yaml) 是这一层的唯一输入：分类白名单、
每个 topic 的 `keywords` / `description` / `avoid`、三级阈值、每日篇数与成本闸门。
实际数值以该文件为准（这里不复制一份，避免两处漂移）。

其中 `avoid` 是最容易被忽略但收益最大的一项：写清楚"不关心什么"对精度的提升，
往往比再加十个关键词更大。

**一级 · 关键词（免费，200→60）**
分类白名单 + 标题/摘要正则命中。用词边界匹配避免 `agent` 命中 `agentic reagent` 之类噪声，命中数与位置（标题权重更高）算个粗分。

**二级 · 语义向量（免费本地，60→30）**
`sentence-transformers` 跑本地模型（`BAAI/bge-small-en-v1.5`，Mac 上走 MPS，约 30ms/篇）。把每个 `topic.description` 编码成兴趣向量，与摘要向量算余弦。**这一级的价值是捞回关键词漏掉的**——比如一篇讲 "state space models for irregular sampling" 的论文不含任何 time-series 关键词，但语义上高度相关。两家的这个端点都不提供 embedding，本地跑既免费又离线。

**三级 · LLM 打分（`deepseek-v4-flash`，→5 深读 + 10 浅读）**
输入标题+摘要+分类+comments，输出受 schema 约束的：

```json
{ "score": 0-10, "matched_topics": ["agent"], "reason": "≤40字", "novelty": "incremental|notable|breakthrough", "read_depth": "deep|shallow|skip" }
```

schema 里所有对象都带 `additionalProperties: false` 且 `required` 覆盖全部字段——这既是
Anthropic structured outputs 的硬要求，也让 DeepSeek 侧的强制工具调用能被同一套客户端
校验器复用（见 §5.3）。实测 127 篇约 $0.012/天，可忽略。

**反馈闭环（Phase 3）**：网站上给每篇加 👍/👎，写回 `data/feedback.jsonl`。累积 100+ 标注后，把标注样本作为 few-shot 塞进三级 prompt（放在 prompt cache 前缀里，不增边际成本），阈值也据此自动校准。这是让筛选真正贴合个人口味的关键，但必须先有数据，所以放后期。

---

## 4. ③ 元信息富化（enrich）

这一节是"关键信息收集"的落地。每个字段都要**带来源和证据**，不能只给一个裸值。

### 4.1 是否被收录（venue_status）

按可信度递降依次尝试，第一个命中即采纳并记录来源：

1. arXiv `journal-ref` / `doi` 字段非空 → `published`（最硬的证据）
2. `comments` 正则：`(Accepted|To appear|Camera[-\s]?ready|Oral|Spotlight|Findings)` 附近 40 字内出现会议名（NeurIPS/ICML/ICLR/CVPR/ACL/EMNLP/AAAI/KDD/WWW/SIGIR/ICCV/ECCV/NAACL/COLM/TMLR/JMLR…）→ `accepted`
3. Semantic Scholar `publicationVenue` + `externalIds.DBLP` + `publicationTypes`
4. DBLP 标题检索
5. 都没有 → `preprint`

```json
"venue_status": { "state": "accepted", "venue": "NeurIPS", "year": 2026,
                  "track": "spotlight", "evidence": "Accepted at NeurIPS 2026 (Spotlight)",
                  "source": "arxiv_comments" }
```

### 4.2 引用数（citations）

- **Semantic Scholar 批量接口**：`POST /graph/v1/paper/batch?fields=citationCount,influentialCitationCount,publicationVenue,externalIds,authors.hIndex`，body `{"ids": ["ARXIV:2608.01234", ...]}`，单次最多 500 个 id。免费 key 建议申请（限速更宽松）。
- **OpenAlex 兜底**：arXiv 现在给每篇都分配 DOI `10.48550/arXiv.<id>`，所以可以直接 `https://api.openalex.org/works/doi:10.48550/arXiv.2608.01234` 拿 `cited_by_count` + `authorships[].institutions`。带上 `mailto=` 进 polite pool，配额很宽。

> **⚠️ 重要的现实约束**：当天新出的论文引用数一律是 0，把它放在卡片上当排序依据是无意义的。所以：
> - **Day-0 显著性信号**用 `author_prominence`（作者最高 h-index、是否有该领域高引作者）和 `institution_tier`，这些当天就有值。
> - 引用数改成**时间序列**：`citations_history: [{date, count}]`，每周重跑 enrich 追加一个点，网站上单独做一个"上升榜"（近 30 天引用增速 Top N）。这才是引用数真正有用的地方。

### 4.3 机构（institutions）

1. LLM 从 PDF 第一页作者块抽取（精读时顺手做，structured output 一个字段，零额外成本）
2. OpenAlex `authorships[].institutions` 交叉校验
3. ROR API `https://api.ror.org/organizations?affiliation=...` 做名称归一（"Google DeepMind" / "DeepMind" / "Google Research" 统一）

### 4.4 代码实现（code）

> **不要依赖 Papers with Code**——该站点已于 2025 年停止服务、数据并入 Hugging Face，API 不可靠。

1. 正则扫 `comments` + 摘要里的 `github.com|gitlab.com|huggingface.co/(models|datasets)` 链接
2. 扫全文（HTML/PDF 文本）——很多论文只在正文脚注给链接
3. **GitHub API 验活**：repo 是否存在、stars、`pushed_at`、是否有 README。很多论文写 "code will be released" 但仓库是空的或 404，必须验证。

```json
"code": { "url": "https://github.com/x/y", "verified": true, "stars": 412,
          "last_commit": "2026-07-28", "source": "abstract" }
```

---

## 5. ④ 精读与写作（summarize）—— 核心价值所在

### 5.1 取全文的三档策略（成本/质量杠杆）

| 档 | 做法 | 输入 token | 何时用 |
|---|---|---|---|
| `text`（默认） | arXiv HTML（`arxiv.org/html/<id>v<n>`，2023-12 后的 LaTeX 投稿基本都有）→ 转 Markdown；没有 HTML 时用 `pdfplumber` 抽文本 | ~12–20k | 绝大多数论文 |
| `pdf` | 原生 PDF document block（base64，≤32MB/600页） | ~2–3× text | 图表是核心贡献时（架构图、可视化类论文） |
| `abstract` | 只给标题+摘要+comments | ~600 | 浅读档（第二梯队的 20 篇） |

推荐默认 `text`：数学公式在 HTML 里是 LaTeX，模型读得比 PDF 里的排版乱码准，而且便宜一半以上。`pdf` 档由配置或三级筛选的 `needs_figures` 标记触发。

### 5.1b 文风：新闻稿的可读性 + 论文的严谨

这是被明确要求调过一轮的部分，规则写在
[`config/prompts/summarize_system.md`](config/prompts/summarize_system.md)：

- **先抑后扬，不铺场**。禁止"随着…的快速发展"式开头；第一段直接写什么地方不成立、代价是什么。
- **第二段第一句就要落地"原来是这么个招"**，再展开机制。类比只在真的准确时用。
- **形容词必须有数字支撑**。`革命性/颠覆性/开创性/里程碑/完美解决/遥遥领先` 直接禁用；
  "大幅提升"无意义，"相对提升 8.5pp" 才是新闻。
- **区分"论文显示"与"这暗示"**，不用"作者称"当没查证的挡箭牌。

### 5.2 输出 schema（受 schema 强约束）

```jsonc
{
  "tldr_zh": "一句话，≤40字",
  "tldr_en": "one sentence, ≤25 words",
  "article_zh": "300–500字，4段：① 问题与背景 ② 方法核心 ③ 实验证据 ④ 意义与局限",
  "key_contributions": ["≤3条，每条一句"],
  "method": {
    "core_idea": "技术性描述，允许出现公式与术语",
    "architecture": "…", "training_data": "…", "compute": "8×A100, 72h"
  },
  "results": [
    { "benchmark": "ETTh1", "metric": "MSE", "value": "0.361",
      "baseline": "PatchTST 0.379", "delta": "-4.7%" }
  ],
  "limitations": ["作者承认的 + 你读出来的，分别标注"],
  "why_it_matters_to_me": "针对我的 topics 说明相关性，≤80字",
  "tags": ["time-series", "state-space", "long-context"],
  "institutions": ["…"], "figures_worth_seeing": ["Fig.3 …"],
  "confidence": { "level": "high|medium|low", "caveat": "全文不完整/仅摘要 等" }
}
```

**"平衡技术性与概括性"如何落到 prompt 里**：不靠形容词，靠结构强制。`tldr` 逼出概括，`method.core_idea` + `results` 逼出技术细节，`article_zh` 的四段式模板固定叙事节奏。再给 1–2 篇**人工写好的范文**做 few-shot——这是最有效的手段，比任何风格描述都管用（范文放进 prompt cache 前缀，边际成本为 0）。

### 5.3 调用要点：两家后端的差异与共同的坑

两家都走 Anthropic Messages API，所以只有一个 SDK、一条调用路径
（[`src/pna/llm.py`](src/pna/llm.py) 的 `call_structured`）。差异集中在
[`src/pna/providers.py`](src/pna/providers.py) 的一张显式能力表里：

| | Anthropic | DeepSeek |
|---|---|---|
| schema 约束 | `output_config.format` 原生 structured outputs | **不支持**；改用强制单次工具调用（`tools` + `input_schema` + `tool_choice: {"type":"tool"}`） |
| prompt caching | 显式 `cache_control` | 服务端自动，`cache_control` 被忽略 |
| 拒答 fallback | 仅 Opus 5（按模型判定，不是按后端） | 无 |
| 峰值加价 | 无 | 北京时间 9-12、14-18 点双倍 |
| 图片 / PDF document block | 支持 | 不支持（所以 `pdf` 档只在 Anthropic 侧可用） |

**为什么要显式表而不是特性嗅探**：DeepSeek 会**接受并静默忽略** `output_config.format`。
发错了不会报错，而是返回散文，等到三个阶段之后才炸成 JSON 解析错误。

**为什么两家的"保证"都不直接信**：每次输出都在客户端按 JSON Schema 校验，失败时把校验器
的具体报错喂回模型重试（最多 3 次，`usage.retries` 单独计数）。DeepSeek 官方文档也提示
JSON 响应偶发为空。

踩坑清单（都是会静默出错或多花钱的；标 *[A]* 的只影响 Anthropic 侧）：

- *[A]* **Opus 5 默认开启思考**，`max_tokens` 是"思考 + 正文"的总上限。设小了会在正文中途截断，而且不报错，只是 `stop_reason == "max_tokens"`。
- *[A]* **prompt caching 是前缀匹配**。system 里绝对不能插日期、论文 id、UUID——一旦插入，后面全部失效。日期放到 user turn 里。Opus 5 的最小可缓存前缀是 512 token（比 4.8 的 1024 更宽松），我们的 system+范文肯定够。
- *[A]* **并发要先热一发**。缓存条目要等第一个响应开始流式返回才可读。所以先发 1 篇，拿到首 token 后再并发剩下的，否则 10 篇全部 cache miss。默认 5 分钟 TTL 够用（10 篇顺序跑完通常在 5 分钟内）；若要拉长并发窗口再考虑 `ttl: "1h"`（写入成本 2×，需 ≥3 次读取才回本）。
- *[A]* **必须处理 `stop_reason == "refusal"`**。Opus 5 的安全分类器可能拒绝——安全/生物方向的正常论文有小概率误触。直接读 `response.content[0]` 的代码会崩。开 `fallbacks="default"` 让服务端自动转 Opus 4.8 重试。
- *[A]* **Opus 5 的行为调优**（来自官方迁移指南，直接影响输出质量）：
  - 默认输出比前代长，且**降 effort 不能可靠缩短可见输出**——必须在 prompt 里明确写字数（我们的 schema 已经写了 300–500 字）。
  - **不要写 "double-check your answer" / "verify before responding"** 这类自检指令。Opus 5 本来就会自我校验，加了反而过度验证、浪费 token。这条反直觉，容易踩。
  - 加一条范围约束，防止它自行扩大任务（"只输出 schema 要求的字段，不要附加分析"）。
- *[A]* **structured outputs 与 citations 功能互斥**（同时用返回 400）。所以事实溯源不走 citations API，改为在 schema 里要求 `results` 字段给出可核对的具体数值（下节做程序化校验）。

### 5.4 幻觉防线（不可省）

摘要类应用最大的风险是编造实验数字。两道机械化检查，全部免费：

1. **数值核对**：从 `results[].value` / `delta` 抽出所有数字，在原文纯文本里做字符串检索。找不到的标记 `unverified_number` 并在网站上打警示标。
2. **术语核对**：`tags` 和 `method.architecture` 里的专有名词必须在原文出现过。
3. 人工黄金集：挑 10 篇你读过的论文，每次改 prompt 后跑一遍人工评分（准确性/技术密度/可读性各 1–5 分），记入 `evals/`。**没有这个基线，prompt 迭代就是盲改。**

---

## 6. ⑤⑥ 存储与网站

### 6.1 存储

- **真相源**：`data/papers/YYYY-MM-DD.jsonl`，一行一篇完整记录（含所有阶段产物）。JSONL 的 diff 在 git 里可读，出错可追溯，比 SQLite 二进制入库友好。
- **派生索引**：`build` 时把 JSONL 灌进 `papers.db`（SQLite + FTS5 全文索引），不入 git，可随时重建。
- `data/feedback.jsonl` 存 👍/👎 标注。

### 6.2 网站（纯静态，不需要 Node）

Jinja2 模板 → `site/` 输出，GitHub Pages 托管。

页面：
- `/` 今日 digest：按 `score` 排序的卡片流。卡片正面 = `tldr` + 标签 + 徽章（收录状态 / ⭐ 代码 / 引用数 / 机构 logo），展开 = `article_zh` + 方法 + 结果表 + 局限。
- `/paper/<arxiv_id>/` 单篇永久页（利于分享和被搜索引擎收录）。
- `/archive/` 按日期 + 按 tag 的归档。
- `/trending/` 近 30 天引用增速榜（依赖 4.2 的引用曲线）。
- `/feed.xml` Atom 输出——这样你可以直接用现有的 RSS 阅读器消费，不必天天开网页。**这是最容易被低估的功能，建议 Phase 2 就做。**

前端交互（标签筛选、全文搜索）用不到 200 行原生 JS + 构建期生成的 `search-index.json` 搞定，不引入任何打包工具。深浅色主题跟随系统。

### 6.3 自动化

`.github/workflows/daily.yml`：

```yaml
on:
  schedule: [{ cron: "0 3 * * *" }]   # UTC 03:00，arXiv 公告后
  workflow_dispatch:
jobs:
  daily:
    steps: [ checkout, setup-python 3.13, pip install -e .,
             pna run --date auto,          # ingest→filter→enrich→summarize
             pna build-site,
             commit data/ ,                 # JSONL 回写仓库
             deploy-pages ]
```

Secrets：`ANTHROPIC_API_KEY`、`SEMANTIC_SCHOLAR_API_KEY`、`GITHUB_TOKEN`（内置）。
另加一个 `weekly.yml`：只跑 `pna enrich --refresh-citations --since 90d`，维护引用曲线。

失败可见性：任务失败时用 workflow 的失败通知；`pna run` 结束打印当日 token/成本汇总，写入 `data/runs/YYYY-MM-DD.json`。

---

## 7. 成本预算

单价（USD / 1M tokens）：

| 模型 | 输入(未命中缓存) | 输入(命中缓存) | 输出 |
|---|---|---|---|
| deepseek-v4-pro | $0.435 | $0.003625 | $0.87 |
| deepseek-v4-flash | $0.14 | $0.0028 | $0.28 |
| claude-opus-5 | $5.00 | $0.50 | $25.00 |
| claude-haiku-4-5 | $1.00 | $0.10 | $5.00 |

**当前配置（DeepSeek，非峰值）：**

| 环节 | 用量假设 | 日成本 |
|---|---|---|
| ingest / filter 一二级 | 免费（本地 embedding） | $0 |
| filter 三级（v4-flash） | ~130 篇 × 500 token 入 / 80 出 | ~$0.012 |
| 深读 ×5（v4-pro, text 档） | 16k 入 + ~3k 出 | ~$0.048 |
| 浅读 ×10（v4-flash） | 0.6k 入 + ~1.5k 出 | ~$0.005 |
| **合计** | | **~$0.065/天 ≈ $1.4/月** |

同配置把深读换成 Opus 5 约 **$27/月**，差价约 20×。

**两个必须记住的成本陷阱：**

1. **DeepSeek 峰值双倍价**：北京时间 9-12 点与 14-18 点。定时任务放在 UTC 05:00
   （= 北京 13:00，非峰值窗口）。UTC 03:00 是北京 11:00，正好撞峰值——改 cron 前先算时区。
   `providers.py` 里的 `multiplier_now()` 会把这个倍数算进成本统计，所以 `data/runs/`
   里记的是真实花费而不是名义价。
2. **重试也要付钱**：schema 校验失败会带着校验器报错重试（最多 3 次）。`usage.retries`
   单独统计，如果这个数字持续偏高，说明 prompt 或 schema 需要调，而不是加大重试次数。

优化杠杆（按性价比排序）：

1. **收紧一二级漏斗**：现在有 127 篇进三级打分，是三级成本的全部来源。把
   `keyword_min_score` 提到 2.0 或启用二级向量筛选，能砍掉一半以上。
2. **`deep_effort` 从 high 降到 medium**：思考 token 约减半。需用黄金集实测质量影响。
3. **A/B 深读模型**：这是唯一值得花钱验证的地方。`limitations[source=reader]`
   （识别不公平基线、缺失消融）最吃模型能力，实测成本约 $0.60（Opus）+ $0.03（DeepSeek）。
4. **Batch API**：Anthropic 侧可再省 50%，但要引入两阶段轮询。DeepSeek 侧无此机制。
   在当前量级（$1.4/月）完全不值得做。

硬闸门：`budget.usd_max_per_day` 用 `count_tokens()` 预估，超限就截断当天的深读列表。

---

## 8. 仓库结构

```
paper-news-agent/
├── pyproject.toml
├── .env.example                    # ANTHROPIC_API_KEY, SEMANTIC_SCHOLAR_API_KEY
├── PLAN.md
├── config/
│   ├── interests.yaml              # 领域/关键词/阈值/预算（唯一需要你手写的文件）
│   └── prompts/{triage.md,summarize.md,style_guide.md,fewshot/*.md}
├── src/pna/
│   ├── cli.py                      # pna ingest|filter|enrich|summarize|build-site|run
│   ├── config.py
│   ├── sources/{oai.py,query_api.py,rss.py,fulltext.py}
│   ├── filter/{keyword.py,embed.py,llm_triage.py}
│   ├── enrich/{semantic_scholar.py,openalex.py,dblp.py,venue.py,code.py,ror.py}
│   ├── summarize/{client.py,schema.py,runner.py,verify.py}
│   ├── store/{jsonl.py,db.py,models.py}
│   └── site/{build.py,templates/,static/}
├── data/{papers/*.jsonl,runs/*.json,feedback.jsonl}
├── evals/{golden_set.yaml,score.py}
├── site/                           # 构建产物
├── tests/
└── .github/workflows/{daily.yml,weekly.yml}
```

---

## 9. 分阶段实施

### Phase 1 — 端到端最小闭环（目标：2 个工作日）
`ingest(OAI) → filter(一级+三级) → summarize(text 档, deep only) → JSONL → 单页 HTML`，本地手动跑。
- 跳过：embedding 二级、enrich、Batch、网站交互。
- 验收：`pna run --date 2026-08-04` 产出 8–10 篇结构化摘要 + 一个能在浏览器打开的 digest 页；`stop_reason` 与 token 用量打印正常；成本 < $2。

### Phase 2 — 元信息与网站（目标：+2 天）
补 enrich 全套（收录/引用/机构/代码，含 GitHub 验活）、embedding 二级筛选、Jinja2 多页站 + Atom feed + 客户端搜索、GitHub Actions 每日自动跑并部署。
- 验收：连续 5 天无人工干预自动更新；随机抽 10 篇人工核对 `venue_status` 与 `code.verified` 准确率 ≥ 90%。

### Phase 3 — 质量与个性化（目标：+3 天，持续迭代）
黄金集评测脚本、数值核对幻觉防线、👍/👎 反馈闭环与阈值自校准、引用曲线 + trending 榜、weekly 刷新任务、成本看板。
- 验收：黄金集平均分 ≥ 4/5；`unverified_number` 比例 < 5%。

### Phase 4 — 选配增强
Batch API 降本、邮件日报、中英双语输出、`pdf` 档（图表理解）、跨天主题聚类（"本周 agent 方向出现 3 篇同类工作"）、与个人 Zotero/Notion 打通。

---

## 10. 已知风险与对策

| 风险 | 对策 |
|---|---|
| arXiv 限流 / OAI 503 | 严格退避 + `resumptionToken` 断点续传；抓取结果先落盘再进下一级 |
| 新论文引用数恒为 0，排序无意义 | Day-0 用作者 h-index / 机构；引用做成时间序列另开 trending 榜（§4.2） |
| LLM 编造实验数字 | 数值/术语机械核对 + 网站警示标 + 黄金集回归（§5.4） |
| 筛选跑偏（漏掉真正相关的） | 二级向量层专治关键词漏检；每周人工扫一遍被 filter 掉的 borderline 列表（score 4–6），据此调阈值 |
| 成本失控 | `count_tokens` 预估 + 每日硬闸门 + runs/ 成本日志 |
| Opus 5 输出过长 / 拒答 | schema 明确字数 + `fallbacks="default"` + 显式检查 `stop_reason`（§5.3） |
| 单篇论文过长撑爆输入 | 超长时按章节裁剪（保留 abstract/intro/method/experiments/conclusion，丢 related work 与附录），并在 `confidence.caveat` 里标注 |
| arXiv HTML 缺失 | 回退 `pdfplumber` 文本；两者都失败则降级到 abstract 档并标注 |

---

## 11. 已确认的决策

| 项 | 决定 | 日期 |
|---|---|---|
| 摘要语言 | 中英双份完整摘要（`article_zh` + `article_en`） | 2026-08-04 |
| 每日篇数 | 5 深读 + 10 浅读 | 2026-08-04 |
| 托管 | GitHub Pages | 2026-08-04 |
| LLM 后端 | DeepSeek（`v4-pro` 深读 / `v4-flash` 粗筛与浅读） | 2026-08-04 |

仍待补充：`config/interests.yaml` 里目前只有 time-series / agent 两个种子 topic，
`avoid` 段落是推测的，需要按真实兴趣扩充。范文 few-shot 要等第一批真实输出被评价后再写。
