# Ralph 用户指南

这份文档是对根目录 [README.md](/D:/project/AI-Coding/ralph-longtask/README.md) 的展开版，面向第一次接触 Ralph 的开发者。

你可以把 Ralph 理解成一个“自动推进开发任务的 CLI”：

- 你先把需求整理成 `prd.json`
- Ralph 再按故事顺序逐个执行
- 每一轮都启动一个全新的 Claude Code 会话
- 只有通过验证的故事才会被自动标记完成

如果你只想快速理解全貌，先读 README。  
如果你准备实际落地使用，请按这份用户指南一步一步操作。

## 两条推荐使用路径

### 路径 A：只使用 Ralph CLI

适合：

- 你已经知道要做什么
- 你只想把需求变成 `prd.json` 然后让 Ralph 执行
- 你还不想引入 OpenSpec 或更完整的 pipeline

### 路径 B：使用 OpenSpec + Superpowers + Ralph

适合：

- 你希望先做设计和评审，再进入执行
- 你想保留 `spec -> review -> convert -> execute -> archive` 的阶段化流程
- 你希望在对话里用 skill 管理 gate，在 CLI 里用 `ralph pipeline` 管理状态

---

## 通用前置条件

### 1. 安装 Node.js

要求：

- Node.js `>= 18`

检查：

```bash
node --version
npm --version
```

### 2. 安装 Claude Code CLI

Ralph 本身不生成代码。  
它调度的是 Claude Code CLI。

安装：

```bash
npm install -g @anthropic-ai/claude-code
```

验证：

```bash
claude --version
```

### 3. 安装 Ralph

在本仓库根目录执行：

```bash
npm install
```

如果你希望系统中任何项目目录都能直接运行 `ralph` 命令，再执行：

```bash
npm link
```

验证：

```bash
ralph --help
ralph-pipeline --help
```

如果不做全局安装，也可以直接运行：

```bash
node D:/project/AI-Coding/ralph-longtask/ralph.js --config .
```

---

## 路径 A：Ralph CLI 单独使用

这是最简单、最推荐的新手上手方式。

### 整体流程

1. 准备项目目录
2. 生成 Markdown PRD
3. 转成 `prd.json`
4. 可选创建 `RALPH.md`
5. 创建 `ralph.config.json`
6. 运行 Ralph
7. 查看结果和继续迭代

---

### 第 1 步：准备项目目录

最终推荐结构：

```text
my-project/
├── src/
├── tasks/
│   └── prd-my-feature.md
├── prd.json
├── progress.txt
├── RALPH.md
├── CLAUDE.md
└── ralph.config.json
```

这些文件的作用：

- `tasks/prd-my-feature.md`
  人类和 AI 都更容易阅读的 Markdown PRD
- `prd.json`
  Ralph 真正执行的任务清单
- `progress.txt`
  Ralph 自动写入的进度日志和 learnings 来源
- `RALPH.md`
  你希望每轮都遵守的固定规则
- `CLAUDE.md`
  项目本身已有的协作规范，可选
- `ralph.config.json`
  Ralph 的项目级配置

---

### 第 2 步：使用 `prd` skill 生成 Markdown PRD

如果你还没有成型的 PRD，先用 `prd` skill。

示例：

```text
/ralph-skills:prd "给我的项目增加通知中心功能"
```

`skills/prd/SKILL.md` 的职责是：

1. 先问必要澄清问题
2. 生成结构化 PRD
3. 保存到 `tasks/prd-[feature-name].md`

你可以预期得到类似文件：

```text
tasks/prd-notification-center.md
```

通常会包含：

- Introduction
- Goals
- User Stories
- Functional Requirements
- Non-Goals
- Success Metrics
- Open Questions

如果你已经有 Markdown PRD，可以直接跳过这一步。

---

### 第 3 步：使用 `ralph` skill 转成 `prd.json`

Ralph CLI 不直接执行 Markdown 文件。  
它执行的是结构化 JSON。

所以你需要用 `ralph` skill 做转换：

```text
/ralph-skills:ralph "把 tasks/prd-notification-center.md 转成 prd.json"
```

`skills/ralph/SKILL.md` 会帮你做几件事：

- 把大需求拆成多个可执行故事
- 调整故事顺序，先依赖、后 UI
- 给每个故事补上可验证的验收标准
- 生成可执行的 `prd.json`

一个最小 `prd.json` 示例：

```json
{
  "project": "my-project",
  "branchName": "ralph/notification-center",
  "description": "Notification center for users",
  "userStories": [
    {
      "id": "US-001",
      "title": "Add notifications table",
      "description": "As a developer, I want to persist notifications so the system can store them.",
      "acceptanceCriteria": [
        "Add notifications table and migration",
        "Typecheck passes"
      ],
      "priority": 1,
      "passes": false,
      "notes": ""
    }
  ]
}
```

### 这里最重要的两条规则

#### 规则 1：每个故事必须足够小

Ralph 每一轮都是“全新上下文 + 单个故事”。  
如果 story 太大，Claude 很可能做不完，或做完也无法通过验证。

合适的 story 例子：

- 添加一个数据库字段和迁移
- 给现有页面新增一个小组件
- 为某个接口补一个明确逻辑

不合适的 story 例子：

- “实现完整 dashboard”
- “重构整个认证系统”
- “完成整个通知模块”

#### 规则 2：验收标准必须可验证

推荐写法：

- `Typecheck passes`
- `Tests pass`
- `Verify in browser using dev-browser skill`
- `Add status column with default pending`

不要写：

- `Works correctly`
- `Good UX`
- `Handles all edge cases`

### UI 故事特别注意

如果某个故事涉及 UI，建议总是加上：

```text
Verify in browser using dev-browser skill
```

否则“代码改了”不等于“界面真的可用”。

---

### 第 4 步：可选创建 `RALPH.md`

如果你希望每一轮 Claude Code 都遵守一套固定规范，就创建 `RALPH.md`。

Windows PowerShell 示例：

```powershell
Copy-Item D:/project/AI-Coding/ralph-longtask/templates/RALPH.md ./RALPH.md
```

你可以在里面写：

- 提交规范
- 测试要求
- 不允许碰的目录
- 前端或后端约定
- 数据库迁移规则

没有 `RALPH.md` 也能运行，只是少了一层固定团队规范。

---

### 第 5 步：创建 `ralph.config.json`

推荐从这份配置起步：

```json
{
  "prdPath": "./prd.json",
  "progressPath": "./progress.txt",
  "maxIterations": 10,
  "cooldownSeconds": 3,
  "permissionsMode": "full",
  "claude": {
    "maxTurns": 50
  },
  "prompts": {
    "agentInstructionPath": "./RALPH.md",
    "extraContextPaths": [
      "./CLAUDE.md"
    ],
    "extraInstructions": "",
    "strictSingleStory": true
  },
  "validation": {
    "checkGitCommit": true,
    "patchPrdPasses": true,
    "validatePrdSchema": true,
    "acceptanceCommands": {
      "typecheck": "npm run typecheck",
      "tests": "npm test",
      "browser": "npm run test:browser"
    }
  }
}
```

### 关键字段解释

| 字段 | 作用 |
|------|------|
| `prdPath` | 指定 Ralph 读取哪个 `prd.json` |
| `progressPath` | 指定进度日志路径 |
| `maxIterations` | 最多执行多少轮 |
| `cooldownSeconds` | 两轮之间休息多久 |
| `permissionsMode` | Claude CLI 是否使用完整权限模式 |
| `claude.maxTurns` | 单轮 Claude 会话最多可转多少次 |
| `prompts.agentInstructionPath` | 指向 `RALPH.md` |
| `prompts.extraContextPaths` | 给每轮附加上下文，比如 `CLAUDE.md` |
| `prompts.strictSingleStory` | 是否注入严格“单轮只做一个故事”的协议 |
| `validation.acceptanceCommands.typecheck` | 处理 `Typecheck passes` |
| `validation.acceptanceCommands.tests` | 处理 `Tests pass` |
| `validation.acceptanceCommands.browser` | 处理 `Verify in browser using dev-browser skill` |

### `maxIterations` 和 `claude.maxTurns` 的区别

这是新手最容易混淆的一组配置。

- `maxIterations`
  控制 Ralph 整个执行循环最多跑多少轮
- `claude.maxTurns`
  控制单轮 Claude 会话内部最多允许多少轮交互

可以把它们理解成：

- `maxIterations` 是“Ralph 总共还能开几次工”
- `claude.maxTurns` 是“Claude 在某一次开工里最多能忙多久”

例如：

```json
{
  "maxIterations": 10,
  "claude": {
    "maxTurns": 50
  }
}
```

意思是：

- Ralph 最多发起 10 次独立执行轮次
- 每次执行轮次里，Claude 最多进行 50 次内部交互

如果你遇到的是：

- 故事还很多，但 Ralph 很快整体停了
  优先检查 `maxIterations`
- 单个故事常常做到一半就结束
  优先检查 `claude.maxTurns`

不过，如果同一个 story 总是需要非常大的 `maxTurns` 才勉强完成，通常更好的做法还是把 story 拆小。

### 一个很重要的注意点

如果某个 UI story 的验收标准里有：

```text
Verify in browser using dev-browser skill
```

但你没有配置：

```json
"browser": "..."
```

那么这个 story 不会被自动标记完成。

### 这些命令只是示例

配置里的：

- `npm run typecheck`
- `npm test`
- `npm run test:browser`

都只是示例。  
你需要换成你项目里真实存在的命令。

---

### 第 6 步：运行 Ralph

进入业务项目目录执行：

```bash
ralph
```

或者限制最多轮数：

```bash
ralph 20
```

如果没有全局安装：

```bash
node D:/project/AI-Coding/ralph-longtask/ralph.js --config .
```

### Ralph 每轮大致会做什么

1. 读取 `prd.json`
2. 找出优先级最高、且 `passes: false` 的故事
3. 组装 prompt
4. 启动一个新的 Claude Code 会话
5. 执行代码修改
6. 运行结构验证、completion signal 检查、git commit 检查和 acceptance commands
7. 如果全部通过，自动把当前故事的 `passes` 标记为 `true`
8. 把结果写入 `progress.txt`

---

### 第 7 步：看 Ralph 跑完后留下什么

#### `prd.json`

通过验证的故事会变成：

```json
"passes": true
```

#### `progress.txt`

会记录：

- 哪一轮处理了哪个 story
- 是否成功
- 验证失败的原因
- learnings 的来源内容

#### `.ralph-run-state.json`

会记录运行护栏状态，例如：

- 哪些 story 因为连续失败被自动跳过
- 这些自动跳过是在什么条件下触发的

如果某个 story 被熔断后你已经人工修好了，可以这样重新放回执行队列：

```bash
ralph --retry-story US-001
```

#### git 历史

如果开启了 `validation.checkGitCommit`，Ralph 会检查当前故事是否真的留下了对应提交痕迹。

#### `archive/`

如果你切换了 `branchName`，旧的 `prd.json` 和 `progress.txt` 会自动归档。

---

### 第 8 步：中断后继续

如果 Ralph 执行过程中被中断：

```bash
ralph --resume
```

这会恢复 Ralph 的执行循环。  
如果当前项目其实停在 pipeline 的 `execute` 之前，它也会优先衔接 pipeline 状态。

---

## 路径 A 常用命令

```bash
ralph
ralph 20
ralph --resume
ralph --story US-003
ralph --skip-story US-001 --skip-story US-002
ralph --max-runtime-minutes 45
ralph --max-failures-per-story 2
ralph --config ./path/to/project
```

---

## 路径 A 新增的运行护栏

- `--story US-XXX`
  只跑一个指定 story，适合排障、补跑或人工接管
- `--skip-story US-XXX`
  跳过某个已知坏 story，本次运行里不再碰它；这个 story 不会被标记为完成
- `--retry-story US-XXX`
  把一个因为连续失败而被持久跳过的 story 拉回执行队列
- `--dry-run`
  只预览当前 run 会挑哪些 story，不会启动 Claude，也不会写入新的执行进度
- `--max-total-tokens <n>`
  当本次 run 的估算总 tokens 达到上限后，在开始下一轮前停下
- `--max-total-cost-usd <n>`
  当本次 run 的估算总成本达到上限后，在开始下一轮前停下
- `--max-runtime-minutes <n>`
  运行超过这个时间后，Ralph 会在开始下一轮前主动停下
- `--max-failures-per-story <n>`
  同一个 story 连续失败达到阈值后，Ralph 会把它加入“当前 run + 后续 run”的自动跳过列表，并继续找别的 story

如果你不显式传 `--max-failures-per-story`，当前默认值是 `3`。  
自动跳过状态会写进项目根目录的 `.ralph-run-state.json`。  
它不会修改 `prd.json` 里的 `passes`；如果你已经修好了这个 story，可以用 `--retry-story` 把它重新放回执行队列。

如果你想先确认 Ralph 当前会怎么排队执行 story，可以先执行：

```bash
ralph --dry-run
```

它会把“可执行队列”和“被跳过队列”分开显示出来，适合真正开跑前先做一次人工确认。
如果当前配置了 token / 成本预算，它也会一并把预算护栏打印出来。

如果你要控制预算，这一版使用的是“近似 token / 成本估算”，不是官方账单接口：

- token 估算 = `字符数 / charsPerToken`
- 成本估算 = 按 input/output token 单价折算

如果你使用 `--max-total-cost-usd`，记得同时配置至少一个单价：

```json
"budget": {
  "maxTotalTokens": 12000,
  "maxTotalCostUsd": 2.5,
  "charsPerToken": 4,
  "inputCostPer1kTokensUsd": 0.003,
  "outputCostPer1kTokensUsd": 0.015
}
```

真正开始执行后，Ralph 还会把“截至当前轮的预算估算”追加进 `progress.txt`，这样你回看运行日志时，不需要自己再手工换算。

## 路径 B：OpenSpec + Superpowers + Ralph

这是更完整的流程，把需求设计、评审、转换和执行串成一条线。

完整说明请看 [PIPELINE_GUIDE.md](/D:/project/AI-Coding/ralph-longtask/doc/PIPELINE_GUIDE.md)。  
如果你要验证主线二是否真的能按步骤跑通，再看 [PIPELINE-SMOKE-CHECKLIST.md](/D:/project/AI-Coding/ralph-longtask/doc/PIPELINE-SMOKE-CHECKLIST.md)。  
这里先给一个总览。

### 5 个阶段

```text
spec -> review -> convert -> execute -> archive
```

含义：

- `spec`
  由 OpenSpec 产出 `proposal.md`、`specs/`、`design.md`、`tasks.md`
- `review`
  由 Superpowers 审查并完善 OpenSpec 文档，再生成 `tasks/prd-*.md`
- `convert`
  由 `ralph` skill 把 PRD Markdown 转成 `prd.json`
- `execute`
  把执行正式交给 Ralph CLI
- `archive`
  由 OpenSpec `archive` 完成提案归档

### 这条路径里几个组件的分工

- OpenSpec
  负责规格输入和归档
- Superpowers
  负责 review、文档收紧和 PRD 生成
- Ralph pipeline CLI
  负责状态机、artifact 探测和 gate 提示
- `skills/pipeline`
  负责对话里的 gate 管理和审批提醒

### 重要边界

当前实现里：

- CLI 会检测 OpenSpec 和 Superpowers 是否可用
- CLI 会根据 artifact 判断当前停在哪个 gate
- CLI 会在 blocked 时明确提示下一步该由 OpenSpec、Superpowers 还是 `ralph` skill 接手
- CLI 不会替你自动完成对话式 OpenSpec / Superpowers 全流程审批

也就是说：

- `ralph pipeline` 管“状态和 gate”
- `pipeline` skill 管“对话和审批”

---

## 配置搜索规则

Ralph 会从当前工作目录向上查找 `ralph.config.json`。

例如你在：

```text
/project/src/components
```

执行命令，它会按顺序尝试：

1. `/project/src/components/ralph.config.json`
2. `/project/src/ralph.config.json`
3. `/project/ralph.config.json`

找到第一个后停止。

如果一个都没找到，就使用默认值。

---

## 环境变量覆盖

所有配置都可以通过 `RALPH_` 前缀的环境变量覆盖。

常见映射：

| 环境变量 | 对应配置 |
|----------|----------|
| `RALPH_PRD_PATH` | `prdPath` |
| `RALPH_PROGRESS_PATH` | `progressPath` |
| `RALPH_MAX_ITERATIONS` | `maxIterations` |
| `RALPH_COOLDOWN_SECONDS` | `cooldownSeconds` |
| `RALPH_PERMISSIONS_MODE` | `permissionsMode` |
| `RALPH_CLAUDE_MAX_TURNS` | `claude.maxTurns` |
| `RALPH_PROMPTS_AGENT_INSTRUCTION_PATH` | `prompts.agentInstructionPath` |
| `RALPH_PROMPTS_EXTRA_INSTRUCTIONS` | `prompts.extraInstructions` |
| `RALPH_PROMPTS_STRICT_SINGLE_STORY` | `prompts.strictSingleStory` |
| `RALPH_VALIDATION_CHECK_GIT_COMMIT` | `validation.checkGitCommit` |
| `RALPH_VALIDATION_PATCH_PRD_PASSES` | `validation.patchPrdPasses` |
| `RALPH_VALIDATION_VALIDATE_PRD_SCHEMA` | `validation.validatePrdSchema` |
| `RALPH_VALIDATION_ACCEPTANCE_COMMANDS_TYPECHECK` | `validation.acceptanceCommands.typecheck` |
| `RALPH_VALIDATION_ACCEPTANCE_COMMANDS_TESTS` | `validation.acceptanceCommands.tests` |
| `RALPH_VALIDATION_ACCEPTANCE_COMMANDS_BROWSER` | `validation.acceptanceCommands.browser` |

示例：

```bash
RALPH_MAX_ITERATIONS=20 RALPH_COOLDOWN_SECONDS=0 ralph
```

---

## 常见问题

### 1. 为什么 Ralph 不直接执行 Markdown PRD？

因为 Ralph 的执行输入是 `prd.json`。  
Markdown PRD 更适合阅读、评审和修改，`prd.json` 更适合自动执行。

### 2. `prd` skill 和 `ralph` skill 有什么区别？

- `prd` skill：把功能描述整理成 Markdown PRD
- `ralph` skill：把 Markdown PRD 转成 `prd.json`

### 3. UI story 为什么总是不过？

常见原因：

- story 太大
- story 包含 `Verify in browser using dev-browser skill`
- 但配置里没有 `validation.acceptanceCommands.browser`

### 4. `ralph --resume` 和 `ralph pipeline resume` 有什么区别？

- `ralph --resume`
  恢复 Ralph 的执行循环
- `ralph pipeline resume`
  恢复 pipeline 状态机

---

## 推荐阅读顺序

1. [README.md](/D:/project/AI-Coding/ralph-longtask/README.md)
2. 当前这份用户指南
3. [PIPELINE_GUIDE.md](/D:/project/AI-Coding/ralph-longtask/doc/PIPELINE_GUIDE.md)
4. [ralph-cli.md](/D:/project/AI-Coding/ralph-longtask/doc/ralph-cli.md)
