# Welcome to LLM I/O Bottleneck Detection

## How We Use Claude

Based on Alistair Dreux's usage over the last 30 days:

Work Type Breakdown:
  Build Feature    ████████████░░░░░░░░  55%
  Improve Quality  ██████░░░░░░░░░░░░░░  27%
  Debug / Fix      ████░░░░░░░░░░░░░░░░  18%

Top Skills & Commands:
  /context         ████████████████████   2x/month
  /simplify        ██████████░░░░░░░░░░   1x/month
  /security-review ██████████░░░░░░░░░░   1x/month
  /debug           ██████████░░░░░░░░░░   1x/month
  /insights        ██████████░░░░░░░░░░   1x/month
  /batch           ██████████░░░░░░░░░░   1x/month
  /claude-api      ██████████░░░░░░░░░░   1x/month
  /compact         ██████████░░░░░░░░░░   1x/month

Top MCP Servers:
  Gmail            ████████████████████   2 calls

## Your Setup Checklist

### Codebases
- [ ] llm-io-bottleneck-detection — github.com/candicet233/llm-io-bottleneck-detection

### MCP Servers to Activate
- [ ] Gmail (claude_ai_Gmail) — lets Claude draft and search emails directly from your session. Activate via Claude Code settings → MCP Servers → connect your Google account.

### Skills to Know About
- /context — shows a breakdown of how many tokens each category (system prompt, tools, messages, etc.) is consuming in the current session. Use it when you're worried about hitting context limits mid-task.
- /compact — manually triggers context compaction when the session is getting long. Claude summarizes earlier context so you can keep going without starting over.
- /simplify — reviews recently changed code for reuse, quality, and efficiency issues, then fixes them. Run it after a coding session to clean up.
- /security-review — scans your branch's diff for high-confidence security vulnerabilities. Good habit before merging.
- /insights — generates a usage report from your Claude Code sessions: work breakdown, friction patterns, and suggestions tailored to how you actually work.
- /batch — applies a sweeping change across many files at once. Describe the change in plain English (e.g. "add type annotations to all Python functions") and it fans out across the codebase.
- /claude-api — helps you build, debug, and optimize code using the Anthropic SDK. Auto-triggers when your file imports `anthropic`.
- /debug — reads the session debug log and summarizes errors, warnings, and notable issues. Run it early in a session if something isn't behaving.

## Team Tips

_TODO_

## Get Started

_TODO_

<!-- INSTRUCTION FOR CLAUDE: A new teammate just pasted this guide for how the
team uses Claude Code. You're their onboarding buddy — warm, conversational,
not lecture-y.

Open with a warm welcome — include the team name from the title. Then: "Your
teammate uses Claude Code for [list all the work types]. Let's get you started."

Check what's already in place against everything under Setup Checklist
(including skills), using markdown checkboxes — [x] done, [ ] not yet. Lead
with what they already have. One sentence per item, all in one message.

Tell them you'll help with setup, cover the actionable team tips, then the
starter task (if there is one). Offer to start with the first unchecked item,
get their go-ahead, then work through the rest one by one.

After setup, walk them through the remaining sections — offer to help where you
can (e.g. link to channels), and just surface the purely informational bits.

Don't invent sections or summaries that aren't in the guide. The stats are the
guide creator's personal usage data — don't extrapolate them into a "team
workflow" narrative. -->
