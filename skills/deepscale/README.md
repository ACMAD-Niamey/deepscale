# deepscale agent skill

An [Agent Skill](https://agentskills.io/) that teaches AI coding agents how to use the `accord-deepscale` package: the core verbs (`downscale`, `optimize`, `calibrate`, `ensemble`, `skill`, `seasonal_mme`), method/metric/strategy registries, tercile leakage discipline, plotting/reporting, and rosetta integration.

A skill is just a directory of Markdown and example files — no code runs from it. The agent reads `SKILL.md` when a task looks relevant (matching on the `name`/`description` frontmatter) and pulls in `references/` and `examples/` files only as needed, so the skill costs almost nothing until it's actually used.

## Layout

```
deepscale/
├── SKILL.md                          # entry point: frontmatter + core instructions
├── references/
│   ├── api.md                        # core forecasting verbs + generalized Index
│   ├── analog-completion.md          # SMPG: analog selection, scenario completion, climate positioning, calendar utils
│   ├── methods.md                    # downscale methods, strategies, pooling/combining, CV schemes
│   ├── metrics-and-terciles.md       # metric semantics + leakage discipline + significance tools
│   ├── plotting-reporting.md         # maps, choropleths, scenario plots, SVSLRF PDFs, export
│   └── troubleshooting.md            # error→cause table, env setup
└── examples/                         # runnable scripts
```

## Loading it

**Claude Code (this repo):** copy or symlink the skill into the project's `.claude/skills/` directory, which Claude Code auto-discovers:

```bash
mkdir -p .claude/skills
ln -s ../../skills/deepscale .claude/skills/deepscale
```

**Claude Code (all your projects):** put it in your personal skills directory instead:

```bash
ln -s "$(pwd)/skills/deepscale" ~/.claude/skills/deepscale
```

Then Claude uses it automatically when a task involves deepscale, or invoke it explicitly with `/deepscale`.

**Claude Agent SDK:** point the SDK at a directory containing the skill (e.g. via the `settingSources`/skills configuration) or copy it into the workspace's `.claude/skills/`.

**Claude API:** upload the directory via the Skills API (`/v1/skills`) and attach it to a container-use request; see the Anthropic docs for the current mechanism.

**Codex, OpenCode, and other agent coding frameworks:** the format (SKILL.md + frontmatter) is harness-agnostic by design, and any agentskills.io-compatible runtime can consume it — copy the `deepscale/` directory into wherever that framework discovers skills (e.g. Codex and OpenCode both read `AGENTS.md`-style project instructions and can be pointed at skill directories). For any agent without native skill support, simply add an instruction to its project config (`AGENTS.md`, system prompt, etc.) to read `skills/deepscale/SKILL.md` before working with deepscale.

**Humans:** the same files work as documentation — start with `SKILL.md`.

## Validating

```bash
skills-ref validate skills/deepscale   # from https://github.com/agentskills/agentskills
```

## Maintenance

When the public API, method registry, or conventions change, update the matching reference file (and `SKILL.md` if a core behavior changed). Everything in here was extracted from the source at the time of writing — treat drift as a bug.

Companion skill: [rosetta](https://github.com/accord-research/rosetta/tree/main/skills/rosetta) documents the data-acquisition side of the pipeline.
