# Check Ref

A read-only agent skill for auditing references already present in Word (`.docx`) and LaTeX (`.tex` + `.bib`) manuscripts.

It checks reference metadata against OpenAlex and Semantic Scholar, uses independent web search to confirm bibliography records, and compares each existing citation claim with the cited paper's available abstract. It writes a Markdown audit report and does not modify the manuscript or bibliography file.

## Install

Copy this folder into your agent's skills directory as `check-ref`, then restart or refresh the agent so it can discover `SKILL.md`.

For Codex on Windows, the user-level directory is typically `%USERPROFILE%\.codex\skills\check-ref\`.

## Use

Ask the agent to audit an existing `.docx`, or a LaTeX `.tex` file together with its `.bib` bibliography. The skill requires internet access to academic metadata services and web search. API keys are optional; anonymous requests may be rate limited.

## Important scope

- Bibliography verification and claim-support review are separate checks.
- Claim support is assessed from available abstracts; it is not a full-text, page-number, table, or figure audit.
- Missing metadata or abstracts should remain unverified rather than being treated as proof that a citation is false.
- The skill is read-only with respect to the manuscript and `.bib` file.

## Local credentials

Optional API credentials belong in `scripts/api_keys.local.json` on the user's machine. That file is excluded by `.gitignore`; never commit personal API keys.
