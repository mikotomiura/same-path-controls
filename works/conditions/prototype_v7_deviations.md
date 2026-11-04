# Prototype v7: deviations during the captures

Conditions: works/conditions/prototype_v7.json (amendment B17). From the production plan to the analysis the conditions
are frozen; every deviation from them, or from the steps that carry them out, is written here, dated and with its
reason, in a commit of its own. The analysis (works/prototype_v7_run.py) records this file's SHA-256 and commit in
record.json, and the paper reports every entry.

Each entry: date (UTC) / block and items concerned / what happened / what was done / whether a value can be affected.

## D1. 2026-11-04 / M2_8B and M2_14B (all blocks, all items) / how the plan is asked

- What happened: before any production capture, the M2 check on the work PC (generic prompt only, no KUPA prompt)
  found that the plan as the conditions word it (series "M2_8B and M2_14B" and "prompt": one call, JSON
  {text, gaps_ms} with format json) does not finish on either model: qwen2.5 7B and 14B wrote gaps_ms as 0, 0, 0...
  up to ollama's HTTP timeout of 600 s, which raised out of capture.llm_plan instead of counting a failed attempt;
  with num_predict 8192 all 6 attempts were cut at the limit and were not JSON (returns-v3/m2-check.md).
- What was done (the researcher's decision of 2026-11-04, option A; commit 2d1baf8): one attempt is two calls with the
  same seed, (1) the essay with the M1 prompt and no format, (2) the gaps of its characters with
  capture.llm_plan.GAPS_PROMPT and a JSON schema whose gaps_ms holds exactly len(text) integers >= 0. The checks
  (len(gaps_ms) == len(text), the first 0, finite, non-negative), the 3 attempts with seeds seed + 1000 a, the counted
  failure without replacement and the LLM stopped before typing are unchanged. The essay is typed with its outer
  whitespace stripped (the gaps are asked for and checked on the stripped text). Each call waits up to 1800 s; a
  read timeout (no answer within it) is a failed attempt.
- Amended the same day, before any production capture (the researcher's decision 1 of
  .steering/20261104-v7-prep-v3-returns/decisions.md, judgement 7, after the Mac review of returns-v3; commit 9cd8547):
  any other error of ollama (connection refused or reset, HTTP 4xx or 5xx, an answer cut short or not JSON) is not
  the model's failure: no attempt is counted, the capture ends 'failed' and the runlist redoes it once, planning
  again. A stop (Esc or capture.stop) during the planning ends the capture 'interrupted', redone once in the same way.
- Seen in the checks after the change (generic prompt, seed 1, returns-v3/m2-check.md): both models pass at the first
  attempt and every gap is 0. Varying the call one factor at a time (no schema, no length, no format, without
  "the first 0", [character, gap] pairs with the characters fixed) gave one repeated value (0, 1, 20, 50, 100) or a
  count (0, 1, 2, ...) on both models; no form gave gaps that change with the characters. The researcher kept option A
  (decision of 2026-11-04): the plan is the model's own, constant gaps included, and is typed as planned.
- Whether a value can be affected: yes. Every M2 value comes from plans made this way; the paper reports this entry
  and that the plans' gaps were constant in the checks.
