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
  again. A stop asked during the planning (capture.stop; the hook, hence Esc, starts only after the planning) is seen
  before each call and after the last attempt and ends the capture 'interrupted', redone once in the same way; a call
  already running is not cut.
- Seen in the checks after the change (generic prompt, seed 1, returns-v3/m2-check.md): both models pass at the first
  attempt and every gap is 0. Varying the call one factor at a time (no schema, no length, no format, without
  "the first 0", [character, gap] pairs with the characters fixed) gave one repeated value (0, 1, 20, 50, 100) or a
  count (0, 1, 2, ...) on both models; no form gave gaps that change with the characters. The researcher kept option A
  (decision of 2026-11-04): the plan is the model's own, constant gaps included, and is typed as planned.
- Seen in the check at the production length (2026-11-06, returns-v3.5/m2-check-v3.5.md; commit 662d3b7; generic
  prompt, seeds 1-3, essays of 1049-1607 characters): all 6 plans passed at the first attempt. Their gaps were all 0
  (7B seeds 1 and 2, 14B seed 1), 0 then one value (14B seed 2: 20 ms; seed 3: 15 ms), or a count in steps of 5 ms
  (7B seed 3: 0, 10, 15, ... up to 5990 ms, 1198 gaps summing to 3591 s, about 60 min of typing for one essay). The
  researcher kept option A (decision of 2026-11-06): such a plan is typed as planned, with no bound on its length.
- Whether a value can be affected: yes. Every M2 value comes from plans made this way; the paper reports this entry
  and that the plans' gaps in the checks were constant or a count, never changing with the characters.

## D2. 2026-11-06 / block 0 (L0/e0-e3) / L0 typed with the IME open, set aside by hand and typed again

- What happened: block 0 ran from 07:32:37Z to 07:37:45Z; the runlog gave done 4 (one attempt each, no stop) and wrote
  its footer. tools/l0check.py (the analysis's reader, prototype_v7_run.load_capture live=True) gave STOP on all 4
  essays: the hook and page layers did not hold the same presses. Counting kinds only (no key, text or time), the hook
  counted 132-212 presses per essay and the page 8-54, the page's other key downs falling outside the counted kinds.
  The researcher said the IME (Japanese input) was used while typing: the capture closes the IME before the signal,
  and it was opened again during the essays, so the page saw the presses as IME processing and the hook as keys.
  The researcher also said some keys were typed before "Recording." was shown. The essays were also far shorter than
  the conditions' 150-250 words.
- What was done (the researcher's decision of 2026-11-06): the runlist refuses to set aside a block with a footer, so
  C:\work\v7-captures\L0\ and runlog.b0.ndjson were moved by hand to C:\work\v7-captures\stopped-20261106T074342Z-L0\
  (nothing deleted; kept on the work PC, not returned), and block 0 was run again with the same command, the IME kept
  closed, in English, 150-250 words per essay. The set-aside captures enter no value.
- Whether a value can be affected: yes for L0, L1_idle and L1_load (L0 is the second session of typing on these 4
  prompts by the researcher, who had seen them once). No other series is concerned.
