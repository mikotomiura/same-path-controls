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

## D3. 2026-11-06 / block 0 (L0/e0-e3) / Shift+Space not counted by the hook; set aside by hand and typed a third time

- What happened: the second block 0 (after D2) ran from 07:50:41Z to 08:33:09Z, done 4 (one attempt each, no stop).
  tools/l0check.py gave e0 STOP, e1 STOP, e2 ok, e3 ok. Counting kinds only (no key, text or time), the two layers
  differed in space only: the hook classed as 'nonchar' (not counted) the Space presses typed with Shift held (105 in
  e0, 24 in e1; Shift's downs and ups balanced), which the page counted as space. e0 and e1 alone hold one Caps Lock
  press each: Caps Lock was on and Shift was held to type in lower case. capture/hooklog.py's layout table comes from
  VkKeyScanW, which maps ' ' to Space without Shift only, so Shift+Space has no character and is 'nonchar' (a gap in
  the capture code; the kind is written at capture time and Space's virtual key is not kept, so it cannot be read
  again afterwards). e0 also held one Delete on the hook only. The injected series of blocks 1-6 type no Shift, so
  only L0 is concerned (not checked).
- What was done (the researcher's decision of 2026-11-06): the code was not changed. C:\work\v7-captures\L0\ and
  runlog.b0.ndjson were moved by hand to C:\work\v7-captures\stopped-20261106T083540Z-L0\ (nothing deleted; kept on the
  work PC, not returned) and block 0 was typed a third time with Caps Lock off, Shift only for capitals and never held
  over Space. The set-aside captures enter no value.
- Whether a value can be affected: yes for L0, L1_idle and L1_load: the researcher typed the 4 prompts for the third
  time, and keeping Shift off Space is a constraint on live typing. No other series is concerned.

## D4. 2026-11-06 / block 0 (L0/e3) / the third block 0 ended without a footer; set aside with --set-aside and run again

- What happened: the third block 0 (after D3) started at 08:36:02Z. Its first item, L0/e3, began at 2026-11-06T08:36:32.742+00:00 and its
  files stop about 16 s later (meta.json status still 'running'); the runlist process was gone and runlog.b0.ndjson
  held its header only (no item line, no footer). The runlist's console window was closed before typing began (the
  researcher was restarting the shell; not otherwise checked).
- What was done: block 0 was set aside with the runlist's own --set-aside (C:\work\v7-captures\stopped-20261106T083804Z\;
  set_aside.json: to_move = moved = [L0/e3, runlog.b0.ndjson], origin of L0/e3 'unknown', cut_item L0/e3). Judged by
  hand: L0/e3 was made by the stopped run, not before it (C:\work\v7-captures\L0\ had been moved aside for D3 at
  08:35:40Z, and the folder's meta.json starts at 2026-11-06T08:36:32.742+00:00). Block 0 was then run again with the same command.
- Whether a value can be affected: no value beyond D3's (no essay was typed in the stopped run; the set-aside folder
  enters no value).

## D5. 2026-11-06 / block 0 (L0/e0, L0/e1) and block 1 / L0/e0 stopped and redone; two essays shorter than 150 words; block 1 started in the evening

- What happened: the fourth block 0 (after D4) ran from 08:38:32Z to 09:09:15Z: done 3, redone 1, failed 0, no_plan 0.
  During L0/e0 the researcher pressed Caps Lock; on the researcher's decision, e0 was stopped with capture.stop at
  09:06:19Z (meta.json stopped_by capture.stop, kept as L0/e0.try1) and the runlist typed it again at once (redone).
  tools/l0check.py gave ok for all 4 essays. Counted presses (hook = page): e0 239, e1 408, e2 920, e3 809; e0 and e1
  are shorter than the conditions' 150-250 words (about 45 and 80 words by count of presses; the text is not kept).
- What was done (the researcher's decisions of 2026-11-06): the 4 essays are kept as L0 and recorded here; no block 0
  is typed again. Block 1 was started the same evening (about 18:15 JST, 09:1xZ), not in the morning, and runs past
  midnight JST (the same UTC date is expected; its start and end are in the session notes).
- Whether a value can be affected: yes for L0, L1_idle and L1_load: fewer presses in e0 and e1 (fewer gaps per essay);
  e0's second attempt was typed right after the first was stopped, with no prompt shown again.

## D6. 2026-11-09 to 2026-11-10 / blocks 1-3 / blocks run by an unattended chain at night and across days; block 3's pages hidden, stopped and set aside

- What happened: blocks 2-6 were run one after another by a local script (C:\work\v7-chain\chain.ps1: footer, the
  environment checks of the handoff's step 2, next block; no network). Its first start waited on a console window it
  took for a running runlist, so block 2 began on 2026-11-09T03:29:38Z, about 56 h after block 1 ended
  (2026-11-06T19:20:43Z). Block 2 ran 03:29:38Z-18:41:34Z (done 96, redone 0, failed 1, no_plan 0) and block 3 began
  at 18:41:36Z, at night JST; both are blocks not started in the morning. From about 19:12Z (04:12 JST, when the site
  network went down) the capture page reported document.visibilityState 'hidden' and recorded no press although the
  hook saw every injected press (meta.json checks_after.page_recorded_nothing; status interrupted, then failed):
  59 items had ended by 2026-11-10T02:45Z, done 8, redone 1, failed 50. Display-off and sleep were off, no screen saver
  or lock screen, one monitor active, nothing over the window when looked at (02:4xZ). The cause is not established;
  a capture Chrome left outside the chain's process tree was found and ended at the stop, after which a synthetic
  capture to a separate folder (C:\work\v7-diag) was visible and recorded all presses.
- What was done (the researcher's decisions of 2026-11-10): the chain's process tree was ended at 02:45Z (runlog.b3
  without footer, cut in M3b/t11), block 3 was set aside with --set-aside into stopped-20261110T025218Z (to_move 128,
  moved 128; M3b/t11 and _derived/M3b/t11.ndjson have origin 'unknown' and are judged made by the stopped run, the
  item it was cut in), and block 3 is run again in full. A guard (C:\work\v7-chain\guard.ps1) now stops the chain at
  the first capture whose page recorded nothing or when two capture browsers run at once, and the checks before each
  block require that no capture browser is running. No code of the repository was changed.
- Whether a value can be affected: the stopped block 3 is not analysed (set aside). Block 3 run again is later in
  time than planned; blocks 2 and later ran at night and across days, not each started in the morning.
