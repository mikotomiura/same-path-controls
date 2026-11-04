"""M1: type an LLM's output as its tokens arrive (capture spec section 4, conditions series.M1 with amendments B6, B11,
B13 and B14; the copy prototype_v7_20d5cbc.json of the handoff overrides the spec where they differ).

Schedule (amendment B13): with a_k the arrival of token k (non-empty tokens only; an empty token is skipped and
counted) and n_k its characters (Unicode code points after NFC), the characters of token k are wished at
a_{k+1} + j (a_{k+1} - a_k) / n_k, j = 0..n_k-1, with a_{K+1} the end of generation (the arrival of the final 'done'
message). Every character is thus wished no earlier than it is known. Times are relative to a_1.
The executable plan is the running maximum of the wished times (character order kept, equal times allowed, nothing
cleaned: amendment B14). A press whose plan is earlier than the moment the previous press was sent waits (queue; no
skipping, no catching up): its sent time is the later of its plan and the previous send. The wish, the plan, the sent
time and the wait are kept for every press.

Characters that the schedule cannot hold (outside printable ASCII; see events.typable) are skipped and counted;
their slot in time is left empty, so the other characters keep their times. A line break is Enter.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import unicodedata
from dataclasses import dataclass, field

from capture import events


def n_chars(token: str) -> int:
    return len(unicodedata.normalize("NFC", token))


def token_keys(token: str) -> tuple[list[str | None], int]:
    """One entry per code point of the NFC token: the key to type, or None when it is skipped. A \\r\\n pair is
    one Enter and its \\r a skipped slot (so that the slots still match the code points). Returns the keys and the
    number skipped (\\r of \\r\\n not counted as skipped)."""
    s = unicodedata.normalize("NFC", token)
    keys: list[str | None] = []
    skipped = 0
    for i, c in enumerate(s):
        if c == "\n":
            keys.append("Enter")
        elif c == "\r":
            keys.append(None)
            if not (i + 1 < len(s) and s[i + 1] == "\n"):
                skipped += 1  # a lone \r
        elif events.typable(c):
            keys.append(c)
        else:
            keys.append(None)
            skipped += 1
    return keys, skipped


def spread(a_k_ms: float, a_next_ms: float, n: int) -> list[float]:
    """The wished times of a token's n characters: from a_{k+1} over a_{k+1} - a_k."""
    step = (a_next_ms - a_k_ms) / n
    return [a_next_ms + j * step for j in range(n)]


@dataclass
class Planner:
    """Builds the plan as tokens arrive. feed() a token's arrival; finish() at the end of generation. Each call
    returns the presses that became known: dicts with key, wished_ms, plan_ms, token."""
    a1_ns: int | None = None
    last_plan_ms: float = float("-inf")
    pending: tuple[int, str] | None = None   # (arrival ns, token) of the last non-empty token
    k: int = 0
    empty_tokens: int = 0
    skipped_chars: int = 0
    moved: int = 0                            # presses whose plan is later than their wish
    shifts_ms: list = field(default_factory=list)

    def _ms(self, t_ns: int) -> float:
        return (t_ns - self.a1_ns) / 1e6

    def _release(self, next_ns: int) -> list[dict]:
        if self.pending is None:
            return []
        t_k, tok = self.pending
        keys, skipped = token_keys(tok)
        self.skipped_chars += skipped
        out = []
        for key, w in zip(keys, spread(self._ms(t_k), self._ms(next_ns), len(keys))):
            if key is None:
                continue
            plan = max(w, self.last_plan_ms)
            if plan > w:
                self.moved += 1
                self.shifts_ms.append(plan - w)
            self.last_plan_ms = plan
            out.append({"key": key, "wished_ms": w, "plan_ms": plan, "token": self.k})
        return out

    def feed(self, t_ns: int, token: str) -> list[dict]:
        if n_chars(token) == 0:
            self.empty_tokens += 1
            return []
        if self.a1_ns is None:
            self.a1_ns = t_ns
        out = self._release(t_ns)
        self.k += 1
        self.pending = (t_ns, token)
        return out

    def finish(self, end_ns: int) -> list[dict]:
        out = self._release(end_ns)
        self.pending = None
        return out


def plan_offline(arrivals: list[tuple[int, str]], end_ns: int) -> tuple[list[dict], Planner]:
    """The whole plan from recorded arrivals (tests, and a check of the live plan)."""
    p = Planner()
    out = []
    for t, tok in arrivals:
        out += p.feed(t, tok)
    out += p.finish(end_ns)
    return out, p


class Typist(threading.Thread):
    """Types the presses put on its queue at a1 + plan, never before the previous press was sent."""

    def __init__(self, injector, guard_hwnd: int | None, stop):
        super().__init__(daemon=True)
        self.q: queue.Queue = queue.Queue()
        self.inj, self.guard, self.stop_flag = injector, guard_hwnd, stop
        self.records: list[dict] = []
        self.a1_ns: int | None = None
        self.stopped_by_esc = False
        self.stopped_focus_lost = False
        self.error: BaseException | None = None
        self.abort = threading.Event()  # set by run() when the stream fails: nothing more is typed

    def run(self):
        from capture.inject import wait_until
        fg = self.inj.user32.GetForegroundWindow
        try:
            i = 0
            while True:
                item = self.q.get()
                if item is None:
                    return
                qlen = self.q.qsize()
                planned = self.a1_ns + round(item["plan_ms"] * 1e6)
                stopped = wait_until(planned, self.stop_flag)  # a stop during a long wait ends it at once
                if self.abort.is_set():
                    return
                if stopped or (self.stop_flag is not None and self.stop_flag.is_set()):
                    self.stopped_by_esc = True
                    return
                if self.guard is not None and fg() != self.guard:
                    self.stopped_focus_lost = True
                    return
                sent, ret, r, n = self.inj.send(item["key"])
                self.records.append({"i": i, "token": item["token"], "wished_ms": item["wished_ms"],
                                     "plan_ms": item["plan_ms"], "planned_ns": planned, "sent_ns": sent,
                                     "returned_ns": ret, "wait_ns": sent - planned, "queue_len": qlen,
                                     "sendinput_returned": r, "n_inputs": n, "ok": r == n})
                i += 1
        except BaseException as e:  # noqa: BLE001 - kept and raised by the session, never swallowed
            self.error = e


def run(model: str, prompt: str, seed: int, method: str, guard_hwnd: int | None, stop, llm_log) -> dict:
    """Generate and type at once. llm_log: an open text file for llm.ndjson. Returns the plan, the inject records
    and the counts."""
    from capture import ollama
    from capture.inject import Injector
    inj = Injector(method)
    typist = Typist(inj, guard_hwnd, stop)
    planner = Planner()
    plan: list[dict] = []
    body = {"model": model, "prompt": prompt, "options": {"temperature": ollama.TEMPERATURE, "seed": seed},
            "keep_alive": "10m"}
    llm_log.write(json.dumps({"type": "request", "body": body, "time_ns": time.time_ns()}) + "\n")
    final = None
    with inj:
        typist.start()
        try:
            with ollama.open_stream(body) as r:
                for k, line in enumerate(r):
                    t = time.perf_counter_ns()
                    d = json.loads(line)
                    if "error" in d:  # ollama's error line: the generation failed (never counted as an empty token)
                        llm_log.write(json.dumps({"type": "error", "n": k, "recv_ns": t, "error": d["error"]}) + "\n")
                        raise RuntimeError(f"ollama: {d['error']}")
                    tok = d.get("response", "")
                    llm_log.write(json.dumps({"type": "chunk", "n": k, "recv_ns": t, "token": tok,
                                              "n_chars": n_chars(tok), "done": bool(d.get("done"))}) + "\n")
                    if d.get("done"):
                        final = {kk: v for kk, v in d.items() if kk not in ("response", "context")}
                        # a final line that also carries text: its token arrives at t and ends at t (spread over 0 ms)
                        new = planner.feed(t, tok) + planner.finish(t) if n_chars(tok) else planner.finish(t)
                    else:
                        new = planner.feed(t, tok)
                    if typist.a1_ns is None and planner.a1_ns is not None:  # also when the only text is on the done line
                        typist.a1_ns = planner.a1_ns
                    for x in new:
                        typist.q.put(x)
                    plan += new
                    if d.get("done") or not typist.is_alive():
                        break
        except BaseException:
            typist.abort.set()  # the stream failed: the queued presses are not typed (review of 2026-11-04)
            typist.q.put(None)
            typist.join(5)
            raise
        typist.q.put(None)
        typist.join()
    llm_log.write(json.dumps({"type": "final", **(final or {})}) + "\n")
    if typist.error is not None:
        raise typist.error
    waits = [x["wait_ns"] for x in typist.records]
    return {"plan": plan, "records": typist.records,
            "counts": {"tokens": planner.k, "empty_tokens": planner.empty_tokens,
                       "skipped_chars": planner.skipped_chars, "planned": len(plan),
                       "typed": len(typist.records), "not_sent": sum(not x["ok"] for x in typist.records),
                       "plan_moved": planner.moved,
                       "plan_moved_median_ms": (sorted(planner.shifts_ms)[len(planner.shifts_ms) // 2]
                                                if planner.shifts_ms else 0.0),
                       "queue_max": max((x["queue_len"] for x in typist.records), default=0),
                       "wait_max_ms": max(waits, default=0) / 1e6,
                       "generation_done": final is not None,
                       "stopped_by_esc": typist.stopped_by_esc, "stopped_focus_lost": typist.stopped_focus_lost}}
