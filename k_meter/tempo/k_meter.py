"""K-Meter — TEMPO PRO STICK perfection from the purple meter.

TAP K once. The K press itself is BLOCKED — it would fire a button shot —
and instead Pro Stick Down is pressed and held: that hold IS the tempo,
the shot starts, and the horizontal PURPLE meter appears. At the perfect
moment the stick is flicked UP (Pro Stick Down released and Pro Stick Up
pressed in one atomic input batch, then held `flick_hold_ms` so 2K reads
the flick). The moment is read live off the meter: the magenta fill
racing left-to-right into the GREEN TICK at the bar's tip. Flick = fill
front reaches the tick (scheduled predictively from the fill speed,
sub-ms injection).

    tap K -> pro stick DOWN, tempo running -> purple bar fills ->
          vision predicts fill-front @ green tick -> flick UP -> hold

Keyboard binds (config `pro_stick_down_key` / `pro_stick_up_key`, same
defaults as precision_timer.py): Pro Stick Down = '.', Pro Stick Up = ';'

Meter signature (measured from real 2K26 screenshots):
    fill  : magenta H 150-151, bar ~128x24 px at 1600x900
    tick  : green H 53-65, fixed at the bar's right tip
    track : dark, to the right of the fill

Fallbacks (in order): meter lost mid-fill -> flick immediately;
no meter and nothing magenta on screen within `no_meter_ms` -> tempo of
`hold_ms` measured from the press (pressed without a shot); hard safety
at `meter_max_ms` so the stick can never stay held forever, and an
all-keys-up on exit so neither stick key is ever left down.

Requires:  pip install keyboard opencv-python mss numpy
Run as administrator (keyboard hook + game window focus).
Do NOT run this at the same time as precision_timer.py /
precision_timer_pad.py — they register the same F-key hotkeys and drive
the same pro stick keys.

Controls (all prefixed by `hotkey_prefix`, default Ctrl+Alt — bare
F-keys collide with other tools and silently retune latency mid-game):
    Ctrl+Alt+F5  pause / resume (K behaves normally while paused)
    Ctrl+Alt+F7  latency +10 ms (EARLIER)   Ctrl+Alt+F6  -10 ms (LATER)
    Ctrl+Alt+F4  latency  +3 ms (EARLIER)   Ctrl+Alt+F3   -3 ms (LATER)
    Ctrl+Alt+F8  status   Ctrl+Alt+F10  reset stats   Ctrl+Alt+F9  quit

meter_offset_ms: ms after (positive) or before (negative) the fill
reaches the tick. meter_lead_ms compensates capture/display latency.
Both hot-reload from k_meter.json while running.

NB latency_ms was tuned against a K key-up. A stick flick goes through
2K's stick-input path instead, so expect to re-trim it (Ctrl+Alt+F3/F4)
once from whatever value the button build settled on.
"""

from __future__ import annotations

import copy
import ctypes
import ctypes.wintypes as wt
import faulthandler
import json
import logging
import math
import os
import threading
import time
import traceback

import keyboard

try:
    import numpy as _np
    import cv2 as _cv2
    from mss import MSS as _MSS
    _VISION_OK = True
except Exception:
    _VISION_OK = False

__version__ = "1.4"      # 1.4: tempo pro stick output (was button K-up)
#                          1.3: alpha-beta fill tracker, banded BGR search

# --------------------------------------------------------------------------- #
# Paths / crash logging / config
# --------------------------------------------------------------------------- #
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _crash_log = open(os.path.join(_script_dir, "crash.log"), "w")
    faulthandler.enable(file=_crash_log, all_threads=True)
except OSError:
    _script_dir = "."

CONFIG_PATH = os.path.join(_script_dir, "k_meter.json")

DEFAULT_CONFIG = {
    # Trigger only. The press is BLOCKED and never reaches 2K — sending it
    # would start a button shot alongside the stick one, and the button
    # meter is not the meter this tool reads.
    "action_key": "k",       # tap once
    # The tempo gesture, as keyboard binds. Read once at start-up: the
    # SendInput batches below are pre-built from them, so these are NOT
    # hot-reloadable (restart after changing them).
    "pro_stick_down_key": ".",   # Pro Stick Down — held for the tempo
    "pro_stick_up_key": ";",     # Pro Stick Up — the flick that releases
    # How long Pro Stick Up stays held after the flick. 2K polls the stick
    # once per rendered frame, so this only has to comfortably outlast a
    # frame (~13.4 ms); 80 ms is precision_timer.py's value and is well
    # clear. It costs nothing timing-wise — the shot is already away, and
    # a separate thread owns the release so the vision thread never blocks
    # on it.
    "flick_hold_ms": 80,
    # Gap between taking the stick OUT of down and flicking it UP. 0 sends
    # both in one atomic batch, so the stick crosses from full-down to
    # full-up inside a single frame — a motion no physical stick can make.
    #
    # TESTED AT 40 AND REVERTED, and the way it failed is the best evidence
    # that 0 is not merely a default but a requirement.
    #
    # Three shots at gap 40, aimed at 99.0 / 102.8 / 107.0 px — an EIGHT
    # pixel sweep, deliberately compensating one frame then two — came back
    # slightly EARLY every time. Not "early then great then late": early at
    # all three, with the release verified landing on each aim point to
    # 0.02 px. A grade that does not respond to an 8 px sweep is not a
    # timing error, so no value of lead_px can fix a gap.
    #
    # The mechanism that fits: 2K wants the UP-flick to be the release, and
    # a gap makes it see the stick sitting at NEUTRAL first, which it grades
    # as an incomplete/early release no matter when it happens. That is
    # exactly the failure the atomic batch above was built to avoid — the
    # comment there reasoned it out in advance, and this measured it.
    #
    # It also did not deliver the tempo grade it was tried for: 1 great and
    # 2 slightly-rushed over the three gap shots, against 7 great out of 9
    # at gap 0. So it is worse on both axes.
    #
    # One thing it did buy: the very first EARLY of the whole session, which
    # finally brackets the good band from below. At gap 0, 99.0 px is great
    # and the early side exists, so the aim point is inside a band rather
    # than sitting on its late edge.
    #
    # A gap > 0 makes the stick pass through NEUTRAL for a real,
    # controller-like interval before going up. Left in the code because it
    # costs nothing at 0 — but do not re-arm it expecting to compensate with
    # lead_px, because that was tried across 8 px and does not work.
    #
    # The scheduled release instant does NOT move: with a gap the down-key
    # release still lands exactly on the meter's crossing, and only the
    # up-flick trails it. That ordering is deliberate — leaving the down
    # position is what ends the shot, so it is the event worth timing, and
    # the up-flick is what 2K reads as tempo.
    #
    # Try 30-50 (roughly 2-4 frames of stick travel). If the tempo grade
    # improves but the release drifts late, 2K is triggering on the UP key
    # instead and the gap has to come back toward 0.
    "flick_gap_ms": 40,
    # Speed ceiling for the gap: it is applied only to fills SLOWER than
    # this, and 0 applies it to every shot. This is what makes the gap
    # usable at all — it graded 3/3 slightly early on fills that gap 0
    # already handles (9 great of 11 at 99.0 px), but it is the only
    # intervention that has ever moved a slow fill off "late": v0.280 with a
    # 40 ms gap came back slightly EARLY, against six consecutive lates at
    # v 0.274-0.277 spread over aims from 96.8 to 106.8 px.
    #
    # 0.29 is where the evidence changes sign — every fill at or above it is
    # great with no gap, every fill below it is late with no gap. So the gap
    # is now a targeted treatment for the band that has never once worked,
    # and it cannot touch the band that always does.
    #
    # EXPECT EARLY BEFORE GREAT. The single slow-fill gap shot was early, so
    # the likely next move is a LATER aim for slow fills only, which needs a
    # slow-side lead_px rather than this knob. If it reads early again, that
    # is the follow-up; if it reads late, the gap does nothing here either
    # and the slow band is simply out of reach.
    "flick_gap_v_max": 0.29,
    "hold_ms": 650,          # fallback tempo when the meter is never seen
    "no_meter_ms": 400,      # give up looking for the meter after this
    # Busy-wait window for sub-ms release accuracy. MUST BE >= lookahead_ms.
    #
    # The old 1.0 ms came from a synthetic bench (p99 overshoot 0.55 ms) that
    # does not survive contact with the game: 8 of 43 in-game shots landed
    # 3-7.4 ms LATE. Raising it to 5 ms cut that to 1 of 19 — but that one
    # still missed, overshooting by 10.3 ms, because the cause is not timer
    # granularity. wait_until() BLOCKS until deadline-margin, and a blocked
    # thread has to be rescheduled onto a core; on a saturated 6-core/
    # 6-thread CPU that wake-up is the whole tail. The error is one-sided —
    # waking early is spun away exactly, waking late is unrecoverable.
    #
    # So don't block at all. wait_s is provably <= lookahead_ms (lead_ready
    # requires remain <= lead_px + v*look_ms, and wait_s = (remain-lead_px)/v),
    # so a margin >= lookahead_ms makes `coarse` non-positive on every LEAD
    # release and _coarse() is never called: pure spin, nothing to reschedule.
    # Costs <=25 ms of one core ONCE PER SHOT (~0.6% duty at this shot rate).
    "spin_margin_ms": 25.0,
    # MAIN KNOB: pipeline lag to cancel — and, because of the way 2K samples
    # input, the ONLY knob that controls which frame the release lands in.
    #
    # 2K polls K once per rendered frame, so the outcome is not the release
    # instant, it is which poll catches it: n_rel = ceil((R - poll_offset)
    # / frame). Every sub-millisecond of scheduling collapses out. Write
    # k = (capture_lag - latency_ms - poll_offset) / frame; only frac(k)
    # matters, and it is the SAME on every shot, because the fill only ever
    # advances on frame boundaries — so the crossing time the tracker
    # estimates is itself boundary-locked. Simulated flip rate against the
    # +-2.5 ms of estimate noise these logs actually show:
    #   frac(k) 0.02 -> 46%   0.10 -> 30%   0.24 -> 10%   0.50 -> 0.7%
    # so the entire slightly-early/slightly-late residual is one number
    # sitting too close to a boundary. Early flips mean frac(k) is just
    # ABOVE 0; moving it to 0.5 means releasing half a frame LATER, i.e.
    # latency_ms DOWN by frame/2 (20 -> 13.3 at frame 13.4).
    #
    # That shift is free, not a timing change: while frac(k) < 0.5 the same
    # poll still catches the release (verified over the whole range), it just
    # stops doing so by a hair. If it ever overshoots past 0.5 the symptom
    # flips to consistently ONE FRAME LATE — that, not a vague drift, is the
    # signal to put it back up.
    #
    # THE STICK FLICK COSTS ~26 MS MORE LAG THAN THE BUTTON DID, and this
    # is the number that pays for it. Measured, not bisected:
    #
    # 24.5 was validated against a K key-up. Carried onto the tempo build
    # it put 13/13 shots at a release point of 111.0 px (sd 0.42 px — the
    # aim was not jittering, it was parked too late) and every one came
    # back Slightly Late, with the fill visibly COLLIDING with the bar's
    # tip and retracting. Two rounds of frame-sized steps (24.5 -> 39)
    # moved the release exactly as predicted, 111.0 -> 106.8 px, and did
    # not move the badge at all — which is the proof that this is NOT poll
    # quantization. A frame of correction that changes nothing means the
    # error is bigger than a frame, so stepping frames was the wrong tool.
    #
    # What settled it, in one shot, was turning on the two instruments
    # already in the file (tick_probe + post_measure_ms):
    #   [tickprobe] tick_len 131, delta +12 px  — IDENTICAL to the button
    #       build's 125/130/131. The tempo meter is the same meter, so
    #       bar_len_px is not the problem and must not be touched.
    #   [overshoot] +8 px (+31 ms)              — where the fill actually
    #       stopped, i.e. how late the flick registered.
    # The file's badge history has shots at overshoot -35, -13, -4 and
    # +7 ms ALL scoring Early; +31 ms is the first confirmed Late. That
    # brackets the Excellent band at +7..+31 and puts its centre at ~+19,
    # so the correction is 31 - 19 = 12 ms earlier: 39 -> 51. Note this is
    # a stop-position relation, a property of the meter and of 2K's
    # grading, so it carries across builds even though the input path did
    # not — which is exactly why it is worth more than four frame guesses.
    #
    # THIS KNOB IS NOT WHAT DECIDES SLOW FILLS, and six badge-labelled
    # shots say so. Sorted by FILL SPEED, not by release position:
    #     v0.274 rel106.8 (lat39) late | v0.288 rel103.8 (51) late
    #     v0.289 rel103.7 (51) late    | v0.292 rel101.0 (60) late
    #     v0.324 rel 99.0 (60) GREAT   | v0.338 rel101.3 (51) GREAT
    # Every fill at v <= 0.292 is late and every one at v >= 0.324 is
    # great, across a sweep of this value from 39 to 60. Release position
    # does NOT order the outcome: v0.292 landed at 101.0 px and was late
    # while v0.338 landed at 101.3 px — 0.3 px LATER — and was great.
    #
    # And the sweep is a controlled test of exactly this knob: going 39 ->
    # 60 moved the slow fills 5.8 px earlier in the bar, ~20 ms, and every
    # one of them stayed late. So there is a speed-dependent term the model
    # does not have, and lead_px = v*latency_ms has the sign backwards for
    # it — it hands slow fills LESS lead in pixels, when the evidence says
    # they need more. Do not answer this by winding latency_ms up: that
    # buys slow fills only by firing the fast ones (already great) early.
    #
    # Leading hypothesis is that the bar itself scales — tick_len measures
    # 125-131 px shot to shot, and a fixed 118.5 px target sits at a
    # different FRACTION of a short bar than a long one. One data point
    # already argues against the naive form of it (v0.274 measured tick_len
    # 131, i.e. slow AND long), so it needs tick_len read against v over
    # several shots before anything is rebuilt on it.
    # Sub-frame scheduling is not involved: err was +0.00..+0.01 ms on all
    # four, and two shots 0.1 px apart landed on opposite sides of the
    # grade, which is what a one-sided error looks like near its edge.
    #
    # 101.2 px is the only release position that has scored GREAT, and the
    # reason 51 produced it exactly once is that lead_px = v*latency_ms:
    # that shot's fill ran at v 0.34 and earned 17.3 px of lead, while the
    # v 0.29 shots got 14.8 px from the same 51 and landed 2.5 px later in
    # the bar. Equalising on the known-good release position at the typical
    # v 0.29 needs 17.3/0.29 = 60. Note what this implies: the lag behaves
    # more like a fixed DISTANCE than a fixed time, so if slow and fast
    # fills keep disagreeing, the parameterisation is what is wrong, not
    # the value — the fix then is a px-anchored aim point, not a bigger
    # latency_ms.
    #
    # The v split then broke too: v0.296 at 100.7 px came back GREAT while
    # v0.292 at 101.0 px was LATE — 0.004 px/ms and 0.3 px apart, opposite
    # grades. So neither speed nor release position separates the outcomes,
    # and what actually orders cleanly is the hit rate per setting:
    #     lat 39 -> 0/1 great | lat 51 -> 1/3 | lat 60 -> 2/3
    # 60 is therefore the right value for a lead expressed in TIME, and the
    # pairs that straddle 0.3 px apart say the rest is not a mean error at
    # all: we are sitting ON the window's late edge, where one input poll
    # (~12 ms) decides the grade. That makes the remaining target the
    # SPREAD, not the mean — see lead_px, which supersedes this knob.
    #
    # An EARLY shot is still welcome in either mode: no labelled shot has
    # ever been early, so the band's lower edge remains unmeasured.
    "latency_ms": 60,        # per-rig; k_meter.json holds the live value
    # 0 = derive the lead from latency_ms above. Non-zero pins the flick to
    # a FIXED fill length instead; 19.5 is the live value in k_meter.json.
    #
    # v*latency_ms is a lead in TIME, so the release POSITION moves with the
    # fill speed: over the v 0.274-0.338 this rig shoots, one latency_ms
    # scatters the flick across ~3 px, and the good window is not much wider
    # than that — so the scatter is the miss rate. Seven badge-labelled
    # shots put every GREAT at 99.0-101.3 px and every LATE at >= 101.0,
    # overlapping, which is what sitting on an edge looks like. So stop
    # scattering and aim at the one position that has scored GREAT at two
    # different speeds: 99.0 px (v0.324, and v0.296 landed 100.7).
    #
    # 19.5 = 118.5 - 99.0, and it puts EVERY speed at 99.0 px. It is also
    # strictly safer than the time-based lead in the release path: 19.5 is
    # below the 21.3 px min_lead_pct clip threshold at every v, which
    # v*latency_ms could not promise (it clips fast fills above ~62 ms), the
    # trigger still comes through LEAD, and the CAP backstop stays far off.
    # The log grew a `lead` field so which mode ran is never a guess.
    #
    # VALIDATED 3/3 great at 99.0 px, over v 0.302 / 0.318 / 0.324 — the
    # same speed band that under a time-based lead produced 101.0 late,
    # 100.7 great and 99.0 great, i.e. a coin flip. The first of the three
    # is the discriminating case rather than a lucky one: at latency_ms 60
    # that fill would have released at 100.4 px, on the exact edge where
    # v0.292/101.0 px missed.
    #
    # It also fixes a second leak the time-based lead had: that shot's
    # trigger fired at 98.6 px, a frame past the 97.2 px the model expects,
    # because the condition can only be tested on frames that actually
    # arrive — and wait_s absorbed it to land on 99.0 anyway. Fill speed and
    # trigger-frame jitter both used to pass straight into the release
    # position; neither does now.
    #
    # If this fires fast fills EARLY while slow ones improve, then the ideal
    # position does scale with speed after all and the answer is a blend of
    # the two terms. Set back to 0 to return to latency_ms, kept tuned at 60
    # for exactly that reason.
    "lead_px": 0,
    # Slow-fill correction on top of lead_px, in px of extra lead per
    # (px/ms) of fill speed BELOW lead_v_ref. Zero above the reference, so
    # it is strictly a tail correction and cannot move a shot that already
    # works.
    #
    # Why it exists: a constant px lead beat a constant time lead outright
    # (8 greats at 99.0 px over v 0.29-0.34), but the one miss at that aim
    # point was v0.276 — the slowest fill on record — graded slightly late.
    # Every fill at v >= 0.29 was great there. So the required lead still
    # grows as the fill slows, just far more weakly than v*latency_ms
    # assumed, and in the opposite direction to it: v*latency gives slow
    # fills LESS lead in px, which is why it scattered.
    #
    # FALSIFIED, gain shipped at 0.0. gain 150 was tried, worked exactly as
    # designed (v0.275 got 21.7 px of lead and released at 96.8 px, 2.2 px
    # earlier than the shot it was built from) and the badge did not move —
    # still slightly late. Together with the earlier latency sweep that is
    # four shots at v <= 0.277 released across a TEN pixel span, 106.8 /
    # 100.4 / 99.0 / 96.8 px, every one graded late, while v >= 0.302 is
    # 8/8 great at a flat 99.0. Release position is simply not the variable
    # that decides the slow tail, so no aim-point term can fix it and a
    # bigger gain would only be a bigger guess.
    #
    # What that leaves: v is most likely a SYMPTOM, not a cause. The fill
    # speed is set by which shot animation 2K chose, which is fixed before
    # the meter is even drawn — so a slow meter marks a shot that was
    # already going to grade badly (the "rushed tempo" line rides along with
    # it on the same shots). If that is right the lever is upstream of this
    # tool entirely: shot context, not release scheduling. Kept in the code
    # because the mechanism is sound and hot-reloadable if a real
    # speed-dependent lag ever turns up; set the gain non-zero to re-arm.
    "lead_v_ref": 0.29,
    "lead_v_gain": 0.0,
    # Cap, so an unusually slow meter cannot ask for an absurd lead: at
    # v0.20 the raw term would want 13.5 px and release at 85 px of fill,
    # far outside anything measured. 6 px is ~1.5 frames.
    "lead_v_max": 6.0,
    # Backstop cap for when speed can't be measured. Must stay clear of the
    # LEAD release point or it preempts it and fires early: with the target
    # fixed, CAP wins whenever v < (bar_len*(1-pct/100)+1)/latency_ms. At 97
    # that is v<0.23 px/ms against an observed floor of 0.27 — too close. At
    # 99 it is v<0.11, a real backstop again. Changes none of the 39 logged
    # shots; it only stops a slow fill from silently taking the wrong path.
    # NB latency_ms is the denominator, so cutting it RAISES this threshold:
    # at 13.3 the cap fires below v<0.16, still 1.7x under the 0.27 floor.
    "release_pct": 99,
    "min_lead_pct": 82,      # minimum percentage of fill before velocity-based release can trigger
    # Same floor for fast fills (v >= 0.41). These floors fight the lead:
    # the faster the fill, the earlier in the bar the earned release point
    # sits, and whichever is LATER wins. At the jumpshot speeds this rig
    # actually shoots (v 0.28-0.34) the gate is absorbed by wait_s and the
    # release still lands exactly where the lead asks. Fast fills are the
    # exception, and latency_ms 60 makes it worse: the earned release point
    # drops below this floor, wait_s clamps at 0, and the flick fires late
    # by 10.4 px at v 0.41 (~25 ms) and 15.8 px at v 0.50 (~32 ms).
    #
    # RESOLVED, and not by touching this number: the one-frame room guard in
    # _track (search `_room`) now lowers whichever floor is in force until it
    # clears the release point, so v 0.41 goes from a worst case of 109.0 px
    # to 99.0 and v 0.50 from 110.1 to 99.0. Note both were WORSE than the
    # 104.3 px a static reading of this floor predicts, because frame-phase
    # overshoot stacks on top of the clip — which is exactly the term that
    # static reading was missing. This value is left at 88 as the nominal
    # floor; the guard relaxes it per shot only as far as the geometry
    # requires, so it still does its real job of refusing a velocity release
    # on a fill we have barely seen.
    "fast_min_lead_pct": 88,
    "meter_offset_ms": 0,    # extra ms after that point (+ only, delays)
    "meter_lead_ms": 0,      # capture/display latency compensation
    "meter_max_ms": 1200,    # hard safety: never hold K longer than this
    "predict": False,        # False = release ONLY from the fill itself
    "autocal": False,        # broken error signal — see MeterEye [measure]
    "hotkey_prefix": "ctrl+alt+",  # "" = bare F-keys (collides with other tools)
    "game_window": "NBA 2K26",
    "force_mss": True,       # force super stable CPU capture backend
    "persistent_span": 118,  # crop hint: how far right of the fill to look
    # THE release target: fill length, in px, at which the shot is perfect.
    # Fixed on purpose. It used to be derived per shot from the "tick span"
    # returned by _locate — but _locate never finds a tick (_find_tick is
    # dead code), it returns `fill_left + persistent_span`, so that span was
    # literally `118 - fill_width_at_the_frame_we_locked_on`. Locking one
    # frame later shortened the target by ~5 px, and 39 logged shots spread
    # the target over 111-121 px: 32 ms peak-to-peak of pure lock-timing
    # noise, which is exactly the slightly-early/slightly-late split. 117 is
    # the mean of that distribution (116.66 over those 39 shots), so the
    # average shot times within ~1 ms of before and only the jitter is gone
    # — and the rounding leans late, away from the one miss that was early.
    #
    # 117 was the mean of the old noisy distribution, and with latency_ms
    # already parked at the frac(k)=0.5 mid-frame value it still released
    # slightly early with no green tick — err +0.00 ms against its own
    # schedule, so the schedule was right and the aim point was not. Phase is
    # spent (moving latency_ms further walks frac(k) back toward a boundary
    # and buys flips, not a shift), so the mean has to move here instead: one
    # render frame of fill is v*frame = 0.31*13.4 ~= 4 px, hence 121. Changing
    # the target does NOT disturb frame phase — frac(k) is built only from
    # capture_lag/latency_ms/poll_offset/frame — which is exactly why this is
    # the right knob for a mean shift.
    #
    # Four single-shot probes then bracketed it, and ordered by target the
    # outcome is MONOTONE:
    #     117 early (ph0.63) | 118 early (ph0.04) | 119 late (ph0.71) | 121 late (ph0.12)
    # so the crossing lies strictly between 118 and 119 and the target — not
    # frame phase — is what moves it. Note ph scattered over 0.04-0.71 across
    # those four without breaking the ordering, which is the evidence AGAINST
    # blaming phase here: if the input poll were choosing the outcome, the
    # early/late labels would not sort by target.
    #
    # No integer can sit in that gap, hence the fraction. This is safe to
    # write: _clamp_config only int()s a value when it is already integral,
    # the release path reads it as float(), and the fill tracker is sub-pixel
    # (see the `sub` and `abx` log fields), so 0.5 px survives end to end.
    #
    # VALIDATED: 118.5 then returned 3/3 "excellent" in-game, so this is a
    # measured setting, not a bisection artifact. Do not move it without a
    # comparable run behind the change.
    #
    # The tick_probe run also settled WHY a constant works here. The real
    # target does move: tick_len measured 131 / 130 / 125 on those three
    # shots, a 6 px spread, which is the actual source of the old
    # early/late alternation — not a mistuned number. But all three were
    # excellent at a FIXED 118.5 with delta ranging +6..+12 px, so the green
    # window comfortably spans that wander. Retargeting per shot off the tick
    # would therefore import 6 px of measurement spread into the release to
    # chase a target the window already covers — strictly worse. The constant
    # is the right shape; keep it.
    "bar_len_px": 118.5,
    "arrow_ratio": 1.227,    # legacy, only used when bar_len_px <= 0
    # Sub-frame release scheduling window: how far BEFORE the crossing the
    # trigger is allowed to fire, with wait_s covering the rest. It is also
    # the only thing that absorbs a capture stall — the trigger cannot fire
    # until remain <= lead_px + v*this, so at 25 ms the window is ~7 px of
    # fill, narrower than one frame plus a stall, and a stalled frame lands
    # past the release point where wait_s clamps to 0 and the shot goes out
    # late. 35 covers a 20 ms stall at every speed and phase; see the
    # `_room` guard in _track, which has to be widened WITH it or the floor
    # binds first and this buys nothing.
    #
    # Cost is real but small: _release spins the last max(spin_margin_ms,
    # this+1) ms rather than blocking, so 35 means ~36 ms of one core per
    # shot instead of 26. Never blocks, which is the property that matters.
    "lookahead_ms": 35,
    "v_lsq": True,           # fit v over ALL samples, not just two endpoints
    "v_long_weight": 0.75,   # weight of long-baseline fill speed vs 60ms window
    "stall_cap_ms": 20,      # frame gaps longer than this are capture stalls, not fill time
    # OFF, and not as a tuning preference — aligning cannot help. The game
    # already quantizes the release itself (ceil to its next input poll), and
    # snapping our instant to a grid UPSTREAM of that cannot undo a
    # downstream quantization. Worked through with g = (capture_lag -
    # poll_offset)/frame: snapping to frame centre changes the sampled frame
    # for a fraction |g| of shots and leaves the rest bit-identical — 0%
    # benefit at g=0, and a whole frame EARLY on 10/20/30% of shots at
    # g=0.1/0.2/0.3. Its best case is a no-op and its normal case is added
    # flips, which is what one round of it produced. Frame phase is
    # latency_ms' job; see there.
    "frame_align": False,
    # Live-measured render period. Read this off the `fr` field in the log
    # (median of n step gaps), never off `dt` — dt is the last gap alone, one
    # noisy sample, and trusting it is what put this at 15.0 for a round.
    # 22 gaps measure 13.4 ms with a 1.0 ms IQR, so 13.3 was right all along.
    "game_frame_ms": 13.4,
    "subframe_fill": True,   # extrapolate across the game's 60fps staircase
    "step_cap_ms": 18,       # max extrapolation past the last observed step
    "autotune": True,        # self-correct latency_ms from measured overshoot
    "autotune_shots": 8,     # shots per correction batch (median of these)
    "autotune_deadband_ms": 6,  # ignore errors smaller than this
    # The overshoot a GREAT shot produces. Read the sign warning below
    # before ever setting "autotune": true on the tempo build.
    #
    # On badge-labelled tempo shots [overshoot] is ANTI-CORRELATED with
    # lateness — the later the shot, the SMALLER the reading:
    #     +29 ms late | +31 ms late | +55 ms GREAT
    # which is the exact opposite of what _autotune is built on ("positive
    # means the fill overshot the target (late)"). With aim 55 and a late
    # shot reading +29, err comes out NEGATIVE, the step lowers latency_ms,
    # and the controller releases even LATER — it would run away from the
    # target on every miss. That is why autotune stays off here, and it is
    # a sign error, not a tuning preference: raising the aim does not fix a
    # signal that moves the wrong way.
    #
    # `stopped` is where the fill's ANIMATION ended, not where the release
    # registered, and 2K animates the meter according to the grade it just
    # awarded: a good release runs the fill on past the tick (+18 px),
    # while a late one has already hit the bar's end and retracts, so its
    # reading saturates near the bar length (+8 px). The number is mostly
    # an EFFECT of the outcome. Kept at the great shot's value so a stray
    # enable sits near a no-op. Steer latency_ms by the badge.
    "autotune_target_ms": 55,
    "trust_min_frames": 8,   # fewer tracked frames -> discard the sample
    "trust_min_span": 95,    # smaller meter read -> truncated, discard
    # --- CPU budget -------------------------------------------------------
    # This rig is a 6-core / 6-thread Ryzen 3500X: no SMT, so every core this
    # tool takes is a core 2K26 does not get, and a dropped render frame
    # costs far more timing accuracy than anything below buys.
    "cv_threads": 1,         # OpenCV worker threads (0 = let OpenCV decide).
                             # Every cvtColor/inRange/connectedComponents used
                             # to fan out across all 6 cores at once.
    "poll_sleep_ms": 0.8,    # yield this long when the backend has no new
                             # frame, instead of spinning the core
    "process_priority": "above_normal",   # normal | above_normal | high
    "vision_priority": "above_normal",    # tracking thread; the final wait
                                          # still boosts to time_critical
    "post_measure_ms": 0,    # post-release overshoot sampling. Only feeds
                             # autotune/autocal; 0 = off. Costs ~150 ms of
                             # solid capture right as the shot animates.
    # --- meter search -----------------------------------------------------
    "mask_bgr": True,        # mask in the capture's own channels (no HSV)
    "mag_rb_min": 160,       # magenta: blue AND red at least this
    "mag_g_max": 105,        # magenta: green below this
    "locate_band_lo": 0.60,  # search band, fraction of window height
    "locate_band_hi": 0.90,
    "locate_full_after_ms": 220,  # band came up empty this long -> whole window
    # --- fill tracker -----------------------------------------------------
    "v_filter": "ab",        # "ab" = alpha-beta tracker, "lsq" = the fit
    "ab_alpha": 0.30,        # position gain
    "ab_beta": 0.105,        # velocity gain
    # One-shot green-tick measurement, logged as [tickprobe], never used for
    # the release. OFF by default and deliberately not tied to `debug`: it
    # runs a cvtColor + connectedComponents mid-shot, inside the tracking
    # loop that is pure release latency on a 6-thread CPU. It already answered
    # the question it was added for (see bar_len_px), so it should only come
    # back on to re-measure the tick after a meter/resolution/HUD change.
    "tick_probe": False,
    "debug": True,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("KMeter")

try:
    _fh = logging.FileHandler(
        os.path.join(_script_dir, "k_meter_log.txt"), encoding="utf-8")
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(_fh)
except OSError:
    pass

# Priority names accepted in the config, mapped to the Win32 constants.
# HIGH_PRIORITY_CLASS puts this process above every thread 2K26 owns; on a
# CPU with a spare thread that is free, on a 6-thread one it is not.
_PRIORITY_CLASSES = {
    "normal": 0x00000020,
    "above_normal": 0x00008000,
    "high": 0x00000080,
}
_THREAD_PRIORITIES = {
    "normal": 0,
    "above_normal": 1,
    "highest": 2,
    "time_critical": 15,
}


def _clamp_config(cfg: dict) -> dict:
    def _num(key, default, lo, hi):
        try:
            val = float(cfg.get(key, default))
        except (TypeError, ValueError):
            val = float(default)
        val = max(lo, min(hi, val))
        cfg[key] = int(val) if val.is_integer() else val

    _num("flick_hold_ms", 80, 15, 400)
    _num("flick_gap_ms", 0, 0, 200)
    _num("flick_gap_v_max", 0.29, 0.0, 1.0)
    _num("hold_ms", 650, 100, 3000)
    _num("no_meter_ms", 400, 100, 1500)
    _num("spin_margin_ms", 1.0, 0.0, 50.0)
    _num("latency_ms", 60, 0, 300)
    _num("lead_px", 0, 0, 60)
    _num("lead_v_ref", 0.29, 0.0, 1.0)
    _num("lead_v_gain", 150.0, 0.0, 600.0)
    _num("lead_v_max", 6.0, 0.0, 30.0)
    _num("release_pct", 97, 50, 100)
    _num("min_lead_pct", 82, 50, 100)
    _num("fast_min_lead_pct", 88, 50, 100)
    _num("meter_offset_ms", 0, -300, 800)
    _num("meter_lead_ms", 0, 0, 200)
    _num("meter_max_ms", 1200, 600, 2500)
    _num("persistent_span", 118, 20, 500)
    _num("bar_len_px", 116, 0, 400)
    _num("arrow_ratio", 1.227, 1.0, 2.0)
    _num("lookahead_ms", 25, 0, 60)
    _num("v_long_weight", 0.75, 0.0, 1.0)
    _num("stall_cap_ms", 20, 8, 100)
    _num("step_cap_ms", 18, 0, 40)
    _num("game_frame_ms", 13.3, 5, 40)
    _num("autotune_shots", 8, 3, 50)
    _num("autotune_deadband_ms", 6, 0, 40)
    _num("autotune_target_ms", 12, -40, 40)
    _num("trust_min_frames", 8, 2, 60)
    _num("trust_min_span", 95, 40, 200)
    _num("cv_threads", 1, 0, 16)
    _num("poll_sleep_ms", 0.8, 0.0, 10.0)
    _num("post_measure_ms", 0, 0, 400)
    _num("mag_rb_min", 160, 80, 255)
    _num("mag_g_max", 105, 0, 200)
    _num("locate_band_lo", 0.60, 0.0, 0.95)
    _num("locate_band_hi", 0.90, 0.05, 1.0)
    _num("locate_full_after_ms", 220, 0, 1200)
    _num("ab_alpha", 0.30, 0.02, 1.0)
    _num("ab_beta", 0.105, 0.0, 1.0)
    if cfg["locate_band_hi"] <= cfg["locate_band_lo"]:
        cfg["locate_band_lo"] = DEFAULT_CONFIG["locate_band_lo"]
        cfg["locate_band_hi"] = DEFAULT_CONFIG["locate_band_hi"]
    cfg["mask_bgr"] = bool(cfg.get("mask_bgr", DEFAULT_CONFIG["mask_bgr"]))
    vf = str(cfg.get("v_filter", DEFAULT_CONFIG["v_filter"])).strip().lower()
    cfg["v_filter"] = vf if vf in ("ab", "lsq") else DEFAULT_CONFIG["v_filter"]
    for _pk, _pmap in (("process_priority", _PRIORITY_CLASSES),
                       ("vision_priority", _THREAD_PRIORITIES)):
        pv = str(cfg.get(_pk, DEFAULT_CONFIG[_pk])).strip().lower()
        cfg[_pk] = pv if pv in _pmap else DEFAULT_CONFIG[_pk]
    cfg["autotune"] = bool(cfg.get("autotune", DEFAULT_CONFIG["autotune"]))
    cfg["v_lsq"] = bool(cfg.get("v_lsq", DEFAULT_CONFIG["v_lsq"]))
    cfg["frame_align"] = bool(cfg.get("frame_align", DEFAULT_CONFIG["frame_align"]))
    cfg["subframe_fill"] = bool(cfg.get("subframe_fill", DEFAULT_CONFIG["subframe_fill"]))
    cfg["predict"] = bool(cfg.get("predict", DEFAULT_CONFIG["predict"]))
    cfg["force_mss"] = bool(cfg.get("force_mss", DEFAULT_CONFIG["force_mss"]))
    cfg["debug"] = bool(cfg.get("debug", DEFAULT_CONFIG["debug"]))
    for _kk in ("action_key", "pro_stick_down_key", "pro_stick_up_key"):
        key = cfg.get(_kk, DEFAULT_CONFIG[_kk])
        cfg[_kk] = (key.strip().lower()
                    if isinstance(key, str) and key.strip()
                    else DEFAULT_CONFIG[_kk])
    hp = cfg.get("hotkey_prefix", DEFAULT_CONFIG["hotkey_prefix"])
    cfg["hotkey_prefix"] = hp if isinstance(hp, str) else DEFAULT_CONFIG["hotkey_prefix"]
    gw = cfg.get("game_window", DEFAULT_CONFIG["game_window"])
    cfg["game_window"] = (gw.strip() if isinstance(gw, str) and gw.strip()
                          else DEFAULT_CONFIG["game_window"])
    return {k: cfg.get(k, DEFAULT_CONFIG[k]) for k in DEFAULT_CONFIG}


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        return copy.deepcopy(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, "r") as f:
            loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError("config is not an object")
            return _clamp_config(loaded)
    except (OSError, ValueError):
        log.warning("[config] unreadable, using defaults")
        return copy.deepcopy(DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    try:
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cfg, f, indent=4)
        os.replace(tmp, CONFIG_PATH)
    except OSError as exc:
        log.warning("[config] save failed: %s", exc)


config = load_config()
active = True
_shutdown = threading.Event()

# --------------------------------------------------------------------------- #
# Windows timer resolution + priority
# --------------------------------------------------------------------------- #
winmm = None
try:
    winmm = ctypes.WinDLL("winmm")
    winmm.timeBeginPeriod(1)
except Exception:
    pass

try:
    _k32 = ctypes.windll.kernel32
    _k32.SetPriorityClass(
        _k32.GetCurrentProcess(),
        _PRIORITY_CLASSES[config["process_priority"]])
except Exception:
    pass

THREAD_PRIORITY_TIME_CRITICAL = 15


def _boost_current_thread(level: int = THREAD_PRIORITY_TIME_CRITICAL) -> None:
    try:
        h = ctypes.windll.kernel32.GetCurrentThread()
        ctypes.windll.kernel32.SetThreadPriority(h, level)
    except Exception:
        pass


def _vision_thread_priority() -> int:
    """Priority the meter-tracking loop runs at between shots.

    The loop polls for ~400 ms per shot. At TIME_CRITICAL that outranks
    2K26's own render and input threads for the whole of it, on a CPU with
    six threads and no SMT to absorb it — the stutter that costs is the one
    that moves the frame we are trying to read. Only the final wait needs to
    win a scheduling fight, and `_release` boosts for exactly that long.
    """
    return _THREAD_PRIORITIES.get(config.get("vision_priority",
                                             "above_normal"), 1)


# --------------------------------------------------------------------------- #
# High-resolution waiting
# --------------------------------------------------------------------------- #
_CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 0x00000002
_TIMER_ALL_ACCESS = 0x1F0003


class PrecisionWaiter:
    def __init__(self) -> None:
        self._timer = None
        self._k32 = None
        try:
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.CreateWaitableTimerExW.restype = wt.HANDLE
            k32.CreateWaitableTimerExW.argtypes = (
                ctypes.c_void_p, wt.LPCWSTR, wt.DWORD, wt.DWORD)
            k32.SetWaitableTimer.restype = wt.BOOL
            k32.SetWaitableTimer.argtypes = (
                wt.HANDLE, ctypes.POINTER(wt.LARGE_INTEGER), wt.LONG,
                ctypes.c_void_p, ctypes.c_void_p, wt.BOOL)
            k32.WaitForSingleObject.restype = wt.DWORD
            k32.WaitForSingleObject.argtypes = (wt.HANDLE, wt.DWORD)
            handle = k32.CreateWaitableTimerExW(
                None, None, _CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                _TIMER_ALL_ACCESS)
            if handle:
                self._timer, self._k32 = handle, k32
        except Exception:
            pass

    @property
    def high_res(self) -> bool:
        return self._timer is not None

    def _coarse(self, sec: float) -> None:
        if self._timer:
            due = wt.LARGE_INTEGER(-int(sec * 10_000_000))
            if self._k32.SetWaitableTimer(
                    self._timer, ctypes.byref(due), 0, None, None, False):
                self._k32.WaitForSingleObject(
                    self._timer, wt.DWORD(int(sec * 1000) + 100))
                return
        time.sleep(sec)

    def wait_until(self, deadline: float, margin_ms: float) -> None:
        coarse = deadline - margin_ms / 1000.0 - time.perf_counter()
        # Below ~0.5 ms the timer's own granularity is the whole wait, so the
        # syscall buys nothing over spinning the last stretch out.
        if coarse > 0.0005:
            self._coarse(coarse)
        while time.perf_counter() < deadline:
            pass

    def close(self) -> None:
        if self._timer:
            try:
                ctypes.windll.kernel32.CloseHandle(self._timer)
            except Exception:
                pass
            self._timer = None


# --------------------------------------------------------------------------- #
# Timing statistics
# --------------------------------------------------------------------------- #
class Stats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.count = 0
            self._mean = 0.0
            self._m2 = 0.0
            self.min_err = math.inf
            self.max_err = -math.inf
            self.last_err = 0.0
            self.misses = 0

    def record(self, err_ms: float) -> None:
        with self._lock:
            self.count += 1
            delta = err_ms - self._mean
            self._mean += delta / self.count
            self._m2 += delta * (err_ms - self._mean)
            self.min_err = min(self.min_err, err_ms)
            self.max_err = max(self.max_err, err_ms)
            self.last_err = err_ms
            if abs(err_ms) > 1.0:
                self.misses += 1

    def snapshot(self) -> dict:
        with self._lock:
            std = math.sqrt(self._m2 / self.count) if self.count > 1 else 0.0
            return {"count": self.count,
                    "mean": self._mean if self.count else 0.0,
                    "std": std,
                    "min": self.min_err if self.count else 0.0,
                    "max": self.max_err if self.count else 0.0,
                    "last": self.last_err, "misses": self.misses}


stats = Stats()

_state_lock = threading.Lock()
_injecting = False
_physically_held = False
_passthrough_held = False
_cycle_active = False

# --------------------------------------------------------------------------- #
# Synthetic key release via pre-built SendInput
# --------------------------------------------------------------------------- #
_ULONG_PTR = wt.WPARAM
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
INPUT_KEYBOARD = 1


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD),
                ("dwFlags", wt.DWORD), ("time", wt.DWORD),
                ("dwExtraInfo", _ULONG_PTR))


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", _ULONG_PTR))


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg", wt.DWORD), ("wParamL", wt.WORD),
                ("wParamH", wt.WORD))


class _INPUTUNION(ctypes.Union):
    _fields_ = (("mi", MOUSEINPUT), ("ki", KEYBDINPUT),
                ("hi", HARDWAREINPUT))


class INPUT(ctypes.Structure):
    _fields_ = (("type", wt.DWORD), ("union", _INPUTUNION))


_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.SendInput.restype = wt.UINT
_user32.SendInput.argtypes = (wt.UINT, ctypes.POINTER(INPUT), ctypes.c_int)

_EXTENDED_KEY_NAMES = {
    "up", "down", "left", "right", "insert", "delete", "home", "end",
    "page up", "page down", "print screen", "right ctrl", "right alt",
    "left windows", "right windows", "menu", "apps",
}


def _build_key_input(key: str, is_up: bool) -> INPUT:
    try:
        sc = keyboard.key_to_scan_codes(key)[0]
    except Exception:
        sc = 0
    flags = KEYEVENTF_SCANCODE
    if is_up:
        flags |= KEYEVENTF_KEYUP
    if sc > 0xFF or key.strip().lower() in _EXTENDED_KEY_NAMES:
        flags |= KEYEVENTF_EXTENDEDKEY
        sc &= 0xFF
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki = KEYBDINPUT(0, sc, flags, 0, 0)
    return inp


# --------------------------------------------------------------------------- #
# Pro stick gesture — pre-built SendInput batches
#
# Tempo shooting is a two-key gesture on keyboard: hold Pro Stick Down for
# the tempo, then flick Pro Stick Up to release. Every event is parsed and
# packed once here, at start-up, so the release path is a single syscall
# with no key lookup, no allocation and nothing to parse.
# --------------------------------------------------------------------------- #
_ps_down = str(config["pro_stick_down_key"])
_ps_up = str(config["pro_stick_up_key"])


def _batch(*inputs: INPUT):
    """Pack events into one SendInput call, delivered atomically in order.

    Returns (array, pointer, count) and the array is kept in the tuple on
    purpose: drop it and ctypes frees the buffer the pointer points at.
    """
    arr = (INPUT * len(inputs))(*inputs)
    return arr, ctypes.cast(arr, ctypes.POINTER(INPUT)), len(inputs)


# Shot start: clear any stale flick key, then pull the stick down. The up-key
# release leads because it is a no-op when that key is already up, and stops
# the previous shot's flick hold from fighting this shot's pull-down when two
# shots land inside `flick_hold_ms` of each other.
_START = _batch(_build_key_input(_ps_up, is_up=True),
                _build_key_input(_ps_down, is_up=False))
# THE RELEASE, `flick_gap_ms` = 0: one call carrying both events, never two
# calls: between two SendInputs there is an instant where NEITHER stick key is
# down, and a game input poll landing in that gap reads the stick as neutral
# instead of flicked up — a whole frame of the gesture lost to something we
# control. One batch is delivered in order with no such gap.
_FLICK = _batch(_build_key_input(_ps_down, is_up=True),
                _build_key_input(_ps_up, is_up=False))
# THE RELEASE, `flick_gap_ms` > 0: the down key alone, so the stick passes
# through NEUTRAL before it goes up. Same instant either way — this is the
# event the meter timing is anchored on, because it is the one that takes the
# stick out of the down position.
_NEUTRAL = _batch(_build_key_input(_ps_down, is_up=True))
# The up-flick, sent `flick_gap_ms` after neutral.
_UP_PRESS = _batch(_build_key_input(_ps_up, is_up=False))
# End of the flick hold.
_FLICK_END = _batch(_build_key_input(_ps_up, is_up=True))
# Shutdown safety.
_ALL_UP = _batch(_build_key_input(_ps_down, is_up=True),
                 _build_key_input(_ps_up, is_up=True))


def _send(batch) -> bool:
    global _injecting
    _injecting = True
    try:
        _arr, ptr, n = batch
        return _user32.SendInput(n, ptr, ctypes.sizeof(INPUT)) == n
    finally:
        _injecting = False


_flick_closer = None      # set in main(); owns the end of the flick hold
# Fill speed of the shot being released, px/ms, published by the tracker so
# the injector can pick a gesture from it. 0.0 = unknown (fallback paths),
# which every speed test below treats as "not in the gap band".
_flick_v = 0.0


def _set_flick_speed(v: float) -> None:
    global _flick_v
    try:
        _flick_v = float(v)
    except (TypeError, ValueError):
        _flick_v = 0.0


def _inject_shot_start() -> bool:
    """Pull the pro stick down — the shot starts and the tempo begins."""
    if _send(_START):
        return True
    try:                                    # keyboard module as the fallback
        keyboard.release(_ps_up)
        keyboard.press(_ps_down)
        return True
    except Exception:
        log.exception("[input] pro stick down failed")
        return False


def _inject_flick() -> None:
    """THE release: take the stick out of down, then flick up and hold.

    Everything after the first event is handed to `_flick_closer` rather
    than waited out here: this runs on the vision thread at TIME_CRITICAL,
    and sitting on a core for `flick_gap_ms + flick_hold_ms` with the shot
    already away is exactly the kind of stall that costs 2K a render frame.
    """
    gap = float(config.get("flick_gap_ms", 0))
    # Speed-gated: a gap is harmful on fills that already work (it graded 3/3
    # early there) and is the only thing that has ever moved a slow fill off
    # "late". flick_gap_v_max > 0 restricts it to fills below that speed;
    # 0 applies it to every shot. Unknown speed (0.0) never gets the gap.
    _gv = float(config.get("flick_gap_v_max", 0))
    if gap > 0 and _gv > 0 and not (0.0 < _flick_v < _gv):
        gap = 0.0
    gap /= 1000.0
    hold = float(config.get("flick_hold_ms", 80)) / 1000.0
    if not _send(_NEUTRAL if gap > 0 else _FLICK):
        try:
            keyboard.release(_ps_down)
            if gap <= 0:
                keyboard.press(_ps_up)
        except Exception:
            log.exception("[input] flick failed")
    t = time.perf_counter()
    closer = _flick_closer
    if closer is not None:
        closer.schedule(t + gap if gap > 0 else None, t + gap + hold)
    else:                                   # no closer — run it inline, but
        if gap > 0:                         # never leave a key down
            time.sleep(gap)
            _send(_UP_PRESS)
        time.sleep(hold)
        _inject_flick_end()


def _inject_flick_end() -> None:
    if not _send(_FLICK_END):
        try:
            keyboard.release(_ps_up)
        except Exception:
            pass


def _inject_all_up() -> None:
    """Never leave either stick key held — used on exit."""
    if not _send(_ALL_UP):
        for _k in (_ps_down, _ps_up):
            try:
                keyboard.release(_k)
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Flick closer — releases Pro Stick Up `flick_hold_ms` after the flick
#
# Deliberately NOT precise and NOT boosted: the shot is already gone by the
# time this runs, and the only requirement is that the hold outlasts a
# render frame. A plain event wait keeps it off the release path entirely
# and lets it exit promptly on shutdown.
# --------------------------------------------------------------------------- #
class FlickCloser:
    def __init__(self) -> None:
        self._evt = threading.Event()
        self._t_up = None       # when to press the up key (flick_gap_ms > 0)
        self._t_end = 0.0       # when to release it
        threading.Thread(target=self._run, name="flickclose",
                         daemon=True).start()

    def schedule(self, t_up, t_end: float) -> None:
        self._t_up = t_up
        self._t_end = t_end
        self._evt.set()

    @staticmethod
    def _until(t: float) -> None:
        rem = t - time.perf_counter()
        if rem > 0:
            # Returns early on shutdown — and the caller still sends its
            # event, because a stuck stick key survives this process.
            _shutdown.wait(rem)

    def _run(self) -> None:
        while not _shutdown.is_set():
            if not self._evt.wait(0.5):
                continue
            self._evt.clear()
            try:
                t_up, t_end = self._t_up, self._t_end
                if t_up is not None:
                    self._until(t_up)
                    _send(_UP_PRESS)
                self._until(t_end)
                _inject_flick_end()
            except Exception:
                log.exception("[flick] closing the flick failed")


# --------------------------------------------------------------------------- #
# Release worker — persistent TIME_CRITICAL thread, retargetable deadline
# --------------------------------------------------------------------------- #
class ReleaseWorker:
    def __init__(self) -> None:
        self._evt = threading.Event()
        self._retarget = threading.Event()
        self._new_deadline = None
        self._press = 0.0
        self._hold = 0.0
        self._waiter = PrecisionWaiter()
        self.hr_timer = self._waiter.high_res
        self.active_cycle = False
        threading.Thread(target=self._run, name="release",
                         daemon=True).start()

    def schedule(self, press_time: float, hold_ms: float) -> None:
        self._press = press_time
        self._hold = float(hold_ms)
        self.active_cycle = True
        self._evt.set()

    def cancel(self) -> None:
        self.active_cycle = False
        self._retarget.set()

    def retarget(self, deadline: float) -> None:
        self._new_deadline = deadline
        self._retarget.set()

    def _run(self) -> None:
        global _cycle_active
        _boost_current_thread()
        while not _shutdown.is_set():
            if not self._evt.wait(0.5):
                continue
            self._evt.clear()
            self._retarget.clear()
            self._new_deadline = None
            try:
                press = self._press
                deadline = press + self._hold / 1000.0
                spin = float(config["spin_margin_ms"]) / 1000.0
                while not _shutdown.is_set() and self.active_cycle:
                    rem = deadline - spin - time.perf_counter()
                    if rem <= 0:
                        break
                    if self._retarget.wait(min(rem, 0.05)):
                        self._retarget.clear()
                        nd = self._new_deadline
                        if nd:
                            # accept past-due: release ASAP, never ignore
                            deadline = max(nd,
                                           time.perf_counter() + 0.002)
                if self.active_cycle:
                    self._waiter.wait_until(
                        deadline, float(config["spin_margin_ms"]))
                    if self.active_cycle:
                        self.active_cycle = False
                        t_rel = time.perf_counter()
                        _inject_flick()
                        held = (t_rel - press) * 1000.0
                        err = held - (deadline - press) * 1000.0
                        stats.record(err)
                        if config["debug"]:
                            log.info("[shot] tempo %.2f ms -> flick "
                                     "(err %+.2f ms)", held, err)
            except Exception:
                log.exception("[worker] release failed")
            finally:
                with _state_lock:
                    _cycle_active = False


# --------------------------------------------------------------------------- #
# MeterEye — watches the horizontal PURPLE meter, schedules the release
# when the magenta fill front reaches the green tick at the bar's tip.
# --------------------------------------------------------------------------- #
class MeterEye:
    # Magenta fill: measured H 150-151. The upper bound used to run to 162,
    # which reaches into court reds (Bulls logo sits right of the meter) —
    # stray red past the fill front inflated both the span and the measured
    # fill speed, firing the release early. Kept tight around the real hue.
    MAG_LO = (146, 120, 150)
    MAG_HI = (157, 255, 255)
    # The same fill, expressed directly in the captured buffer's own channels
    # so no colour conversion is needed at all. Magenta is high blue + high
    # red + low green, and because the blue and red bounds are IDENTICAL the
    # box is channel-order agnostic — it matches BGR, RGB, BGRA and RGBA
    # alike, which is what lets one mask serve both capture backends.
    #
    # Grid-searched against the HSV mask over 63 real 2K26 frames: identical
    # fill front on 60 of them, worst case 1 px, never a miss, and 1.8 stray
    # pixels per frame inside the search band (largest stray blob 45 px
    # against the bar's ~2900). On the 1600x270 band that is 0.40 ms versus
    # 1.30 ms for cvtColor+inRange.
    #
    # Doing the same predicate exactly, as numpy vector ops on the split
    # channels, was tried and is a trap: 27 ms, eleven times SLOWER than the
    # HSV path, because a dozen elementwise temporaries over 1.4 Mpx is all
    # memory traffic. cv2.inRange is one SIMD pass and no allocation.
    MAG_RB_MIN = 160             # blue AND red at least this
    MAG_G_MAX = 105              # green below this
    TICK_LO = (45, 60, 90)       # green tick: measured H 53-65
    TICK_HI = (80, 255, 255)
    BAR_FRAC = 0.082             # bar length / window width (measured)

    def __init__(self, release_worker: "ReleaseWorker") -> None:
        self.worker = release_worker
        self._armed = threading.Event()
        self._over_hist = []   # measured overshoot per shot, for _autotune
        self._press = 0.0
        self._rect = None
        self._rect_t = -10.0
        self.last = "idle"
        self.persistent_span = 118  # Learned default span for player's custom meter size
        self._tick_span = None      # per-shot fill->tick distance (scales)
        # One waiter for the life of the thread. This used to be built inside
        # _release, which leaked a kernel timer handle on every single shot.
        self._waiter = PrecisionWaiter()
        self._mbuf = {}             # region size -> reused mask array
        self._bnd = None            # cached (channels, lo, hi) -> inRange bounds
        threading.Thread(target=self._run, name="metereye",
                         daemon=True).start()

    # ---------------------------------------------------------------- vision
    def _bounds(self, nch: int):
        """inRange bounds for a buffer with `nch` channels, cached."""
        lo = int(config.get("mag_rb_min", self.MAG_RB_MIN))
        hi = int(config.get("mag_g_max", self.MAG_G_MAX))
        key = (nch, lo, hi)
        cached = self._bnd
        if cached is None or cached[0] != key:
            if nch == 4:
                val = ((lo, 0, lo, 0), (255, hi, 255, 255))
            else:
                val = ((lo, 0, lo), (255, hi, 255))
            self._bnd = cached = (key, val)
        return cached[1]

    def _mask(self, buf):
        """Magenta mask for a captured buffer, into a reused output array.

        Never slices the alpha channel off first: `buf[:, :, :3]` is a
        strided view and forcing it contiguous costs 2.6 ms on the search
        band, six times the mask itself. inRange takes the 4-channel buffer
        as it comes.
        """
        h, w = buf.shape[0], buf.shape[1]
        nch = buf.shape[2] if buf.ndim == 3 else 1
        out = self._mbuf.get((h, w))
        if out is None:
            out = _np.empty((h, w), _np.uint8)
            if len(self._mbuf) >= 8:        # only a handful of sizes recur
                self._mbuf.pop(next(iter(self._mbuf)))
            self._mbuf[(h, w)] = out
        if config.get("mask_bgr", True):
            lo, hi = self._bounds(nch)
            _cv2.inRange(buf, lo, hi, dst=out)
        else:
            src = buf[:, :, :3] if nch == 4 else buf
            _cv2.inRange(_cv2.cvtColor(src, _cv2.COLOR_BGR2HSV),
                         self.MAG_LO, self.MAG_HI, dst=out)
        return out

    @staticmethod
    def _pace() -> None:
        """Give the core back while waiting for the next rendered frame."""
        s = float(config.get("poll_sleep_ms", 0.8)) / 1000.0
        if s > 0:
            time.sleep(s)

    def arm(self, press_time: float) -> None:
        self._press = press_time
        self._last_v = 0.30
        self._armed.set()

    def _game_rect(self):
        try:
            u32 = ctypes.windll.user32
            hwnd = u32.FindWindowW(None, config["game_window"])
            if not hwnd:
                return None
            r = wt.RECT()
            if not u32.GetClientRect(hwnd, ctypes.byref(r)):
                return None
            pt = wt.POINT(0, 0)
            u32.ClientToScreen(hwnd, ctypes.byref(pt))
            w, h = r.right - r.left, r.bottom - r.top
            if w < 400 or h < 300:
                return None
            return pt.x, pt.y, w, h
        except Exception:
            return None

    def _grab(self, camera, sct, x, y, w, h):
        """Raw BGRA capture of the region, or None if there is no NEW frame.

        dxcam is a Desktop Duplication client: it hands over a frame only
        when the desktop actually changed, and returns None otherwise. This
        used to fall straight through to a full mss BitBlt on every one of
        those Nones — and mss costs 6.9 ms for even the small tracking crop,
        18.6 ms for the whole window. That is the entire explanation for the
        6.0 ms loop interval in the logs: over half of all iterations were
        re-capturing, on the CPU, pixels the GPU had already told us had not
        changed.

        Returning None instead is both cheaper and more accurate. Every
        frame that does arrive is now a genuinely new rendered frame, so the
        fill's staircase steps line up with real frame boundaries instead of
        being smeared across duplicate reads.
        """
        if camera is not None:
            return camera.grab(region=(int(x), int(y),
                                       int(x + w), int(y + h)))
        shot = sct.grab({"left": int(x), "top": int(y),
                         "width": int(w), "height": int(h)})
        return _np.frombuffer(shot.raw, _np.uint8).reshape(
            shot.height, shot.width, 4)

    def _autotune(self, over_ms: float) -> None:
        """Nudge latency_ms from measured overshoot.

        `over_ms` is how late the release landed: positive means the fill
        overshot the target (late), negative means it stopped short
        (early). Raising latency_ms releases EARLIER, so the correction
        moves with the sign of the error.

        Acts as soon as the error is real, not after a fixed shot count.
        A single shot carries ~one game frame (~17 ms) of noise, so one
        sample can never justify a move — but four shots all reading -13 ms
        obviously can, and waiting for a magic 8th is just lost games.

        So it asks the only question that matters after EVERY shot: is the
        error bigger than the noise in the samples I have? It compares the
        median error against its own standard error (from the MAD, so a
        single wild shot can't fake confidence). A large consistent error
        clears that bar at 2-3 shots and gets corrected immediately; a
        small ambiguous one keeps collecting until it either becomes clear
        or hits `autotune_shots` and gets acted on anyway.

        Step size scales with the error too — a 30 ms miss is not corrected
        in 8 ms crawls, but the gain is under 1.0 so it converges instead
        of oscillating.
        """
        if not config.get("autotune", True):
            return
        self._over_hist.append(float(over_ms))
        h = self._over_hist
        n = len(h)
        if n < 3:                       # 1-2 samples are never evidence
            return
        s = sorted(h)
        median = s[n // 2]
        # Aim at a setpoint, not at zero. The magenta keeps animating for a
        # beat after 2K registers the release, so a fill that stops exactly
        # on target was actually released BEFORE it — 2K scores that "Early".
        # Badge-verified shots at -35, -13, -4 and +7 ms ALL came back Early,
        # which is what proved the offset is real. Chasing zero walked
        # latency_ms 53 -> 67 in the wrong direction over two batches.
        aim = float(config.get("autotune_target_ms", 25))
        err = median - aim
        dead = float(config.get("autotune_deadband_ms", 6))
        cap = int(config.get("autotune_shots", 8))

        # Spread of the samples themselves, via MAD -> robust sigma.
        mad = sorted(abs(x - median) for x in h)[n // 2]
        # Floor at the real per-shot noise (~1 game frame). Claiming
        # less makes 2 identical readings look like proof and the
        # loop lurches past the target on a coincidence.
        sigma = max(1.4826 * mad, 14.0)
        stderr = sigma / math.sqrt(n)
        # Real if it clears both the deadband and 2 standard errors.
        sure = abs(err) > max(dead, 2.0 * stderr)
        if not sure and n < cap:
            return

        self._over_hist = []
        if abs(err) < dead:
            log.info("[autotune] %d shots, median %+.0f ms (aim %+.0f) — "
                     "inside deadband, holding latency_ms=%d",
                     n, median, aim, int(config["latency_ms"]))
            return
        step = max(-15.0, min(15.0, 0.7 * err))   # scale, but damped
        new = int(round(config["latency_ms"] + step))
        new = max(0, min(300, new))
        if new == int(config["latency_ms"]):
            return
        log.info("[autotune] %d shots, median %+.0f ms vs aim %+.0f "
                 "= %+.0f ms (%s, +-%.0f) — latency_ms %d -> %d",
                 n, median, aim, err,
                 "late" if err > 0 else "early", stderr,
                 int(config["latency_ms"]), new)
        config["latency_ms"] = new
        save_config(config)

    def _find_tick(self, hsv, x_from, x_to, y0, y1):
        """Green tick inside the row band, to the right of the fill.
        Returns tick centre x (crop coords) or None."""
        x_from = max(0, x_from); x_to = min(hsv.shape[1], x_to)
        y0 = max(0, y0); y1 = min(hsv.shape[0], y1)
        if x_to - x_from < 3 or y1 - y0 < 3:
            return None
        g = _cv2.inRange(hsv[y0:y1, x_from:x_to],
                         self.TICK_LO, self.TICK_HI)
        
        # Filter out the thin green player indicator ring using connected components
        ncc, lab, st, ce = _cv2.connectedComponentsWithStats(g)
        best_tick = None
        best_area = 0
        for i in range(1, ncc):
            tx, ty, tw, th, ta = st[i]
            # Extremely sensitive tick detection for thin Hall of Fame green ticks
            if th >= 3 and ta >= 2:
                if ta > best_area:
                    best_area = ta
                    comp_mask = (lab == i)
                    xs = _np.nonzero(comp_mask.any(axis=0))[0]
                    best_tick = x_from + int(xs.mean())
        return best_tick

    def _locate(self, camera, sct, rect, full=False):
        """Find the purple bar. Returns
        ((bar_y0, bar_y1, front_x, end_x) or None, n_magenta_blobs).

        n_magenta_blobs is -1 when the backend had no new frame, so the
        caller can tell "nothing on screen" apart from "nothing to look at
        yet" — only the first of those is evidence there is no meter.

        Searches a horizontal BAND, not the whole window. Which axis to cut
        is not a guess: across 966 logged detections the bar appeared
        anywhere from x=68 to x=1443 of 1600, so horizontally it really is
        the full width and a centred crop would throw away about half of all
        shots. Vertically it is tight — every one of 63 confirmed bars sat
        between 0.649 and 0.853 of window height — so the default 0.60-0.90
        band drops 70% of the pixels with two bar-heights of margin at each
        edge, and `locate_full_after_ms` widens to the whole window if the
        band ever does come up empty.

        Full window, HSV, label everything:  12.3 ms single-threaded.
        Band, BGR box, label only the rows that carry magenta:  0.64 ms.
        """
        gx, gy, gw, gh = rect
        if full:
            by0, bh = 0, gh
        else:
            by0 = max(0, min(gh - 40,
                             int(gh * float(config.get("locate_band_lo",
                                                       0.60)))))
            bh = max(40, min(gh - by0,
                             int(gh * float(config.get("locate_band_hi",
                                                       0.90))) - by0))
        buf = self._grab(camera, sct, gx, gy + by0, gw, bh)
        if buf is None:
            return None, -1
        m = self._mask(buf)
        # While the meter is not up yet there is nothing bar-sized on screen,
        # and that is almost every iteration of the search. Bailing here for
        # the price of one countNonZero is what makes the loop affordable.
        if _cv2.countNonZero(m) < 40:
            return None, 0
        # Something magenta is up: narrow to the rows carrying it before
        # paying for connected components. The threshold has to stay low —
        # a fill that has only just appeared is a few px wide, so its rows
        # hold only a few magenta pixels each.
        rows = _cv2.reduce(m, 1, _cv2.REDUCE_SUM, dtype=_cv2.CV_32S)
        idx = _np.nonzero(rows[:, 0] >= 3 * 255)[0]
        if idx.size == 0:
            return None, 0
        r0 = max(0, int(idx[0]) - 4)
        r1 = min(m.shape[0], int(idx[-1]) + 5)
        ncc, lab, st, ce = _cv2.connectedComponentsWithStats(m[r0:r1])
        cands = []
        for i in range(1, ncc):
            x, y, w, h, a = st[i]
            if a >= 40 and 8 <= h <= 45 and w >= 3 and w <= 220:
                cands.append((a, x, y, w, h))

        span_px = float(config.get("persistent_span", 118))
        for a, x, y, w, h in sorted(cands, reverse=True)[:3]:
            solid = a / max(w * h, 1)
            if 12 <= h <= 40 and solid >= 0.70:
                end = x + int(span_px)
                y += r0 + by0          # band coords -> window coords
                return (y - 6, y + h + 6, x + w, end), len(cands)
        return None, len(cands)

    def _release(self, press, t_anchor, why, offset_ms):
        target = press + t_anchor + (offset_ms
                                     - config["meter_lead_ms"]) / 1000.0
        target = max(target, press + 0.15)
        target = min(target,
                     press + config["meter_max_ms"] / 1000.0 - 0.03)
        
        # Fast-path: cancel the background worker thread and release directly
        self.worker.cancel()

        # Outrank the game for the wait and the injection, and only for
        # those. The tracking loop that got us here runs a notch above
        # normal — see _vision_thread_priority.
        _boost_current_thread(THREAD_PRIORITY_TIME_CRITICAL)
        try:
            # Hold the invariant by construction, not by config discipline:
            # once the release is committed we must not block, because being
            # rescheduled onto a core is what produced every late outlier.
            # wait_s <= lookahead_ms, so spinning that whole window means
            # _coarse() is never reached on the LEAD path even if someone
            # tunes spin_margin_ms down or lookahead_ms up.
            guard_ms = float(config.get("lookahead_ms", 25)) + 1.0
            if config.get("frame_align", False):
                # frame_align can push wait_s up to half a frame past the
                # lookahead bound, so the no-block window has to cover it too.
                guard_ms += float(config.get("game_frame_ms", 15.0)) / 2.0
            spin_ms = max(float(config["spin_margin_ms"]), guard_ms)
            self._waiter.wait_until(target, spin_ms)
            t_rel = time.perf_counter()
            _inject_flick()
        finally:
            _boost_current_thread(_vision_thread_priority())

        held = (t_rel - press) * 1000.0
        err = held - (target - press) * 1000.0
        stats.record(err)

        self.last = f"{why} {t_anchor*1000:.0f} ms"
        if config["debug"]:
            log.info("[meter] %s at %.0f ms -> flicked up at %.2f ms (err %+.2f ms)",
                     why, t_anchor * 1000, (t_rel - press) * 1000, err)

    def _run(self) -> None:
        _boost_current_thread(_vision_thread_priority())
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
        # Keep OpenCV on one core. By default it fans every cvtColor,
        # inRange and connectedComponents out across all six, which on a
        # 6-thread CPU means the vision pass briefly owns the whole machine
        # — and the work below is now small enough that thread hand-off
        # costs more than it saves (the banded search measured 0.64 ms on
        # one thread against 0.87 ms on six).
        try:
            nt = int(config.get("cv_threads", 1))
            if nt > 0:
                _cv2.setNumThreads(nt)
        except Exception:
            pass
        camera = None
        if not config.get("force_mss", False):
            try:
                import dxcam
                # BGRA so both backends hand back the same 4-channel layout
                # and one mask path serves them, with no conversion either
                # side. A 3-channel frame would need its alpha re-added or
                # its stride fixed before inRange, and both cost more than
                # the mask.
                camera = dxcam.create(output_color="BGRA")
                log.info("[capture] backend: dxcam (GPU, BGRA)")
            except Exception as e:
                log.info("[capture] dxcam initialization failed: %s (falling back to mss)", e)
                camera = None
        else:
            log.info("[capture] backend: mss (CPU) forced by config")
        try:
            with _MSS() as sct:
                while not _shutdown.is_set():
                    if not self._armed.wait(0.5):
                        continue
                    self._armed.clear()
                    try:
                        self._track(camera, sct)
                    except Exception:
                        log.exception("[meter] tracking failed")
                        # Never let a tracking bug ride the 1200 ms safety
                        # guard — that holds K nearly twice as long as a
                        # normal shot and 2K scores it a hard "Late". Fall
                        # back to the same press-anchored hold the no-meter
                        # path uses, so a crash costs a normal-length shot
                        # instead of a guaranteed miss.
                        try:
                            fallback = (self._press
                                        + float(config["hold_ms"]) / 1000.0)
                            self.worker.retarget(
                                max(fallback, time.perf_counter() + 0.005))
                            log.warning("[meter] fell back to %d ms hold",
                                        int(config["hold_ms"]))
                        except Exception:
                            log.exception("[meter] fallback retarget failed")
        finally:
            if camera is not None:
                try:
                    camera.release()
                    log.info("[capture] released dxcam instance")
                except Exception:
                    pass

    def _track(self, camera, sct) -> None:
        _set_flick_speed(0.0)   # only the LEAD path may set this
        press = self._press
        now = time.perf_counter()
        if self._rect is None or now - self._rect_t > 5.0:
            self._rect = self._game_rect()
            self._rect_t = now
        rect = self._rect
        if rect is None:
            self.last = "no game window"
            return
        gx, gy, gw, gh = rect
        guard = press + config["meter_max_ms"] / 1000.0 - 0.06
        lock = None            # (band_y0, band_y1, front_x, target_x) window
        cands_seen = 0
        gave_up = False        # blind fallback armed, but still watching
        gave_up_at = 0.0       # when that fallback release is due
        seen = 0
        lost = 0
        hist = []              # (t, fill_w_px)
        tick_offset = None     # locked-in green tick offset from left edge
        dbg = (-1, -1)         # last front, target
        t_found = None
        front_at_found = None
        # Fill LENGTH at the first tracked frame. Velocity must be measured
        # from this, never from the absolute front: the meter is anchored to
        # the shooter, so 2K's camera pan slides the whole bar sideways while
        # it fills. The front then moves for two reasons and we would credit
        # all of it to fill speed — panning toward centre (shooter on the
        # left) inflated v by ~40%, and since lead_px and the scheduled wait
        # both divide by v, that fired the release early on every left-side
        # shot. front - left cancels the translation out.
        fw_at_found = None
        # DIAGNOSTIC ONLY — never feeds the release. bar_len_px is a hardcoded
        # guess because the real measurement (_find_tick) has always been dead
        # code, so we cannot tell a mistuned constant apart from a target that
        # legitimately moves shot to shot. This probes the green tick ONCE per
        # shot and logs where it actually is. If tick_len is stable across
        # shots, a constant is the right shape and only its value is wrong; if
        # it wanders, no constant can ever be right and the target has to be
        # measured per shot. Costs one cvtColor + one CC pass on a small crop,
        # once per shot, and only when the tick_probe flag is on.
        tick_probed = False
        t_fw = None
        prev_fill_w = None     # staircase step tracking (see fill_est)
        step_gaps = []         # observed gaps between fill steps = game frames
        # Running sums for a least-squares fit of fill vs time. Both of the
        # old velocity estimates were TWO-POINT: they subtracted the ends of
        # a window and threw away the ~55 samples between. The fill is a
        # staircase, so quantising either endpoint by one step is a ~9%
        # error on v over a 60 ms window, which becomes ~2 ms of release
        # error once multiplied by the prediction horizon. Fitting a line
        # through every sample averages the staircase out instead.
        lsq_n = 0
        lsq_sx = lsq_sy = lsq_sxx = lsq_sxy = 0.0
        # Alpha-beta tracker over the same fill length: two scalars of state,
        # one update per frame, no history. See the update below for why it
        # beats the fit.
        ab_x = None            # smoothed fill length, px
        ab_v = 0.0             # fill speed, px/ms
        ab_t = None
        last_step_fill = None  # fill at the last REAL (>=2px) advance
        last_step_t = None
        fill_at_step = None
        t_step = None
        t_prev = None
        active_ms = 0.0        # elapsed time with capture stalls removed
        stalls_ms = 0.0        # how much stall we discarded, for the log
        while not _shutdown.is_set():
            now = time.perf_counter()
            if now >= guard:
                self.last = ("meter never found" if lock is None
                             else "no release signal")
                # Do not simply return: the worker's own deadline is
                # meter_max_ms, so returning here holds K for the full
                # 1200 ms — nearly twice a normal shot and a guaranteed
                # Late. Nothing better is knowable this late, so release
                # now instead of sitting on the remaining 60 ms.
                if not gave_up:
                    try:
                        self.worker.retarget(now + 0.005)
                    except Exception:
                        log.exception("[meter] guard retarget failed")
                log.warning("[meter] %s — safety release at %.0f ms "
                            "(last front %d tick %d)",
                            self.last, (now - press) * 1000.0,
                            dbg[0], dbg[1])
                return
            if lock is None:
                # Stop once the blind fallback has fired. Without this the
                # thread spins to the 1200 ms guard and logs a "meter never
                # found — safety release" line for a shot that already
                # released correctly at hold_ms.
                if gave_up and now >= gave_up_at - 0.02:
                    return
                if now - press < 0.080:  # Ignore first 80ms to avoid fading meters or quick false locks
                    time.sleep(0.002)
                    continue
                # Give the band its chance, then widen. A meter outside the
                # band costs a slightly later lock, never a lost shot.
                found, ncand = self._locate(
                    camera, sct, rect,
                    full=(now - press) * 1000.0
                    > float(config.get("locate_full_after_ms", 220)))
                if ncand < 0:            # backend has no new frame yet
                    self._pace()
                    continue
                cands_seen = max(cands_seen, ncand)
                if found:
                    lock = found
                    seen = 1
                    t_found = now
                    front_at_found = found[2]
                    # `found[3] - found[2]` is NOT a measurement of the meter.
                    # _locate returns found[3] = fill_left + persistent_span,
                    # a constant, so this difference is just
                    # `persistent_span - fill_width_at_lock`. Kept only to
                    # feed the legacy bar_len_px<=0 path.
                    _ts = found[3] - found[2]
                    self._tick_span = _ts if 60 <= _ts <= 200 else None
                    if config["debug"]:
                        # bar_h is the one real scale signal available here
                        # (found = (y-6, y+h+6, ...)). If the meter genuinely
                        # renders bigger up close, THIS is what moves — log it
                        # so a future session can answer that from data
                        # instead of scaling off lock-timing noise.
                        _bar_h = found[1] - found[0] - 12
                        log.info("[meter] purple bar found at +%.0f ms "
                                 "(fill %d, w_at_lock %d, bar_h %d, "
                                 "target_len %g)",
                                 (now - press) * 1000, found[2],
                                 int(config.get("persistent_span", 118)) - _ts,
                                 _bar_h,
                                 float(config.get("bar_len_px", 116)))
                elif (not gave_up
                      and now - press > float(config["no_meter_ms"]) / 1000.0):
                    # No LOCK this shot. Deliberately not gated on
                    # `cands_seen == 0` any more: magenta on screen that
                    # never passes the solidity/height test is not a meter
                    # we can time, and treating it as "still coming" armed
                    # no fallback at all — the shot then rode the 1200 ms
                    # safety hold, which 2K scores a hard Late. Seen in a
                    # log where the bar locked at +348 ms, was lost before
                    # a single tracked frame, and every re-locate after it
                    # produced rejected candidates only.
                    # No meter this shot. Fall back to a normal-length hold
                    # anchored on the PRESS, not on the moment we gave up —
                    # otherwise the give-up delay is added to the hold and
                    # the shot releases far later than a metered one.
                    target = press + float(config["hold_ms"]) / 1000.0
                    self.worker.retarget(max(target, now + 0.005))
                    self.last = "no meter — quick abort"
                    if config["debug"]:
                        log.info("[meter] no lock within %d ms (cands %d) — "
                                 "holding %d ms total (release in %.0f ms)",
                                 int(config["no_meter_ms"]), cands_seen,
                                 int(config["hold_ms"]),
                                 (target - now) * 1000.0)
                    # Arm the fallback but KEEP LOOKING. The release isn't
                    # until press+hold_ms, and meters have been observed
                    # appearing as late as +596 ms — giving up at
                    # no_meter_ms threw away every meter that landed in
                    # the gap between the two, for no gain. If one shows
                    # up before the fallback fires, tracking takes over
                    # and retargets; if not, the blind hold stands.
                    gave_up = True
                    gave_up_at = target
                continue
            band_y0, band_y1, front_x, target_x = lock
            # Tight crop around the BAR ONLY — a small grab keeps the
            # capture+process loop short, which is pure release latency.
            cx0 = max(0, target_x - int(gw * self.BAR_FRAC) - 40)
            cx1 = min(gw, target_x + 35)
            cy0 = max(0, band_y0 - 20)
            cy1 = min(gh, band_y1 + 20)
            buf = self._grab(camera, sct, gx + cx0, gy + cy0,
                             cx1 - cx0, cy1 - cy0)
            if buf is None:
                # No new rendered frame. Not a lost meter — the fill cannot
                # have moved, so counting it as one would trip the
                # meter-lost release on a backend that is simply idle.
                self._pace()
                continue
            t_now = time.perf_counter() - press
            m = self._mask(buf)
            npx = _cv2.countNonZero(m)
            if npx >= 25:
                seen += 1
                lost = 0
                # One pass for the whole bounding box: same numbers as the
                # two np.nonzero(any(...)) reductions this replaces, 0.020 ms
                # against 0.140 ms for the old mask-and-scan.
                _bx, _by, _bw, _bh = _cv2.boundingRect(m)
                front = cx0 + _bx + _bw - 1
                left = cx0 + _bx
                # Fill LENGTH, not front position — see fw_at_found above.
                fill_w = front - left
                if fw_at_found is None:
                    fw_at_found = fill_w
                    t_fw = now
                    t_prev = now
                else:
                    # Cap each frame's contribution: anything longer than
                    # stall_cap_ms is a dropped/late frame, not fill time.
                    _cap = float(config.get("stall_cap_ms", 20)) / 1000.0
                    _gap = now - t_prev
                    if _gap > _cap:
                        stalls_ms += (_gap - _cap) * 1000.0
                    active_ms += min(_gap, _cap) * 1000.0
                    t_prev = now
                # Static tracking safety: if we have tracked for > 150ms but the fill has not grown by at least 8px,
                # we are likely locked onto a static background element or jersey.
                if now - t_fw > 0.150 and fill_w - fw_at_found < 8:
                    self._release(press, t_now, "static lock-on abort", 0)
                    return
                band_y0 = cy0 + _by - 20
                band_y1 = cy0 + _by + _bh - 1 + 20
                # Fixed target length — see bar_len_px in DEFAULT_CONFIG for
                # why this must NOT be derived from the lock-time span.
                # `left` is still tracked live every frame, so the target
                # keeps riding the camera pan; only its LENGTH is constant.
                expected_len = float(config.get("bar_len_px", 116))
                if expected_len <= 0:            # legacy per-shot scaling
                    expected_len = (
                        float(self._tick_span)
                        * float(config.get("arrow_ratio", 1.227))
                        if getattr(self, "_tick_span", None)
                        else float(config.get("persistent_span", 130)))
                # --- tick probe (diagnostic, see tick_probed above) --------
                # Fire once, past halfway: early on the fill the tick can sit
                # outside the crop, and we want it while the bar is clearly
                # on screen. Wrapped broadly — a probe that throws must never
                # cost a shot, since nothing downstream depends on it.
                if (config.get("tick_probe", False) and not tick_probed
                        and expected_len > 0
                        and fill_w >= 0.5 * expected_len):
                    tick_probed = True
                    try:
                        _hsv = _cv2.cvtColor(buf, _cv2.COLOR_BGR2HSV)
                        _tk = self._find_tick(_hsv, _bx + _bw + 2,
                                              _hsv.shape[1],
                                              _by, _by + _bh)
                        if _tk is not None:
                            # _find_tick returns crop coords; `left` is in
                            # window coords, so lift the tick the same way
                            # front/left were lifted before differencing.
                            _tick_x = cx0 + _tk
                            log.info("[tickprobe] tick at %d -> tick_len %d "
                                     "(target_len %g, delta %+d px, "
                                     "fill_w %d, bar_h %d)",
                                     _tick_x, _tick_x - left, expected_len,
                                     (_tick_x - left) - expected_len,
                                     fill_w, _bh)
                        else:
                            log.info("[tickprobe] no green tick found "
                                     "(fill_w %d, bar_h %d, scan %d..%d)",
                                     fill_w, _bh, _bx + _bw + 2,
                                     _hsv.shape[1])
                    except Exception as exc:
                        log.info("[tickprobe] failed: %r", exc)
                target = left + expected_len
                lock = (band_y0, band_y1, front, target)
                dist = target - front
                dbg = (front, target)
                hist.append((t_now, fill_w))
                while len(hist) > 1 and t_now - hist[0][0] > 0.060:
                    hist.pop(0)
                # Release when the fill is `latency_ms` short of full,
                # measured in the fill's OWN speed. The screen we read
                # lags the game's internal state by a fixed time, so the
                # compensation must be a time converted to pixels — a
                # fixed % cannot track it when the fill speed changes.
                span = expected_len
                # Timestamp each staircase step here; the extrapolation
                # itself happens below, once v is known.
                if fill_w != prev_fill_w:
                    # Frame-period sampling. Require a REAL advance: fill_w
                    # is front-minus-left and the camera pans both edges, so
                    # a 1 px pan drift looks like a step and would drag the
                    # measured period below the true frame time. One rendered
                    # frame moves the fill ~5 px, so >=2 px separates a real
                    # step from pan noise.
                    if last_step_fill is None:
                        last_step_fill, last_step_t = fill_w, now
                    elif fill_w - last_step_fill >= 2:
                        _g = now - last_step_t
                        if 0.005 < _g < 0.040:
                            step_gaps.append(_g)
                        last_step_fill, last_step_t = fill_w, now
                    prev_fill_w = fill_w
                    fill_at_step = fill_w
                    t_step = now
                remain = span - fill_w
                # Alpha-beta tracker on the fill length. Predict where the
                # fill should be, measure, correct position and speed by a
                # fixed fraction of the miss. Two scalars, no history, no
                # accumulators — and unlike the growing least-squares fit
                # below it is not a whole-fill average, so it follows a fill
                # whose speed changes instead of trailing it.
                #
                # Scored on release-time error against simulated fills
                # (constant slow, constant fast, accelerating, decelerating,
                # speed-kink at 250 ms), sampling one frame per render as
                # the fixed capture path now does, the tracker beat the fit
                # on all five. On the speed-kink it was 3.4 ms rms against
                # the fit's 11.6 ms; on a constant fill 2.0 against 3.3.
                # Gains from a sweep: alpha 0.30 with beta 0.105 gave the
                # best worst-case (4.2 ms) and the best mean (3.1 ms).
                #
                # x_hat is also continuous between rendered frames, which is
                # what `subframe_fill` was extrapolating by hand — when this
                # filter is driving, that hack is redundant and skipped.
                if ab_x is None:
                    ab_x, ab_t, ab_v = float(fill_w), now, 0.0
                else:
                    _abdt = (now - ab_t) * 1000.0
                    if _abdt > 0.5:
                        ab_t = now
                        _pred = ab_x + ab_v * _abdt
                        _resid = fill_w - _pred
                        ab_x = _pred + float(config.get("ab_alpha", 0.30)) * _resid
                        ab_v = min(1.0, max(0.0, ab_v + (
                            float(config.get("ab_beta", 0.105))
                            / _abdt) * _resid))
                # px/ms from the last few fill-length samples (60ms window)
                lsq_n += 1
                lsq_sx += t_now
                lsq_sy += fill_w
                lsq_sxx += t_now * t_now
                lsq_sxy += t_now * fill_w

                v = 0.0
                if len(hist) >= 2:
                    dt = (hist[-1][0] - hist[0][0]) * 1000.0
                    adv = hist[-1][1] - hist[0][1]
                    if dt > 15 and adv > 0:
                        v = adv / dt
                # The 60 ms window is only 3-5 frames, so one noisy reading
                # skews it badly — and both lead_px and the scheduled wait
                # divide by v, so the error lands on the release twice.
                # Prefer a long baseline over the whole fill: same slope,
                # far less noise.
                if t_fw is not None:
                    # Elapsed time with capture stalls excluded. Wall clock
                    # keeps running when frames stop arriving, but the fill
                    # we can SEE does not advance — so a stall inflates the
                    # denominator only, reads v too low, and a low v shrinks
                    # lead_px and releases late. One 70 ms hiccup did exactly
                    # that: v 0.27 (session low) and a 725 ms hold, ~95 ms
                    # longer than every other shot. Capping each frame's
                    # contribution keeps a stall from being counted as time
                    # the fill was observed standing still.
                    dt_long = active_ms
                    adv_long = fill_w - fw_at_found
                    if dt_long > 120 and adv_long > 0:
                        v_long = adv_long / dt_long
                        w = float(config.get("v_long_weight", 0.75))
                        v = (w * v_long + (1.0 - w) * v) if v > 0.05 else v_long
                if v > 0.05:
                    # Least-squares slope wins when we have enough of the
                    # fill to fit. Guarded on the denominator and on a sane
                    # result so a degenerate fit can never poison the
                    # release; the two-point estimate stays as the fallback.
                    v_fit = 0.0
                    if config.get("v_lsq", True) and lsq_n >= 12:
                        _den = lsq_n * lsq_sxx - lsq_sx * lsq_sx
                        if _den > 1e-9:
                            _sl = (lsq_n * lsq_sxy - lsq_sx * lsq_sy) / _den
                            _sl /= 1000.0          # px/s -> px/ms
                            if 0.05 < _sl < 1.0:
                                v_fit = _sl
                                v = _sl
                    self._last_v = v
                elif hasattr(self, "_last_v"):
                    v = self._last_v
                else:
                    v = 0.30  # Default fallback
                # Hand over to the tracker once it has settled. A few frames
                # of warm-up matter: it starts at v=0 and the first residual
                # is the whole fill, so an early reading is meaningless.
                use_ab = (config.get("v_filter", "ab") == "ab"
                          and seen >= 4 and 0.05 < ab_v < 1.0)
                if use_ab:
                    v = ab_v
                    self._last_v = v
                # The game renders at ~60 fps while we capture at ~200 Hz,
                # so fill_w is a staircase: it holds for ~3 captures, then
                # jumps ~5 px. Reading it raw means the fill is stale by
                # 0-16.7 ms depending purely on where in the game's frame
                # we happened to look — ~8 ms late on average with ~5 ms
                # of spread. The constant part gets absorbed by
                # arrow_ratio; the SPREAD is what makes one setting land
                # early on one shot and late on the next. Extrapolating
                # from the last step keeps the estimate continuous
                # instead of lagging the last drawn frame.
                fill_est = fill_w
                if use_ab:
                    # x_hat already is the sub-frame estimate, updated at
                    # this instant, so no extrapolation window to cap.
                    fill_est = ab_x
                    remain = span - fill_est
                elif (config.get("subframe_fill", True)
                        and v > 0.05 and t_step is not None):
                    # Never extrapolate past one game frame: a missing
                    # step means the fill stopped, not that it kept going.
                    age_ms = min((now - t_step) * 1000.0,
                                 float(config.get("step_cap_ms", 18)))
                    fill_est = fill_w + v * age_ms
                    remain = span - fill_est
                # Use the configured latency for both jumpshots and layups to keep timing consistent
                lead_px = v * float(config["latency_ms"])
                # lead_px > 0 overrides that with a FIXED pixel lead, i.e.
                # release at a constant fill length whatever the speed. See
                # the config entry: v*latency_ms is a lead in TIME, and it
                # smears the release across ~3 px over the speed range this
                # rig shoots, which is most of the good window.
                _lp = float(config.get("lead_px", 0))
                if _lp > 0:
                    lead_px = _lp
                    # Slow fills need MORE lead than a constant, even though
                    # a constant already beats a lead in time. ONE-SIDED on
                    # purpose: above lead_v_ref this adds nothing, so the
                    # eight labelled greats at v 0.29-0.34 / 99.0 px cannot
                    # be disturbed by it. See lead_v_gain in the config.
                    _vr = float(config.get("lead_v_ref", 0.29))
                    _vg = float(config.get("lead_v_gain", 150.0))
                    if _vg > 0 and v < _vr:
                        lead_px += min(float(config.get("lead_v_max", 6.0)),
                                       _vg * (_vr - v))
                cap_px = span * float(config["release_pct"]) / 100.0
                # Stall detection: front has not advanced by more than 1px for at least 30ms near the end (>= 95% full)
                stalled = False
                if len(hist) >= 2:
                    t_latest, pos_latest = hist[-1]
                    for t_old, pos_old in reversed(hist[:-1]):
                        if t_latest - t_old >= 0.030:
                            if pos_latest - pos_old <= 1 and fill_w >= 0.95 * span:
                                stalled = True
                            break
                base_min_pct = float(config.get("min_lead_pct", 82))
                if v >= 0.41:
                    # Fast fills need a BIGGER lead in pixels, not a later
                    # release — forcing 95% here made every fast meter fire
                    # after the lead the speed had already earned.
                    min_lead_pct = max(base_min_pct,
                                       float(config.get("fast_min_lead_pct",
                                                        88.0))) / 100.0
                else:
                    min_lead_pct = base_min_pct / 100.0
                # The floor is a FILL FRACTION, but it can only be tested on
                # frames that actually arrive — so its effective position is
                # up to one frame of fill past its nominal one. If that lands
                # after the release point, `remain` is already below lead_px
                # when the trigger fires, wait_s clamps to 0, and the flick
                # goes out late by however far the frame overshot. It is
                # visible in the log as `sched+0.0ms` on a LEAD release.
                #
                # Measured: floor 82% put the gate at 97.2 px against a
                # 99.0 px release point — 1.8 px, half a frame at v0.277.
                # One shot's first qualifying frame landed at 100.4 px and
                # fired 1.4 px late; the shot before it qualified at 98.6 px
                # and was fine. That is a coin flip decided by frame phase,
                # so hold the invariant by construction instead of by
                # config discipline: keep the floor clear of the release
                # point. Only ever LOWERS the floor, and only when the
                # geometry demands it — the lookahead term still bounds how
                # early the trigger can go.
                #
                # THREE frames, and the third one is not padding. One frame
                # is the right margin for frame PHASE alone, and it is not
                # enough in practice: a shot logging `stall14ms` put a 14 ms
                # capture stall on top of a 12.5 ms frame gap, so the next
                # usable frame arrived two frames late — at 104.3 px, remain
                # already under the lead — and the release clamped ~6 px
                # late. Stalls are normal on this capture path, so the
                # margin has to cover one.
                #
                # But widening THIS alone fixes nothing, which is worth
                # knowing before someone tunes it: simulated adversarially
                # (stall placed at every frame, over every phase), 2 frames
                # still clamped to 100.5 px because the floor was no longer
                # what bound the trigger — `lookahead_ms` was. The trigger
                # also cannot fire until remain <= lead_px + v*lookahead,
                # and at 25 ms that window is ~7 px, narrower than one frame
                # plus a stall. So the fix is a PAIR: three frames here, so
                # the floor drops out of the way, and lookahead_ms 35 so the
                # trigger is allowed early enough to absorb the stall. With
                # both, worst case is exactly 99.0 px at stalls of 0, 14 and
                # 20 ms; with either alone it still clamps.
                _room = (span - lead_px
                         - 3.0 * v * float(config.get("game_frame_ms", 13.4)))
                if _room > 0:
                    min_lead_pct = min(min_lead_pct, _room / span)
                # Sub-frame scheduling: fire on the frame BEFORE the ideal
                # crossing and schedule the release for the exact moment it
                # happens. Waiting for `remain <= lead_px` meant the trigger
                # landed wherever the next capture happened to fall — up to a
                # frame of fill (~4-5px, ~15 ms) past the ideal point, and
                # always on the late side.
                look_ms = float(config.get("lookahead_ms", 25))
                lead_ready = (v > 0 and fill_w >= min_lead_pct * span
                              and remain <= lead_px + v * look_ms)
                if seen >= 2 and t_now >= 0.10 and (
                        lead_ready or fill_w >= cap_px - 1 or stalled):
                    pct = 100.0 * fill_w / span
                    why = ("LEAD" if lead_ready
                           else ("CAP" if not stalled else "STALL"))
                    # Seconds until remain == lead_px at the current speed.
                    wait_s = 0.0
                    if lead_ready and v > 0:
                        wait_s = max(0.0, (remain - lead_px) / v / 1000.0)
                    # Measure the game's render period, every shot. Every step
                    # in fill_w IS a frame boundary, so step_gaps is a direct
                    # read of the render clock and needs no configured
                    # constant. Measured unconditionally, because the phase
                    # below is worth logging whether or not we align to it.
                    frame = None
                    frame_ms = 0.0
                    frame_spread = "-"
                    if len(step_gaps) >= 6:
                        _sg = sorted(step_gaps)
                        _med = _sg[len(_sg) // 2]
                        # Spread of the step gaps. A real render clock is
                        # tight (p25~p50~p75); a wide spread means something
                        # is splitting frames and the median is an
                        # underestimate, not a frame time.
                        _q25 = _sg[len(_sg) // 4]
                        _q75 = _sg[len(_sg) * 3 // 4]
                        frame_spread = "%.1f/%.1f" % (_q25 * 1000.0,
                                                      _q75 * 1000.0)
                        if (0.0100 <= _med <= 0.0230
                                and (_q75 - _q25) <= 0.0040):
                            frame = _med
                            frame_ms = _med * 1000.0
                    # Snapping to frame centre is off by default and should
                    # stay off — see frame_align in DEFAULT_CONFIG for why it
                    # can only add frame flips, never remove them. Kept
                    # switchable so the claim stays falsifiable in-game.
                    align_ms = 0.0
                    if (config.get("frame_align", False)
                            and lead_ready and t_step is not None
                            and frame is not None):
                        d = ((press + t_now + wait_s) - t_step) % frame
                        # d < frame, so shift > -frame/2 always: it is the
                        # NEAREST centre, never a whole frame away.
                        shift = frame / 2.0 - d
                        # Clamping to 0 loses the alignment, but only when
                        # the aligned instant is already behind us — and then
                        # we are still inside half a frame of it. Adding a
                        # whole frame instead would be a guaranteed frame
                        # late, which is far worse.
                        wait_s = max(0.0, wait_s + shift)
                        align_ms = shift * 1000.0
                    # Where inside the frame the release lands, as a fraction
                    # of it, measured from the last observed fill step. THIS
                    # is the number that decides early vs late: the model says
                    # it is near-constant across shots, and a miss is a shot
                    # whose phase sat near 0.00/1.00 and let noise tip it over
                    # the boundary. latency_ms shifts it; 0.5 is the target.
                    phase = -1.0
                    if frame is not None and t_step is not None:
                        phase = (((press + t_now + wait_s) - t_step)
                                 % frame) / frame
                    # How far the whole bar slid sideways since detection —
                    # this is the camera pan the old front-based velocity
                    # was mistaking for fill speed.
                    pan_px = (left - front_at_found
                              if front_at_found is not None else 0.0)
                    # Mean interval between usable frames. This sets the
                    # floor on release precision: we can only ever act on
                    # the frames we actually get, and lookahead_ms has to
                    # be at least one interval to fire BEFORE the crossing.
                    frame_dt = (active_ms / max(1, seen - 1)) if seen > 1 else 0.0
                    # Publish the speed so the injector can choose a gesture
                    # from it — this is the ONLY path with a trustworthy v,
                    # so the fallback releases deliberately leave it unset.
                    _set_flick_speed(v)
                    self._release(
                        press, t_now + wait_s,
                        f"{why} {pct:.0f}% rem{remain:.1f}px "
                        f"v{v:.2f}px/ms comp{int(config['latency_ms'])}ms "
                        f"lead{lead_px:.1f}px "
                        f"sched+{wait_s * 1000:.1f}ms x{int(left)} "
                        f"pan{pan_px:+.0f}px stall{stalls_ms:.0f}ms "
                        f"frm{seen} dt{frame_dt:.1f}ms sub{(fill_est - fill_w):+.1f}px "
                        f"align{align_ms:+.1f}ms fr{frame_ms:.1f}ms"
                        f"[{frame_spread}]n{len(step_gaps)} "
                        f"ph{('%.2f' % phase) if phase >= 0 else '-'} "
                        f"{'ab' if use_ab else 'lsq'}"
                        f"{f' abx{ab_x:.1f}px abv{ab_v:.3f}' if ab_x is not None else ''}",
                        config["meter_offset_ms"])
                    
                    # --- Post-release latency & error measurement ---
                    # Nothing here changes this shot: it measures where the
                    # fill actually stopped, logs it, and feeds autotune.
                    # It costs a solid `post_measure_ms` of capture right as
                    # the shot animates, so it is off by default — set
                    # post_measure_ms to ~150 to get the [overshoot] line
                    # back while tuning, with or without autotune on.
                    post_ms = float(config.get("post_measure_ms", 0))
                    if post_ms <= 0:
                        return
                    try:
                        t_inject = time.perf_counter()
                        t_limit = t_inject + post_ms / 1000.0
                        post_hist = []

                        while time.perf_counter() < t_limit:
                            buf_post = self._grab(camera, sct, gx + cx0, gy + cy0, cx1 - cx0, cy1 - cy0)
                            if buf_post is None:
                                self._pace()
                                continue
                            t_frame = time.perf_counter()
                            m_post = self._mask(buf_post)
                            if _cv2.countNonZero(m_post) >= 25:
                                _px, _py, _pw, _ph = _cv2.boundingRect(m_post)
                                curr_front = cx0 + _px + _pw - 1
                                post_hist.append((t_frame, curr_front))
                            time.sleep(0.001)

                        if post_hist:
                            # First frame the front never grows past again.
                            # One reverse pass carrying the suffix maximum —
                            # same answer as the old nested scan, which was
                            # quadratic in the number of sampled frames.
                            stop_idx = None
                            suffix_max = -1 << 30
                            for i in range(len(post_hist) - 1, -1, -1):
                                if suffix_max <= post_hist[i][1] + 1:
                                    stop_idx = i
                                suffix_max = max(suffix_max, post_hist[i][1])

                            if stop_idx is not None:
                                t_observed = post_hist[stop_idx][0]
                                max_pos = post_hist[stop_idx][1]
                                measured_latency_ms = (t_observed - t_inject) * 1000.0
                                
                                # Determine the shot timing direction
                                # If it stopped at or past the target boundary
                                if max_pos >= target - 1:
                                    # If it stabilized almost immediately, it hit the boundary before release registered (LATE)
                                    if measured_latency_ms < 25.0:
                                        direction = "Late"
                                        time_error_ms = -15.0  # Fixed error to trigger a step adjustment
                                    else:
                                        direction = "Perfect"
                                        time_error_ms = 0.0
                                else:
                                    direction = "Early"
                                    # Calculate actual time error based on fill speed
                                    time_error_ms = (target - max_pos) / v if v > 0.05 else 0.0

                                # Debug-only, and no longer says "Perfect":
                                # this check only asks whether the fill
                                # reached `target`, which magenta always
                                # does, so it labelled shots 2K scored
                                # "Slightly Late" as perfect. Kept as raw
                                # numbers for diagnostics; never trust the
                                # verdict, and never tune from it.
                                # The VERDICT here is meaningless (it only
                                # asks whether the fill reached target,
                                # which magenta always does). But
                                # stopped-minus-target is a real, direct
                                # measurement of how late the release
                                # landed, in pixels of fill — the only
                                # ground truth available without reading
                                # 2K's badge. Positive = overshot = late.
                                over = max_pos - target
                                over_ms = over / v if v > 0.05 else 0.0
                                # Only feed the controller shots it actually
                                # tracked. A meter picked up late gives a
                                # truncated span and a 2-frame velocity guess,
                                # and its overshoot is noise dressed up as a
                                # measurement — one of those in a batch of 8
                                # can drag latency_ms 15 ms off target.
                                min_frm = int(config.get("trust_min_frames", 8))
                                min_span = int(config.get("trust_min_span", 95))
                                bad = ("frames" if seen < min_frm
                                       else "span" if span < min_span
                                       else None)
                                log.info("[overshoot] %+d px (%+.0f ms) "
                                         "target %d stopped %d "
                                         "span %d frm %d%s",
                                         over, over_ms, target, max_pos,
                                         span, seen,
                                         f" — UNTRUSTED ({bad})" if bad else "")
                                if bad:
                                    log.info("[autotune] sample discarded: "
                                             "meter was not tracked cleanly")
                                else:
                                    self._autotune(over_ms)
                                
                                # --- Closed-loop Auto-calibration ---
                                # Only calibrate on jumpshots to avoid layup anomalies
                                # Disabled by default: the "Perfect" test
                                # below only checks that the fill reached
                                # `target`, which magenta always does — so it
                                # scores late shots as perfect and walks
                                # latency_ms the wrong way. Set "autocal":
                                # true to re-enable.
                                if config.get("autocal", False) and 0.05 < v < 0.41:
                                    if direction == "Early":
                                        # Proportional correction: release later
                                        correction = -time_error_ms * 0.5
                                    elif direction == "Late":
                                        # Fixed step correction: release earlier
                                        correction = +4.0
                                    else:
                                        correction = 0.0
                                    
                                    if abs(correction) > 0.1:
                                        old_latency = config["latency_ms"]
                                        new_latency = max(10.0, min(80.0, old_latency + correction))
                                        new_latency = round(new_latency, 1)
                                        if new_latency != old_latency:
                                            config["latency_ms"] = new_latency
                                            save_config(config)
                                            log.info("[measure] Auto-calibrated latency: %.1f ms -> %.1f ms (applied %+.1f ms correction)",
                                                     old_latency, new_latency, correction)
                    except Exception as e:
                        log.debug("Post-release measurement failed: %s", e)
                    
                    return
                # Optional look-ahead (config "predict": true). Off by
                # default — prediction estimates the arrival time and
                # any error in it shows up as early/late shots.
                if config["predict"] and len(hist) >= 3:
                    dt = hist[-1][0] - hist[0][0]
                    adv = hist[-1][1] - hist[0][1]   # px advanced
                    if dt > 0.02 and adv >= 2:
                        v = adv / dt                 # px per second
                        if v >= 25 and dist > 0 and dist / v <= 0.30:
                            self._release(
                                press, t_now + dist / v,
                                f"FULL predicted d{dist} v{v:.0f}",
                                config["meter_offset_ms"])
                            return
            else:
                lost += 1
                # Meter vanished mid-fill: release immediately rather
                # than hold blind.
                if seen >= 4 and lost >= 3:
                    self._release(press, t_now, "meter lost", 0)
                    return
                if not gave_up and lost >= 3:
                    # Locked, then lost the bar before we ever tracked it
                    # (seen < 4, so the branch above cannot fire and the
                    # fill estimate is worthless). Releasing here would be
                    # early; doing nothing rode the 1200 ms guard and was
                    # a hard Late. Arm the same press-anchored normal hold
                    # the no-lock path uses and KEEP LOOKING — if the bar
                    # comes back, tracking retargets over this.
                    target = press + float(config["hold_ms"]) / 1000.0
                    self.worker.retarget(max(target, now + 0.005))
                    gave_up = True
                    gave_up_at = target
                    self.last = "lock lost — blind hold armed"
                    if config["debug"]:
                        log.info("[meter] lock lost after %d frames — "
                                 "holding %d ms total (release in %.0f ms)",
                                 seen, int(config["hold_ms"]),
                                 (target - now) * 1000.0)
                if lost >= 10:
                    lock = None
                    hist = []
                    seen = 0
                    lost = 0
                    # Whatever we re-lock onto is a fresh bar; carrying the
                    # old length and speed into it would have the tracker
                    # chasing a step change that never happened.
                    ab_x = ab_t = None
                    ab_v = 0.0


worker = None
meter_eye = None

# --------------------------------------------------------------------------- #
# K hook — swallow both the press and the release; the press becomes a pro
# stick pull-down and the meter decides when the stick flicks up.
# --------------------------------------------------------------------------- #
def _hook(event) -> bool:
    now = time.perf_counter()
    try:
        return _hook_impl(event, now)
    except Exception:
        log.exception("[hook] error")
        return True


def _game_focused() -> bool:
    """Is NBA 2K26 the foreground window?

    Typing a 'k' in any other app used to arm a shot: the hook is global,
    so it fired while the game sat idle in the background, the meter was
    never found, and the log filled with phantom failures. Cheap Win32
    call, and on any error it returns True so a broken check can never
    silently stop the tool from working in-game.
    """
    try:
        u32 = ctypes.windll.user32
        fg = u32.GetForegroundWindow()
        if not fg:
            return True
        n = u32.GetWindowTextLengthW(fg)
        buf = ctypes.create_unicode_buffer(n + 1)
        u32.GetWindowTextW(fg, buf, n + 1)
        return config["game_window"].lower() in buf.value.lower()
    except Exception:
        return True


def _hook_impl(event, now: float) -> bool:
    global _physically_held, _passthrough_held, _cycle_active
    if _injecting:
        return True

    if event.event_type == keyboard.KEY_DOWN:
        if _physically_held:
            return _passthrough_held      # auto-repeat
        _physically_held = True
        if not active or not _game_focused():
            _passthrough_held = True
            return True
        with _state_lock:
            if not _cycle_active:
                _cycle_active = True
                # Pull the stick down FIRST: this is what actually starts
                # the shot, and every deadline downstream is measured from
                # `now` (stamped at hook entry), so the gap between the two
                # is pure error. One syscall, ~20 us.
                _inject_shot_start()
                worker.schedule(now, config["meter_max_ms"])
                if meter_eye is not None:
                    meter_eye.arm(now)
                if config["debug"]:
                    log.info("[trigger] K down -> pro stick DOWN, tempo "
                             "running (meter read)")
                # K must NOT reach the game: it would fire a button shot
                # next to the stick one. The stick is the whole gesture.
                return False
        return False                      # tap during active cycle

    # KEY_UP
    was_held = _physically_held
    _physically_held = False
    if _passthrough_held:
        _passthrough_held = False
        return True
    if not was_held and not _cycle_active:
        return True
    return False                          # the stick owns the release


# --------------------------------------------------------------------------- #
# Config hot-reload
# --------------------------------------------------------------------------- #
_HOT_FIELDS = ("flick_hold_ms", "flick_gap_ms", "flick_gap_v_max",
               # pro_stick_down_key / pro_stick_up_key are NOT here: the
               # SendInput batches are packed from them once at start-up.
               "hold_ms", "no_meter_ms", "fast_min_lead_pct", "arrow_ratio",
               "bar_len_px", "persistent_span", "lookahead_ms", "v_long_weight", "v_lsq", "stall_cap_ms", "subframe_fill", "step_cap_ms", "frame_align", "game_frame_ms", "autotune", "autotune_shots", "autotune_deadband_ms", "autotune_target_ms", "trust_min_frames", "trust_min_span", "spin_margin_ms", "latency_ms", "lead_px", "lead_v_ref",
               "lead_v_gain", "lead_v_max", "release_pct", "min_lead_pct",
               "meter_offset_ms", "meter_lead_ms", "meter_max_ms",
               "predict", "force_mss", "debug", "tick_probe",
               # Tunable mid-session. cv_threads, process_priority and
               # vision_priority are deliberately NOT here: they are applied
               # once at thread start and a live change would not take.
               "poll_sleep_ms", "post_measure_ms", "mask_bgr", "mag_rb_min",
               "mag_g_max", "locate_band_lo", "locate_band_hi",
               "locate_full_after_ms", "v_filter", "ab_alpha", "ab_beta")


def _config_watcher() -> None:
    last_mtime = None
    while not _shutdown.is_set():
        _shutdown.wait(1.0)
        try:
            mtime = os.path.getmtime(CONFIG_PATH)
        except OSError:
            continue
        if mtime == last_mtime:
            continue
        last_mtime = mtime
        fresh = load_config()
        changed = {k: fresh[k] for k in _HOT_FIELDS
                   if fresh[k] != config[k]}
        if changed:
            config.update(changed)
            log.info("[config] reloaded: %s",
                     ", ".join(f"{k}={v}" for k, v in changed.items()))


# --------------------------------------------------------------------------- #
# Status / runtime controls
# --------------------------------------------------------------------------- #
def print_status() -> None:
    s = stats.snapshot()
    print("\n" + "=" * 56)
    print(f"     K-Meter v{__version__}  (purple-meter TEMPO PRO STICK)")
    print("=" * 56)
    print(f"  Status    : {'ACTIVE' if active else 'PAUSED'}")
    print(f"  Key       : {config['action_key']}  (tap once — press blocked)")
    print(f"  Pro stick : down '{config['pro_stick_down_key']}'"
          f"  ->  flick up '{config['pro_stick_up_key']}'"
          f"  (gap {int(config['flick_gap_ms'])} ms,"
          f" held {int(config['flick_hold_ms'])} ms)")
    print(f"  Meter     : flick {int(config['latency_ms'])} ms before the "
          f"shape fills  (cap {int(config['release_pct'])}%)")
    print(f"  Safety    : {int(config['meter_max_ms'])} ms max tempo"
          f"   engine spin {float(config['spin_margin_ms']):.1f} ms"
          f" {'high-res' if worker and worker.hr_timer else 'standard'}")
    if meter_eye is not None:
        print(f"  Last shot : {meter_eye.last}")
    print(f"  Shots     : {s['count']}   inject err >1ms: {s['misses']}")
    if s["count"]:
        print(f"  Inject err: last {s['last']:+.2f}  avg {s['mean']:+.2f}"
              f"  sd {s['std']:.2f} ms")
    print("-" * 56)
    print("  F5 pause   F7 earlier +10ms   F6 later -10ms")
    print("             F4 earlier +3ms    F3 later -3ms")
    print("  F8 status   F10 reset stats   F9 quit")
    print("=" * 56 + "\n")


def adjust_offset(delta: int) -> None:
    config["meter_offset_ms"] = int(
        max(-300, min(800, config["meter_offset_ms"] + delta)))
    save_config(config)
    log.info("[tune] meter offset = %+d ms", config["meter_offset_ms"])


def adjust_latency(delta: int) -> None:
    """+delta = compensate more = release EARLIER."""
    config["latency_ms"] = int(
        max(0, min(300, config["latency_ms"] + delta)))
    save_config(config)
    log.info("[tune] latency comp = %d ms  (shots fire %s)",
             config["latency_ms"], "EARLIER" if delta > 0 else "LATER")


def toggle_active() -> None:
    global active
    active = not active
    print_status()


def reset_stats() -> None:
    stats.reset()
    log.info("[stats] reset")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
_ERROR_ALREADY_EXISTS = 183


def _single_instance() -> bool:
    try:
        k32 = ctypes.windll.kernel32
        k32.CreateMutexW(None, False, "KMeter_single_instance")
        return k32.GetLastError() != _ERROR_ALREADY_EXISTS
    except Exception:
        return True


def main() -> None:
    global worker, meter_eye, _flick_closer

    if not _single_instance():
        log.error("[fatal] another K-Meter is already running — close "
                  "that window first")
        return
    if not _VISION_OK:
        log.error("[fatal] vision deps missing: "
                  "pip install opencv-python mss numpy")
        return

    _flick_closer = FlickCloser()
    worker = ReleaseWorker()
    meter_eye = MeterEye(worker)
    threading.Thread(target=_config_watcher, name="cfgwatch",
                     daemon=True).start()

    keyboard.hook_key(config["action_key"], _hook, suppress=True)

    controls = {
        "f5": toggle_active,
        "f6": lambda: adjust_latency(-10),   # later
        "f7": lambda: adjust_latency(+10),   # earlier
        "f3": lambda: adjust_latency(-3),
        "f4": lambda: adjust_latency(+3),
        "f8": print_status,
        "f10": reset_stats,
        "f9": _shutdown.set,
    }
    # Bare F-keys collide with other tools' hotkeys, which silently walks
    # latency_ms around mid-session. A modifier prefix makes every binding
    # deliberate. Set "hotkey_prefix": "" to go back to bare F-keys.
    prefix = str(config.get("hotkey_prefix", "ctrl+alt+"))
    for hk, fn in controls.items():
        combo = prefix + hk
        try:
            keyboard.add_hotkey(combo, fn)
        except Exception:
            log.warning("[hotkey] could not register '%s'", combo)
    log.info("[hotkey] tuning keys bound as %sF3/F4/F6/F7 "
             "(F5 pause, F8 status, F9 quit)", prefix or "")

    print_status()
    log.info("[start] K-Meter v%s ready — tempo pro stick ('%s' down, "
             "'%s' flick up), flicking %d ms before the shape fills. "
             "Tap '%s' to shoot.",
             __version__, config["pro_stick_down_key"],
             config["pro_stick_up_key"], config["latency_ms"],
             config["action_key"])

    try:
        _shutdown.wait()
    except KeyboardInterrupt:
        _shutdown.set()

    log.info("[exit] stopping")
    try:
        keyboard.unhook_all()
    except Exception:
        pass
    # Unconditional: an up for a key that is already up is a no-op, and a
    # stick key left down outlives this process.
    _inject_all_up()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("[fatal]")
        traceback.print_exc()
        input("\nFatal error — see above / crash.log.  Press Enter.")
    finally:
        if winmm:
            try:
                winmm.timeEndPeriod(1)
            except Exception:
                pass
