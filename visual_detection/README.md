# Chasing Detection Task

A PsychoPy implementation of a chasing / pursuit detection paradigm. Participants
watch a field of moving dots and press the spacebar as soon as they detect that
one dot has begun chasing the red prey dot.

Single file: `updated_visual_attention_task.py`.

---

## Requirements

- Python 3.8+
- [PsychoPy](https://www.psychopy.org/) (tested against the standalone distribution)
- NumPy

## Running

```bash
python updated_visual_attention_task.py
```

A dialog collects four values before the window opens:

| Field | Purpose |
| --- | --- |
| Subject Number | Integer. Also seeds the RNG and names the output files. |
| Practice | Checkbox. Writes to separate `_practice_` files and offsets the seed by 100000. |
| Screen width (cm) | Monitor geometry, stored so pixels stay convertible to degrees. |
| Viewing distance (cm) | As above. |

The task runs fullscreen. **Escape aborts at any point** — buffers are flushed and
partial data is written, and the block is marked incomplete rather than lost.

Data is written to a `data/` directory created relative to the working directory
you launch from.

---

## The display

A 900 × 900 px square field with a black border on a mid-grey background.
Coordinates are in pixels, centred on the field, so positions run roughly ±450.

- **Prey** — one red dot, radius 12 px, moving 0.5 px/frame (~30 px/s at 60 Hz).
- **Predators** — 2, 4 or 6 black dots of the same size, moving 1 px/frame
  (~60 px/s). All predators are visually identical.

Every dot drifts with a random-walk heading and reflects off the walls with
jitter, so bounces are not mirror-exact.

## Task structure

**Three blocks**, of 2, 4 and 6 predators, in an order randomised per subject.
Each block runs for **120 seconds of clock time** (not a frame count).

Each block is a sequence of **trials**:

1. Every dot — prey included — is scattered onto a fresh cell of a 5 × 5 grid
   (150 px spacing, ±20 px jitter) with a new random heading.
2. Dots move randomly. On each frame with no chase running, a lottery may select
   one predator to begin chasing (≈0.7% per pool per frame). A chase only starts
   if the selected predator is more than 96 px from the prey on *both* axes, so
   pursuit never begins from close range.
3. Once chasing, that predator heads straight at the prey every frame and stops
   wiggling. Because it moves twice as fast as the prey, it will eventually catch it.
4. The trial ends on the first of:
   - **Hit** — spacebar pressed while a chase is running.
   - **False Alarm** — spacebar pressed with no chase running.
   - **Miss** — the chaser reaches the prey (within 18 px on both axes).
5. On a Hit the display freezes and the caught chaser flashes green for 1 s. The
   field then blanks to just the border for 0.5 s, and the next trial begins.

Trials keep starting until the 120 s block clock expires. A trial in progress
when the clock runs out is cut short; if a chase was active at that moment it is
logged as `Unresolved Chase` and should be treated as censored, not as a miss.

### Presses that are not scored

A press is logged as `Discarded Press` rather than scored when it lands:

- during the 200 ms blackout at the start of a trial or immediately after a
  previous press, or
- on the same frame the chase began or the blackout lifted — it cannot have been
  a response to something the participant had not yet seen.

---

## Parameters

All parameters live in the `P` dict at the top of the file, and the whole dict is
copied into the JSON sidecar on every run, so any value you change is recorded
with the data.

The ones most likely to need adjusting:

| Parameter | Default | Effect |
| --- | --- | --- |
| `block_duration_secs` | 120.0 | Length of each block. |
| `block_defs` | 2 / 4 / 6 predators | Block list. `predator_dots` must equal `numdotspergroup × numdotgroups`. |
| `flash_duration` | 1 | Seconds the display freezes and the caught chaser flashes green. |
| `trial_blank_secs` | 0.5 | Blank gap between trials. |
| `blackout_secs` | 0.200 | Window in which presses are logged but not scored. |
| `chasedistthresh_mult` | 8 | Minimum prey–chaser separation to start a chase, in dot diameters (8 × 12 px = 96 px). |
| `use_euclidean_onset` | False | `False` tests each axis separately; `True` uses Euclidean distance. |
| `catch_dist_mult` | 1.5 | Capture radius, in dot diameters. |
| `chasechxfreq1` / `chasechxfreq2` | 0.5035 | Chase-onset probability per frame per selection pool. |
| `stepsize` / `targetstepsize` | 1 / 0.5 | Predator and prey speed, px per frame. |
| `end_trial_on_hit` / `_false_alarm` / `_catch` | all True | Which outcomes end a trial and reset the display. |
| `showcue` | False | Ring the prey for `cueduration` seconds at block start. |

**Note on flash and gap timing.** With the defaults, every trial costs 1.5 s of
non-stimulus time. Since these now fire on every trial rather than occasionally,
a shorter `flash_duration` (0.3–0.5 s) leaves more of the 120 s for the task.

---

## Output

Three files per run, in `data/`.

### `Chase1_s{subj}_{main|practice}_events.csv`

One row per event, **appended** across runs. 27 columns; every row carries all of
them, blank where not applicable. Event types:

| `event_type` | Meaning |
| --- | --- |
| `Hit` | Spacebar during an active chase. |
| `Miss` | Chaser reached the prey. |
| `False Alarm` | Spacebar with no chase running. |
| `Discarded Press` | Unscoreable press; `key_name` gives the reason. |
| `Other Key` | Any non-space, non-escape key. |
| `Unresolved Chase` | Chase still running when the block clock expired (censored). |
| `Block End` / `Block Aborted` | Terminator row. Absence of `Block End` marks an aborted block. |

**Columns**

| Column | Description |
| --- | --- |
| `datetime` | Wall clock when the row was written, ISO to the millisecond. |
| `run_id` | 8-character hex id, unique to each launch of the script. |
| `subject`, `practice`, `seed` | Session identity. `practice` is 0/1. |
| `block_order` | Position of this block in the randomised order (1–3). |
| `block_label`, `predator_dots` | Which condition (`"4 predators"`, `4`). |
| `trial_num` | Trial index within the block, from 1. |
| `event_num` | Event index within the block, from 1, across all event types. |
| `event_type` | See table above. |
| `key_name` | Key identity for `Other Key`; discard reason for `Discarded Press`. |
| `rt_ms` | Hits only. Press time minus chase onset, in ms. |
| `trial_start_time` | Block-clock seconds when this trial began. |
| `chase_onset_time` | Block-clock seconds when the chase began. |
| `button_press_time` | Block-clock seconds of the press. |
| `chase_finish_time` | Block-clock seconds the chase ended (Miss, Unresolved, terminator). |
| `chaser_idx` | Dot index of the chaser (prey is always index 0). |
| `chaser_group` | Which selection pool the chaser came from (1 or 2). |
| `onset_dx`, `onset_dy` | Prey minus chaser position at chase onset, px. |
| `onset_dist` | Euclidean prey–chaser distance at chase onset, px. |
| `onset_dist_from_trial_start` | How far the chaser had travelled from its trial-start position by the time it began chasing, px. |
| `prey_x`, `prey_y` | Prey position at the moment of the event, px. |
| `chaser_x`, `chaser_y` | Chaser position at the moment of the event, px. |

Rows are written and flushed as each event happens, so a crash or a forced quit
still leaves everything up to that point on disk.

### `Chase1_s{subj}_{main|practice}_blocks.csv`

One row per block. Identity columns as above, plus:

| Column | Description |
| --- | --- |
| `block_duration_s` | Actual elapsed block time. |
| `n_frames` | Frames drawn during trials (excludes feedback and blank gaps). |
| `n_trials` | Trials started in this block. |
| `n_chases` | Chase onsets — the hit-rate denominator. |
| `n_hits`, `n_misses`, `n_false_alarms` | Scored outcomes. |
| `n_unresolved` | Chases censored by the block clock. |
| `n_discarded` | Presses that fell in a blackout or on an onset frame. |
| `n_other_keys` | Non-space key presses. |
| `time_at_risk_s` | Seconds during trials with no chase running — the false-alarm denominator. |
| `frmrate_measured` | Refresh rate measured at startup. |
| `n_dropped_frames` | Frames exceeding the refresh threshold (measured rate + 4 ms). |
| `completed` | 1 if the block ran its full duration, 0 if aborted. |

### `Chase1_s{subj}_{run_id}_params.json`

Provenance sidecar: every entry of `P`, the seed, monitor geometry and pixel
dimensions, measured refresh rate, Python / NumPy / PsychoPy versions, platform,
time on the instruction screen, and total experiment duration.
