# wan2gp-lora-browser

A WanGP / Wan2GP **extension plugin** that replaces the native LoRA selection and
multiplier controls in **Media Generator → Advanced Mode → LoRAs** with a compact
thumbnail browser and a real-time strength editor.

<!-- Add a screenshot of the panel here once you have run it against your install. -->

## What it does

- **Thumbnail browser** for the LoRAs WanGP offers for the currently selected
  model. No folder navigation, no paths — the panel simply shows what the model
  can actually use.
- **Click to include / exclude.** A newly included LoRA starts at `1.0` on every
  editable phase.
- **Edits stick.** Taps on `+`/`-`, slider drags and typed values are coalesced
  and written one exchange at a time, so a fast burst lands on the value you
  stopped at instead of springing back to an earlier one.
- **Strength controls** with three deliberately different jobs: a `0..1` slider
  on a `0.05` grid for fast coarse setting, `±0.01` buttons for exact nudging,
  and a numeric field for exact direct entry between `-10` and `10`. Negative
  multipliers work. A value the slider cannot represent is never rewritten to
  fit it — the thumb goes to the nearest place it can and the number keeps the
  truth until you actually move the slider.
- **Phase chips.** On a multi-phase model each phase is a chip showing its own
  value; tapping one chooses what the strength control and the timeline edit.
- **Step schedules you can draw.** WanGP's comma multipliers (`1,0.8,0.4,0`) are
  a first-class editing surface: regions you draw, move, resize and re-weight on
  an inline timeline, one slot per inference step, per phase. Scheduling is a
  *mode* — it replaces the plain strength control rather than sitting under it.
  See [Step schedules](#step-schedules).
- **Image previews first, video previews as a still.** A LoRA with only a video
  preview gets its *first frame* decoded server-side and cached; the video itself
  is never loaded in the browser.
- **Sort and search that understand your library.** Sort by name, Civitai name,
  recently added, active or favourites. Search matches filenames, Civitai names,
  trigger words and tags, with `kw:`, `name:`, `file:` and `tag:` prefixes to
  target one field.
- **Civitai names.** When a LoRA has a catalogue folder beside it, its Civitai
  name is shown instead of the raw filename. Toggle with the `Aa` button.
- **Inspect** (the `i` button on any tile or active row, or right-click a tile)
  opens a near-full-screen view of that LoRA's catalogue: description, trigger words,
  and every downloaded image and video with the prompt that produced it —
  all copyable.
- **Fetch from Civitai** inside Inspect builds that catalogue on demand for any
  LoRA that has none, for every model family, identifying the file by its own
  checksum.
- **Video previews play in place.** A LoRA whose only preview is a video still
  gets a first-frame thumbnail; a play badge swaps in a muted, looping,
  on-demand player. Nothing is fetched until you click it.
- **Stack profiles** — save, recall, update, rename, delete, and set a default
  per model — plus favourites, tags and a persistent thumbnail zoom. A profile
  is never locked to the model it was saved under: only the availability of its
  LoRAs decides whether it can be recalled.
- **Everything WanGP already does keeps working.** Presets, `.lset` files,
  accelerator profiles, settings recovered from generated images and videos,
  queue edits and model switches all flow into the panel automatically.

## Installation

1. Open WanGP and go to the **Plugins** tab.
2. Under *Install New Plugin*, paste this repository's URL.
3. Click **Download and Install Plugin**.
4. Enable the plugin, click **Save Settings**, and restart WanGP.

No edits to `wgp.py` are required, and the plugin adds no pip dependencies — it
uses the Pillow and OpenCV that WanGP already ships.

## How it fits into WanGP

**WanGP stays the source of truth.** The panel is a presentation and editing
layer over the native `loras_choices` and `loras_multipliers` components. Those
components keep running underneath it; the plugin only hides their visible
wrappers, and only after it has rendered real state at least once.

```
loras_choices / loras_multipliers   <- canonical WanGP state
        │  change events                     ▲  one atomic write per action
        ▼                                    │
   JSON payload  ──────────►  panel  ──────────►  typed action messages
```

If the panel fails to initialise, the native controls are left visible and
usable. That is deliberate: a frontend error must never cost you access to LoRA
configuration.

### Multiplier handling

WanGP's multiplier string is positional and dense. The plugin preserves all of
its syntax:

| Syntax | Meaning | Editor behaviour |
| --- | --- | --- |
| `0.8` | one multiplier | strength control |
| `0.4;0.9` | per guidance phase | one phase chip each |
| `1,0.5,0.25` | step/time varying schedule | editable timeline, spread over the whole run |
| `0,0,0.5,0.5` | a schedule that is off at first | two empty slots, then one region |
| `1,0.8;0.7,0.5` | a schedule inside each phase | one editable timeline per phase |
| `1;0.7,0.5,0.2` | scalar phase + scheduled phase | phase 1 is a number, phase 2 a timeline |
| `0.5:0.9` | LoRA multiplier branches | preserved verbatim, read-only |
| `0.9,0.8;1,1` on a 3-phase model | ambiguous phase expansion | preserved verbatim, read-only |
| `1 1\|0.8` | accelerator boundary | LoRAs left of the bar are marked *WanGP-managed* |

Anything the editor will not edit is **never silently flattened**. It round-trips
byte for byte, the panel says why it is read-only, and replacing it with a plain
multiplier takes an explicit confirmation.

The last read-only case is deliberate. A token that declares fewer `;` parts than
the model's phase capacity is expanded by WanGP's own parser using
`model_switch_phase`, and this plugin does not replicate that logic — so rather
than re-serialising such a token from a guess about which phase each list lands
in, it is kept exactly as imported.

Strengths are stored against a stable LoRA identity (the relative path WanGP
itself uses), never against a row index — so removing the middle LoRA of a stack
cannot shift another LoRA's multiplier onto the wrong file.

### Step schedules

A comma in a WanGP multiplier makes it vary over the run. The panel edits those
as **regions on a timeline**, and scheduling a phase is a mode:

```
[i] Detail Enhancer                                   [Schedule ▴]  [×]
[Phase 1 ∿2]  [Phase 2 0.5]

Step schedule   Phase 1   30 global steps    [+ Region] [Slots: 30 ▾] [Clear phase] [Close]

  empty slots = 0
            ┌──────────┐                    ┌────┐
  ──────────│   0.95   │────────────────────│ 0.2│────────
            └──────────┘                    └────┘
   1    4    7    10   13   16   19   22   25   28   30

  Selected region  [Steps 13–19]  [Delete]
  [−]  ------------ slider ------------  [0.2]  [+]
```

**Scheduling needs One Phase guidance.** WanGP stretches each phase's comma list
to fill that phase's interval, so a list runs one value per step only when its
length matches the number of steps in the interval it covers. In One Phase the
switch points sit at the end of the run, so phase 1 covers every step and a
30-value schedule at 30 steps runs exactly as drawn. With two or more phases,
phase 1 covers only up to `model_switch_step` — derived at generation time from
the sampler's timesteps and the switch threshold, and unknowable while editing —
so a four-slot schedule drawn against a four-step run would be squeezed into
however many steps phase 1 turns out to be. Rather than draw something WanGP is
not going to do, the panel does not offer scheduling there at all.

Switching guidance to two or more phases therefore resets a scheduled LoRA to
**0**: the schedule said the multiplier varies over the run, and no single number
carries that over, so the LoRA switches off and waits for you to set what you
want. LoRAs on a plain strength are untouched — that means the same thing in any
phase mode. Coming back to One Phase keeps each phase's own value.

**Scheduling replaces the plain strength control.** While a phase is scheduled
the row has no slider, `+`/`-` or numeric field of its own — the region's
controls are the only ones on screen, open or collapsed. That is not cosmetic:
a scheduled phase is its regions, so a plain strength would be a number WanGP
never applies.

**Slots no region covers are zero.** There is no base strength hiding behind the
timeline. Boxes over steps 1–2 and 4–6 mean step 3 is off, and stays off until
something covers it. A phase you have not drawn on yet is simply not scheduled —
it is worth its plain strength, and opening the scheduler writes nothing.

- **Draw a region by dragging across empty timeline space**, or tap once for one
  of the default width. A new region starts at the strength the row already had.
  Drawing stops at the neighbouring region rather than overrunning it — dragging
  is for that.
- `+ Region` does the same without aiming: the free tail, else the first hole
  that fits, else the largest. Either way a full timeline is refused rather than
  overlapped.
- Drag a region's body to move it, its edges to resize it. Bounds snap to whole
  slots, and on drop the dragged region wins: a partial overlap shrinks its
  neighbour, dropping inside one splits it, covering one deletes it. The slots a
  region leaves behind go back to zero.
- **The whole run is always in view.** The timeline takes the width it is given
  and the slots divide it, however many there are — it never scrolls sideways.
- **Full screen** opens the same editor as a dialog the width of the window,
  with more height per region. It is live, not a preview: everything drawn or
  dragged there goes straight to WanGP, and the row behind it stays in step.
  Close it with the button, the backdrop, or Escape.
- **Close** leaves the scheduler and keeps the schedule; the row then shows what
  it is doing (`∿ 0.95 at steps 4–10, 0.2 at steps 13–19`) and tapping that goes
  back in. **Clear phase** is the one that removes it, returning that phase — and
  only that phase — to a plain multiplier and its ordinary strength control.
- Schedule state is held **per phase** internally, so a model that carries a
  phase 2 value in One Phase mode keeps it untouched while phase 1 is scheduled.
  Linking is a multi-phase affair and so never coexists with a schedule.

**One slot per inference step, always.** Timelines follow the step counter live:
change it in WanGP and every schedule in the panel comes with it, without being
asked.

How a schedule follows depends on where it came from, because a slot means two
different things:

- A schedule **drawn in the panel** is step-aligned — slot *i* is step *i* — so
  it is truncated or extended at the end. Going 4 → 5 steps keeps steps 1–4
  exactly as they were and leaves step 5 undefined (so, 0, until you draw on it);
  going 5 → 3 drops steps 4–5 and keeps the rest. A region straddling the new end
  is clipped to it; one entirely beyond it is gone, and does not come back if you
  lengthen the run again.
- A schedule **that arrived from a preset, an `.lset` file or a hand edit** was
  authored at its own resolution, and WanGP spreads it across the whole run.
  Truncating that would change what it renders, so it is resampled into step
  alignment once — and is step-aligned from then on.

Read-only multipliers are never refitted: branch syntax, an ambiguous phase
count, or a list too long to drag all stay exactly as imported. `Slots:` is still
there for choosing a resolution deliberately.

This is the one place the panel rewrites a schedule you did not touch. Binding
the timeline to the step counter is what makes the two agree, and the trade is
that changing steps writes `loras_multipliers` for every scheduled LoRA.

**Slot numbers say what they are.** A comma-only token spans the whole run, so
when the timeline has one slot per step those slots really are steps and are
labelled that way. A phase-specific schedule is expanded inside a phase interval
that depends on runtime model switching, which the plugin cannot see — those
timelines say *phase-relative* rather than inventing global step numbers, even
when they have one slot per step.

**Opening a timeline writes nothing.** A schedule that works out to one value
everywhere is emitted as that single value, so opening the panel — or adding a
region and not yet changing its strength — leaves the native multiplier exactly
as it was. An imported schedule is likewise rendered, never re-serialised, until
you make a material edit.

`Slots:` re-grids a schedule to a different resolution by resampling it — what
you want when a schedule should keep its shape at a different granularity, as
opposed to the step counter moving, which keeps the steps that still exist.

### Phases

Phase behaviour is resolved from live WanGP data (`guidance_phases`,
`lora_multiplier_phases`, `guidance_max_phases`, `lock_guidance_phases`), not
from a hardcoded model list. For MiniMax H3 that yields:

| Mode | Controls |
| --- | --- |
| One Phase | one strength control |
| Two Phases | Phase 1 + Phase 2 |
| Two Phases with Tiling | Phase 1 + Phase 2 (tiling does not add a third phase) |

Because H3 declares two multiplier phases regardless of guidance mode, a phase 2
you tuned survives a trip through One Phase inside the native token itself. For
models whose capacity follows the guidance mode, the plugin remembers the hidden
value and restores it when the mode comes back.

WanGP manages H3's phase-two Turbo LoRA itself; the plugin does not compete with
that. LoRAs WanGP places on the accelerator side of the multiplier string are
marked as managed and are excluded from *Disable all* and from stack profiles.

### Stack profiles across models

A profile stores only the user side of the LoRA stack — ids and multiplier
tokens — and the model it happened to be saved under is recorded as provenance,
not as a restriction. Model families that share one LoRA folder (every LTX 2
variant, for instance) see each other's LoRAs, so recalling a profile saved on
one variant while another is loaded is expected to work.

The only rule is availability. A profile whose LoRAs are all present is applied
in full; one the current model can supply only in part is applied as far as it
goes, with a warning naming what was skipped, and is flagged with a ⚠ in the
dropdown; a profile with nothing available is refused. Profiles the current
model can apply in full are listed first.

### Security notes

- The LoRA directory is **never** exposed via `gr.set_static_paths()`. Tiles
  receive a small derived WebP as a `data:` URI.
- The frontend addresses LoRAs by stable ID only; paths are resolved server-side
  and validated against the current model's LoRA root.
- Absolute paths never reach the browser, and all display text is inserted as
  text nodes rather than HTML.

## The Civitai catalogue

Civitai names, trigger words and the Inspect view all come from a sidecar folder
named after the LoRA:

```
<lora dir>/
  cool_lora.safetensors
  cool_lora.png | .mp4        host preview
  cool_lora/                  sidecar, named after the file stem
    summary.txt               name, creator, trigger words  (the search index)
    cool_lora.json            combined Civitai record       (the Inspect view)
    modelVersion.json         raw version record
    model.json                raw model record
    cool_lora.txt             trigger words, one per line
    media/001.jpg 001.json    downloaded media + prompt metadata
```

Indexing reads only `summary.txt` — a few hundred bytes — so a large library
stays fast; the full record is parsed only when you open Inspect. Reading is
forgiving about which of those files exist: a folder holding only the combined
JSON, or only loose images, still fills the panel rather than going blank.

### Fetching a catalogue

Open Inspect on any LoRA and press **Fetch info from Civitai**. The plugin
hashes the `.safetensors`, looks that exact file up on Civitai, and writes the
folder above: records, trigger words, summary, every preview image and video,
and the prompt metadata Civitai holds for each one. If there is no host preview
yet, the first image and video are copied out beside the LoRA so the tile gets a
thumbnail.

Identification is by checksum alone, so this works for every family WanGP can
load — MiniMax H3, the LTX 2 line, Wan — with nothing to configure per model.
Re-fetching is additive: media already on disk is never downloaded again, while
the JSON and summary documents are rebuilt, which is how an older or hand-made
sidecar is brought into the shape the panel reads.

**Fetch all missing info** — the last item in the *Manage profiles* (`⋯`) menu —
does the same for every LoRA the current model offers whose catalogue is not
complete. The run happens in the background with progress on the status line,
and the same menu item becomes *Stop fetching* while it is going.

Completeness is judged against the Civitai record already on disk, which lists
the media entries a LoRA is supposed to have. A folder is incomplete when it is
absent, or is missing its `summary.txt`, its records, its trigger words, any
media file, or any of the `media/NNN.json` prompt records — so a catalogue built
with the images but no prompts is repaired rather than passing as finished. The
Inspect view names what a given LoRA still lacks. Anything genuinely complete is
skipped, so an already-enriched library is not re-hashed.

LoRAs Civitai has never heard of have no folder at all, so they are reported as
not found and looked up again on the next run; nothing on disk separates them
from a LoRA whose catalogue has yet to be built.

Two things are worth knowing. Only prompts the uploader actually published come
down — many video LoRAs have none, and those media show without a caption. And
media is only ever fetched from Civitai's own hosts, with the filename chosen
here rather than taken from the URL.

### Civitai API key

Public models need no key. For anything that does, either export
`CIVITAI_API_KEY` before starting WanGP (preferred — it is never written to
disk) or set one through *Manage profiles* (`⋯`) → *Set Civitai API key*, which
stores it in the plugin's settings file **in plain text**. The environment
variable always wins.

## Preferences

Favourites, tags, zoom, stack profiles and any Civitai API key you set are
stored in
`wan2gp_lora_browser.json` next to WanGP's own config file (falling back to a
`data/` folder inside the plugin if that location cannot be discovered). Writes
are atomic, and a corrupt file is backed up rather than crashing WanGP.

## Development

```bash
pip install pytest Pillow
python -m pytest
```

The test suite covers the parts that can be reasoned about without a running
WanGP: multiplier parsing and serialisation, schedule compilation, region
placement and collision resolution, state reconciliation, phase switching,
preview matching and caching, profile and metadata persistence, and the JSON
bridge in `plugin.py` (against stubbed WanGP/Gradio modules).

The panel itself is also tested in a real browser, against that same Python
bridge rather than a mock of it — the grid's three-row cap, the weight controls,
and region dragging are all things only layout and pointer events can answer.
Those tests skip unless Playwright and a Chromium build are present:

```bash
pip install playwright && playwright install chromium
python -m pytest tests/test_panel_browser.py
```

```
plugin.py                  WanGP lifecycle, component wiring, JSON bridge
lora_browser/
  multiplier_codec.py      parse/serialise loras_multipliers
  schedule.py              region model: compile, reconstruct, place, collide
  ui_payloads.py           state model, phase resolution, action application
  inventory.py             native LoRA list -> displayable entries
  thumbnails.py            preview matching, first-frame decode, cache
  catalogue.py             read the Civitai sidecar folder
  civitai.py               fetch and build that folder on demand
  metadata_store.py        favourites, tags, zoom, sort (atomic JSON)
  profile_store.py         stack profiles
assets/                    panel CSS and JavaScript
```

### Decisions worth knowing

Four things the design spec leaves to implementation, settled here:

- **Numeric precision.** Direct entry keeps four decimals — the same precision
  the serialiser emits — rather than the two the editor used to round to. An
  untouched imported token is not rewritten at all, whatever its precision.
- **Linked phases.** Linking moves the plain strength of every visible phase
  that has one, skipping any phase that is scheduled. It never copies regions
  between phases; schedules stay independent.
- **What fills the gaps.** Nothing: an uncovered slot is 0, not a base. The
  strength a phase had before scheduling survives only as what a new region
  starts at and what *Clear phase* restores.
- **Irregular imported schedules.** Resampled into step alignment the first time
  they are seen, which preserves what they render; after that they follow the
  step counter like any other. A list too long to drag usefully is preserved and
  shown read-only instead, with an explicit re-grid offered.
- **Multi-phase coordinates.** Phase-relative by default. Exact global step
  numbers are claimed only for a schedule that really does span the whole run at
  one slot per step.

## Status and compatibility

Built against **WanGP 12.642** — the plugin API, `refresh_lora_list`,
`get_lora_dir`, the `loras_choices` / `loras_multipliers` components and the
multiplier grammar in `shared/utils/loras_mutipliers.py` were all verified
against that source. Every WanGP global is read through a guarded lookup, so a
build that renames or drops one degrades gracefully rather than breaking the
tab.

The automated suite runs without WanGP, and the browser tests drive the real
panel against the real Python bridge. The in-app acceptance pass — installing
from GitHub into a live WanGP instance and walking the checklist in the design
spec against actual generations — still needs to be done on a machine with WanGP
running.

## Licence

MIT — see [LICENSE](LICENSE).
