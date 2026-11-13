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
  A scheduled phase has no single value, so its chip wears a `∿ Scheduled`
  badge instead of a number — `∿` is the panel's mark for "the strength changes
  from step to step".
- **Every active row starts with its LoRA's picture** — the tile's preview, a
  video preview's first frame with a play button, or a placeholder — in one
  column the same width on every row, so the rows stay aligned. Pressing the
  picture opens Inspect.
- **Step schedules you can draw.** WanGP's comma multipliers (`1,0.8,0.4,0`) are
  a first-class editing surface: regions you draw, move, resize and re-weight on
  an inline timeline, one slot per inference step, per phase — and on LTX-2 and
  MiniMax H3, per pipeline stage. A scheduled row keeps the same strength
  control as any other, pointed at the selected region. See
  [Step schedules](#step-schedules).
- **Switch a LoRA off without losing it.** Every row has an on/off switch: off
  keeps the strength, the schedule and the LoRA's place in the list, and WanGP
  neither loads nor applies it. See [Switching a LoRA off](#switching-a-lora-off).
- **Paste a whole stack as text.** *Paste stack* opens a box: paste one
  `name: x multiplier` line per LoRA, press Enter, and that is the stack —
  schedules and all. *Copy current stack* writes the same format back out. See
  [Pasting a stack as text](#pasting-a-stack-as-text).
- **A list view for long stacks.** One line per LoRA — picture, name, what each
  phase applies and its switch — with a scheduled phase shown as how many steps
  it runs on. Click a line to open that LoRA's full editor under it; one is
  open at a time. *Cards* shows every editor at once, as before. See
  [The list view](#the-list-view).
- **Visual only / Audio only on LTX-2.** Apply just the picture side or just
  the sound side of a LoRA, one LoRA at a time or every LoRA in one go. The
  panel builds an exact copy holding only that side, in the background, the
  moment you choose it. See [Visual only and audio only](#visual-only-and-audio-only-ltx-2).
- **Separate picture and sound strengths on LTX-2.** Give a LoRA, or a whole
  group, one strength and schedule for the picture and another for the sound.
  It stays one row with two editors; the panel quietly runs it as two files.
  See [Separate picture and sound strengths](#separate-picture-and-sound-strengths-ltx-2).
- **Groups and bakes.** LoRAs that share one strength and schedule can be
  grouped, so they move as one, and a group can be baked into a single LoRA at
  a rank you choose to save memory — and unbaked again. See
  [Groups and bakes](#groups-and-bakes).
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
uses the Pillow, OpenCV and NumPy that WanGP already ships.

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
| `#lora_off 1 0.8 x.safetensors` | a LoRA switched off in the panel | listed, dimmed, not applied |
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
as **regions on a timeline**. A row looks the same scheduled or not: the
timeline and its tools sit above the one strength control every row has.

```
[●] Detail Enhancer                       [∿ Schedule ▴]  [i]  [×]
    [Phase 1 (∿ Scheduled)]  [Phase 2 0.5]

    [⤢ Full screen]  [Clear schedule]                       [Delete]
    empty steps = 0
              ┌──────────┐                    ┌────┐
    ──────────│   0.95   │────────────────────│ 0.2│────────
              └──────────┘                    └────┘
     1    4    7    10   13   16   19   22   25   28   30
    [−]  ------------ slider ------------  [0.2]  [+]
```

The strength control under the timeline is the plain row's own control, pointed
at the selected region — the highlighted one; tap another to switch. **Delete**
removes that region and is only there while one is selected. Before anything is
drawn, the control edits what the phase is worth, which is also the strength a
new region starts at.

**`∿` means scheduled.** A phase whose strength changes from step to step shows
a solid `∿ Scheduled` badge on its chip where a plain phase shows its number;
the wave on the row's *Schedule* button fills in the same way while any phase
of that LoRA is scheduled, and so does the line a closed timeline leaves
behind. The badge is drawn in the theme's text colour, so it stands out in
any theme — including WanGP's dark one, whose accent is the same white as its
text. The chip's own outline still marks the phase being edited, so the two
never share a cue.

**Where a timeline can be exact.** WanGP stretches each phase's comma list to
fill the steps that phase runs, so a list runs one value per step only when its
length matches them. That decides where the panel schedules:

- **One Phase, on every model.** Phase 1 covers the whole run, so a 30-value
  schedule at 30 steps runs exactly as drawn.
- **LTX-2 and MiniMax H3, in either mode.** Their phases are not cut out of one
  run at a switch step: each is a pipeline stage whose length is known while you
  edit, so each phase gets a timeline exactly as long as its stage (below).
- **Elsewhere, with two or more phases** — Wan 2.2, say — phase 1 covers only up
  to `model_switch_step`, derived at generation time from the sampler's
  timesteps and the switch threshold and unknowable while editing, so a timeline
  would show something WanGP is not going to do. The panel offers none there,
  and **does not touch a schedule either**: the row says what the schedule does,
  read-only, and One Phase edits it again exactly as it was.

**Switching phase mode never changes phase 1.** A scheduled phase 1 is the same
schedule, and a plain phase 1 the same strength, in One Phase and in Two.

**Staged models.** On LTX-2, 2.3 and 2.5, phase 1 is the first denoising pass
and phase 2 the refinement after the upscale, each running its own fixed sigma
table. On MiniMax H3, phase 1 is the full denoise — the step counter, exactly
as in One Phase — and phase 2 the enhancement pass after the latent upscale,
run for WanGP's *Phase 2 Steps*:

| Model | Phase 1 (stage 1) | Phase 2 (stage 2) |
| --- | --- | --- |
| LTX-2 Distilled (2.3, 2.5, MSR, JoyAI Echo) | 8 steps, whatever the counter says | 3 steps |
| LTX-2 Dev, Euler or res2s sampler | the step counter | 3 steps |
| LTX-2 Dev, a *Distilled 8 Steps* sampler | 8 steps | 3 steps |
| MiniMax H3 | the step counter | *Phase 2 Steps* (default 3) |

One Phase runs stage 1 only. The step counter moves stage 1, and on H3 *Phase 2
Steps* moves stage 2 the same way. H3 keeps two multiplier phases in every
guidance mode, so its stage 2 stays in the token through One Phase and is there
again, as it was, in Two Phases.

The two stages are never one shared schedule. WanGP would play a list with no
`;` in both — one value per step in stage 1, then squeezed into stage 2's few —
and a stage-1 schedule squeezed into the refinement is not something anyone
drew. So when such a list enters Two Phases, it stays stage 1's, and **stage 2
carries on from the end of stage 1**, matched by noise level rather than
stretched. On a distilled LTX-2 stage 1 — the distilled checkpoints, or Dev on
a *Distilled 8 Steps* sampler — the refinement re-runs the noise levels of
stage 1's last three steps, so stage 2 gets steps 6–8:

| One Phase (8 steps) | Two Phases |
| --- | --- |
| `1,1,1,1,0,0,0,0` (early only) | `1,1,1,1,0,0,0,0;0` — nothing to carry, stage 2 empty |
| `0,0,0,0,0,0,1,1` (the last steps) | `0,0,0,0,0,0,1,1;0,1,1` |
| `0,0,0,0,1,1,1,1` (late detail) | `0,0,0,0,1,1,1,1;1,1,1` |
| `1,1,1,1,0.8,0.6,0.4,0.2` (fade out) | `1,1,1,1,0.8,0.6,0.4,0.2;0.6,0.4,0.2` |

Where stage 1's steps do not line up with stage 2's — LTX-2 Dev with Euler or
res2s, and MiniMax H3 — stage 2 holds the strength stage 1 ends at. The same
goes for a list with no `;` that arrives any other way in Two Phases, such as a
preset. What stage 2 gets is ordinary regions with no link to stage 1: edit
either stage and the other stays as it is. *Clear schedule* empties it; a
region drawn on an empty stage 2 starts at full strength.

**Fill from stage 1.** The same copy, on request: an empty Phase 2 timeline —
cleared, or a plain strength — offers it, and its tooltip shows what it will
write. *Clear schedule* afterwards puts back whatever stage 2 was before.

**One strength control, never two.** While a phase has regions, the row's
strength control edits the selected region; there is no plain strength left to
show. That is not cosmetic: a scheduled phase is its regions, so a plain
strength would be a number WanGP never applies.

**Steps no region covers are zero.** There is no base strength hiding behind
the timeline. Boxes over steps 1–2 and 4–6 mean step 3 is off, and stays off
until something covers it. A phase you have not drawn on yet is simply not
scheduled — it is worth its plain strength, and opening the scheduler writes
nothing.

- **Draw a region by dragging across empty timeline space**, or tap once for one
  of the default width; an empty timeline says so. A new region starts at the
  strength the row already had. Drawing stops at the neighbouring region rather
  than overrunning it — dragging is for that — and a full timeline is refused
  rather than overlapped.
- Drag a region's body to move it, its edges to resize it. Bounds snap to whole
  steps, and on drop the dragged region wins: a partial overlap shrinks its
  neighbour, dropping inside one splits it, covering one deletes it. The steps a
  region leaves behind go back to zero.
- **The whole run is always in view.** The timeline takes the width it is given
  and the steps divide it, however many there are — it never scrolls sideways.
- **Full screen** opens the same editor as a dialog the width of the window,
  with more height per region. It is live, not a preview: everything drawn or
  dragged there goes straight to WanGP, and the row behind it stays in step.
  Close it with `×`, the backdrop, or Escape.
- **Schedule ▴** hides the timeline and keeps the schedule; the row then shows
  what it is doing (`∿ Scheduled  0.95 at steps 4–10, 0.2 at steps 13–19`) and
  tapping that goes back in. **Collapse all**, above the rows, hides every open
  timeline at once, the same way. **Clear schedule** is the one that removes
  it, returning that phase — and only that phase — to a plain multiplier and its
  ordinary strength control.
- Schedule state is held **per phase** internally, so a model that carries a
  phase 2 value in One Phase mode keeps it untouched while phase 1 is scheduled.
  Linking acts on plain strengths only, so it is not offered while a timeline is
  open.

**One slot per inference step, always.** There is no resolution to choose:
timelines follow the step counter live — change it in WanGP and every schedule
in the panel comes with it, without being asked.

How a schedule follows depends on where it came from, because a slot means two
different things:

- A schedule **drawn in the panel** is step-aligned — slot *i* is step *i* — so
  the step count only moves where the run ends; nothing on the timeline moves or
  stretches. Going 4 → 5 steps keeps steps 1–4 exactly as they were and adds an
  empty step 5 (so, 0, until you draw on it). Going 5 → 3 stops the run after
  step 3: WanGP is given three values, and a region crossing the new end shows
  only up to it.
- **A lower step count hides; it does not delete.** What a shorter run no longer
  reaches is kept — the timeline says `+2 steps kept` — and raising the count
  brings it back exactly as it was. So 5 → 3 → 5 is the schedule you started
  with, and typing `12` over `4`, which passes through `1`, loses nothing.
  Editing at the lower count still works: a region crossing the end is one
  region, so a new strength or a delete reaches the part past the end too, while
  moving it or pulling its right edge in lets that part go. The memory lasts
  while WanGP runs; a restart or a model switch starts again from the
  multipliers as they stand.
- A schedule **that arrived from a preset, an `.lset` file or a hand edit** was
  authored at its own resolution, and WanGP spreads it across the whole run.
  Truncating that would change what it renders, so it is resampled into step
  alignment once — and is step-aligned from then on.

Read-only multipliers are never refitted: branch syntax, an ambiguous phase
count, or a list too long to drag all stay exactly as imported. A list too long
to drag is offered one way back to editable — **Fit to N steps**, which
resamples it to one value per step and rewrites the multiplier.

This is the one place the panel rewrites a schedule you did not touch. Binding
the timeline to the step counter is what makes the two agree, and the trade is
that changing steps writes `loras_multipliers` for every scheduled LoRA.

**Slot numbers say what they are.** A comma-only token spans the whole run, and
a stage of known length runs one value per step, so those regions are labelled
in steps (`Steps 4–10`). A phase-specific schedule on a model whose phases are
cut from one run is expanded inside an interval that depends on runtime model
switching, which the plugin cannot see — its regions say *slots* rather than
inventing step numbers, even when there is one slot per step.

**Opening a timeline writes nothing.** A schedule that works out to one value
everywhere is emitted as that single value, so opening the panel — or adding a
region and not yet changing its strength — leaves the native multiplier exactly
as it was. An imported schedule is likewise rendered, never re-serialised, until
you make a material edit.

### Switching a LoRA off

The switch at the start of every row turns a LoRA off without losing anything:
its strength, its schedule — steps past the end of the run included — and its
place in the list all stay. WanGP neither loads nor applies it; switch it back
on and it is exactly what it was. Its tile in the grid keeps a hollow check, and
clicking the tile switches it on again. Removing it (`×`) is what forgets it.

Off is stored where WanGP keeps the rest of the stack, so it travels wherever
that does — presets, `.lset` files, the queue, settings saved with generated
media. The LoRA leaves `loras_choices`, and its multiplier moves to a comment
line in `loras_multipliers`, which WanGP's parser drops before it reads a single
token:

```
0,-2,0,0;1 0.7;0.5
#lora_off 2 0.6,0.6,0,0;1 detail_tweaker_h3.safetensors
```

That is `#lora_off <position> <multiplier> <lora>`; the position keeps the row
where it was. Selecting the same LoRA again any other way — a preset, WanGP's
own picker — turns it back on.

**Disable all** switches every LoRA off this way and **Enable all** brings them
all back. WanGP-managed accelerator LoRAs have no switch and are left out of
both. Stack profiles remember which LoRAs were off.

### Pasting a stack as text

**Paste stack**, above the active rows, opens a box with the cursor already in
it. Paste a stack — one LoRA per line — and press **Enter**:

```
Vocal_Enhance_1: x0,0,0,0,0.325,0.325,0,0;0
Vocal_Percussion_1: x0.8;0.375,0.25,0.25
Body_Structure_1: x1;0.25
Camera_Movement_1: x1,1,0.875,0.875,0.875,0.875,0.25,0.25;0
```

Each line is `name: x<multiplier>`, and the multiplier is a WanGP token exactly
as `loras_multipliers` takes it: `;` between phases, `,` between steps, `:`
between branches. So the lines above, on LTX-2 in Two Phases, are four LoRAs:
the first scheduled over stage 1's eight steps and off in stage 2, the second
at 0.8 in stage 1 and scheduled over the refinement's three steps. They arrive
exactly as written and open on the timeline like any other schedule.

- The `x` is optional, and so are spaces around the numbers. A name on its own
  loads at 1. Blank lines, `#` and `//` comments, list bullets and code fences
  are skipped. **Shift+Enter** starts a new line; **Escape** closes the box and
  keeps what you typed for next time.
- **The pasted stack replaces the current one** — switched-off LoRAs included —
  the way recalling a profile does. WanGP-managed accelerator LoRAs stay.
- **Names are matched against what WanGP offers the current model**, most exact
  first: the LoRA's path (`styles/detail.safetensors`, with or without the
  extension), its filename with or without the extension, its Civitai name, and
  last any of those with case, spaces and punctuation ignored — so
  `vocal enhance 1` finds `Vocal_Enhance_1.safetensors`. A name two LoRAs answer
  to is refused rather than guessed at; give its folder to settle it.
- **A line that cannot load is listed, not guessed at:** an unknown or
  ambiguous name, a multiplier WanGP could not read, a LoRA already on an
  earlier line, or one WanGP manages. Everything else loads, and the box stays
  open with the text and the reasons, so you can fix a line and press Enter
  again. When nothing can load, nothing changes.
- **The phase mode is applied at once.** A line naming more phases than the
  mode runs is cut the way a preset's would be, and the box says so — in One
  Phase on LTX-2, switch to Two Phases and paste again to keep stage 2 as
  written. A stage-1 list with no `;` in Two Phases gets a stage 2 that carries
  on from stage 1, as described in [Step schedules](#step-schedules).

**Copy current stack**, in the same box, copies the LoRAs that are on in this
format, each named by the shortest name that reads back to it — so a stack
copied here and pasted later, or on another machine with the same files, comes
back as it was.

### The list view

The active LoRAs open as a list: one line per LoRA with its switch, its
picture, its name and what each phase applies. A phase at one strength shows
that strength; a scheduled phase shows how many steps it runs on —
`∿ 6 steps` — never each step's strength (hover it to see which steps).

Nothing in the list edits a value. Click a line and that LoRA's editor opens
under it — the same editor *Cards* shows, phase chips, strength, schedule and
all, plus the controls described below — and opening another line closes it.
**Done** or **Collapse all** closes it again. **Cards** shows every LoRA's editor
at once instead; the panel remembers which view you last used.

**Select** turns the lines into checkboxes, for changing several LoRAs at once:
make them visual only, audio only or normal (on LTX-2), or group them. On a
narrow panel each line puts its phases under its name, each with its own
`P1` / `P2` label.

### Visual only and audio only (LTX-2)

An LTX-2 LoRA can change the picture, the sound, or both. Every LoRA's editor
on an LTX-2 model has **Applies to: Normal · Visual only · Audio only**.

- **Visual only** applies only what the LoRA changes in the picture, **Audio
  only** only what it changes in the sound, and **Normal** all of it. The editor
  shows how the LoRA splits between the two, and what WanGP loads for it.
- The panel builds a copy of the LoRA holding only that side's tensors. They
  are byte-for-byte the original's — nothing is re-computed or re-quantised —
  so the side you keep is exactly what it was. Attention from one side into the
  other belongs to the side it writes into; the few layers both sides read,
  such as the text-embedding projection, are kept in either. A LoRA with
  nothing on the other side is linked rather than copied.
- **The copy is built the moment you choose the mode**, in the background,
  and used from then on. Choosing that mode for that LoRA later uses the same
  copy at once; if the original file changes, the copy is built again.
- While anything is building, a bar above the rows shows progress and warns
  that **if you generate before it finishes you may get an unexpected result**:
  WanGP reports a file that is not built yet as missing and does not start.
  The panel warns rather than holding back WanGP's Generate button.
- **Audio / visual ▾** above the rows changes every LoRA at once — *Make every
  LoRA visual only*, *audio only* or *normal* — or *Select which LoRAs…* to pick
  them. A LoRA that only changes the picture has nothing to apply in audio
  only, so it is switched off, and normal (or visual only) switches it back on;
  the same goes the other way round. A LoRA you switched off yourself stays off.
- A mode lives in WanGP's own stack, the way *off* does, so presets, the queue
  and settings saved with generated media keep it: the LoRA's entry in
  `loras_choices` names its copy (`_lora_browser/<name>.visual.safetensors`),
  and a LoRA switched off by its mode is a `#lora_skip` line in
  `loras_multipliers`. Stack profiles and stack text carry modes and groups
  too: a copied stack reads `Rain_Ambience [audio, group A]: x0.5`, and pasting
  it back restores both.

Visual only and audio only are offered on LTX-2 models only. MiniMax H3 and
other models keep the list view, groups and bakes.

### Separate picture and sound strengths (LTX-2)

**Applies to** has a fourth choice, **Separate**: the LoRA's picture and its
sound each get a strength and a schedule of their own. It is still one row and
one editor, now with a **Picture** section and a **Sound** section — each with
its own phase chips, slider and Schedule button, working exactly as a LoRA's
do. In the list, each phase shows the picture's value above the sound's.

- Both start where the LoRA's strength was, so choosing Separate changes
  nothing until you move one. Going back, **Normal** and **Visual only** keep
  the picture's settings and **Audio only** the sound's.
- Underneath, WanGP is given the LoRA as two files, each with its own
  multiplier: its visual-only copy and its *sound half*
  (`_lora_browser/<name>.sound.safetensors`), which holds the sound side alone.
  The few layers both sides read stay in the visual copy, so they are applied
  once, at the picture's strength — never twice. Both are exact copies of the
  original's tensors, built the moment you choose Separate, with the same
  warning while they build.
- The switch, `×`, its tile and Inspect treat the two as the LoRA they are.
  Stack profiles keep both strengths, and stack text writes the LoRA as two
  lines — `Rain_Ambience [picture]: x1;1` and `Rain_Ambience [sound]: x0.4;0.2`
  — which paste back the same way.
- A LoRA that only changes one of picture and sound has no second strength to
  give, so Separate is greyed out for it.

A **group** gets the same with its **Strengths** switch: *One for both* or
*Picture and sound*. With two, the group's editor has a Picture section and a
Sound section; each member still has its own Normal / Visual / Audio choice
and takes the group's strength for each side it brings. Baked, each side with
two or more LoRAs becomes a file of its own (the sound one's name ends
`.sound.safetensors`), both at the rank you choose. Switching back to one
strength keeps the picture's.

### Groups and bakes

LoRAs that share one strength and schedule — or should — can be **grouped**,
so they move as one: **Group with…** in a LoRA's editor, or **Select** a few
and press **Group**. If their settings differ, the panel asks whose the group
keeps. A group is one line with its members listed under it; its editor has
one set of phase chips, strength and schedule for all of them, each member's
own Normal / Visual / Audio choice, and `×` to take a member out. **Ungroup**
makes them separate LoRAs again, each keeping the group's settings.

Grouping alone builds nothing and changes nothing WanGP loads. WanGP already
applies every active LoRA as one sum per layer, so two LoRAs side by side cost
no more memory than one file holding both exactly would.

**Bake** is what saves memory: it re-fits the group's combined change as one
LoRA at the rank you choose. It starts at the rank of the highest-rank LoRA in
the group, and the dialog lists each size with what it saves. A rank at or
above the group's combined rank is exact; a smaller one keeps the strongest
directions of the combined change, layer by layer (a truncated SVD), and once
it is built the group says how much of the change it kept. The originals are
never touched: **Unbake** goes back to them and **Change size…** bakes again.
Strength and schedule still apply live to the baked file, and changing a
member's mode or the group's members bakes it again to match. LoRAs that are not
plain low-rank pairs (DoRA, LoKr, full weight differences) cannot be baked, and
the dialog says so.

A group is stored as `#lora_group <letter> <lora>` lines in `loras_multipliers`;
a baked group's lines also name its file
(`_lora_browser/_bakes/<hash>.r<rank>.safetensors`), which is what
`loras_choices` then holds. A group with separate picture and sound strengths
lists each member as the halves it is selected as; nothing else marks it.

### Built files

Visual-only and audio-only copies and baked groups are kept in the plugin's own
folder, under `derived/<LoRA folder>-<hash>/`, and linked into the model's LoRA
folder as `_lora_browser/` — a symlink, a junction on Windows, or a plain
folder where neither can be made — which is how WanGP finds them like any other
LoRA. They never show in the thumbnail grid. **Built files** in the status bar
lists them with what each was built from, its size and what uses it (this stack
or a saved profile), and **Delete unused** removes the ones nothing uses.
Anything deleted is built again from its originals if it is ever used again.

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
you tuned survives a trip through One Phase inside the native token itself. And
because its phase 1 is the full step count in either mode, a switch never
changes phase 1's strength or schedule (see [Step schedules](#step-schedules)).

On models whose phase count follows the guidance mode — LTX-2 among them —
WanGP refuses a multiplier that names more phases than the mode runs (`1;1` in
One Phase is an error, not a strength). So the moment the mode narrows, the
panel rewrites every token to what WanGP accepts, without waiting for an edit,
and keeps what it cut for the session: switch back and a phase 2 strength
returns as it was. A scheduled LTX-2 LoRA is the exception: what One Phase cut
from it is not kept, and back in Two Phases its stage 2 is filled from stage 1
again (see above). Branch syntax and anything else preserved as imported is
never cut.

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

Video LoRAs — the LTX-2 line especially — often have nothing but videos on
Civitai, and Civitai's API names every video `<id>.mp4` whatever was uploaded.
Each download is therefore saved as what its bytes are, so a WebM stays a WebM
and plays; a web page or an empty response is reported as a failure rather than
saved, so the next fetch tries again instead of counting it as done. A tile with
only a video preview gets its first frame as the thumbnail.

### Civitai API key

Public models need no key. For anything that does, either export
`CIVITAI_API_KEY` before starting WanGP (preferred — it is never written to
disk) or set one through *Manage profiles* (`⋯`) → *Set Civitai API key*, which
stores it in the plugin's settings file **in plain text**. The environment
variable always wins.

## Preferences

Favourites, tags, zoom, the list or cards view, stack profiles and any Civitai
API key you set are stored in
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
  stack_text.py            a stack as "name: x multiplier" text, both ways
  inventory.py             native LoRA list -> displayable entries
  thumbnails.py            preview matching, first-frame decode, cache
  catalogue.py             read the Civitai sidecar folder
  civitai.py               fetch and build that folder on demand
  metadata_store.py        favourites, tags, zoom, sort, view (atomic JSON)
  profile_store.py         stack profiles
  lora_files.py            safetensors headers, byte-exact subset copies,
                           LTX-2 picture/sound classification
  bake.py                  re-fit a group as one LoRA (NumPy QR + SVD)
  derived.py               built-file ids, the store and its link
  builder.py               the background build queue
  stack_modes.py           modes, groups and bakes in the native stack
  stack_files.py           which built files a stack needs, and their state
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
  starts at and what *Clear schedule* restores.
- **Irregular imported schedules.** Resampled into step alignment the first time
  they are seen, which preserves what they render; after that they follow the
  step counter like any other. A list too long to drag usefully is preserved and
  shown read-only instead, with **Fit to N steps** offered.
- **Multi-phase coordinates.** Phase-relative by default. Exact global step
  numbers are claimed only for a schedule that really does span the whole run at
  one slot per step.

- **Building is triggered by the stack, not by a button.** Every payload
  queues whatever built file the stack selects and does not have yet, so a
  mode or bake that arrives from a preset, a profile, a pasted stack or the
  queue is built exactly as one chosen in the panel is.
- **A built file is never a guess.** Splits copy the original's tensor bytes;
  a copy that would keep every tensor is a hard link. A file is written to a
  temporary name and renamed into place only when complete, so WanGP never
  sees half of one; and a manifest of each original's size, time and checksum
  rebuilds anything whose original changed.

## Status and compatibility

Built against **WanGP 12.642** — the plugin API, `refresh_lora_list`,
`get_lora_dir`, the `loras_choices` / `loras_multipliers` components and the
multiplier grammar in `shared/utils/loras_mutipliers.py` were all verified
against that source. MiniMax H3's staged phases and *Phase 2 Steps*
(`num_inference_steps2`), and WanGP dropping `#` lines everywhere it reads the
multiplier text, were verified against WanGP 18.83. Every WanGP global is read through a guarded lookup, so a
build that renames or drops one degrades gracefully rather than breaking the
tab.

The automated suite runs without WanGP, and the browser tests drive the real
panel against the real Python bridge. The in-app acceptance pass — installing
from GitHub into a live WanGP instance and walking the checklist in the design
spec against actual generations — still needs to be done on a machine with WanGP
running.

## Licence

MIT — see [LICENSE](LICENSE).
