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
  value; tapping one chooses which phase the controls under it edit. A phase
  run per step has no single value, so its chip wears a `∿ Per step` badge
  instead of a number — `∿` is the panel's mark for "the strength changes from
  step to step".
- **Every active row starts with its LoRA's picture** — the tile's preview, a
  video preview's first frame with a play button, or a placeholder — in one
  column the same width on every row, so the rows stay aligned. Pressing the
  picture opens Inspect.
- **One strength or per step.** Each phase runs at one strength or per step —
  one or the other. Per step is WanGP's comma multiplier (`1,0.8,0.4,0`) as a
  bar for every inference step, per phase — and on LTX-2 and MiniMax H3, per
  pipeline stage: tap or drag across bars to select steps, then set them with
  the same slider, number and `±` as one strength, any value from −10 to 10.
  The open editor fits the width it has, phone included, and *Full screen*
  opens it in a sheet when you want the room. See
  [Step schedules](#step-schedules).
- **Switch a LoRA off without losing it.** Every row has an on/off switch: off
  keeps the strength, the schedule and the LoRA's place in the list, and WanGP
  neither loads nor applies it. See [Switching a LoRA off](#switching-a-lora-off).
- **Copy one LoRA's schedule onto others.** *Copy* at the end of a LoRA's
  strengths takes its strength and schedule, every phase and step; *Paste* on
  another LoRA gives it the same, and *Paste* in **Select** gives it to every
  LoRA selected. See [Copying a schedule](#copying-a-schedule).
- **Paste a whole stack as text.** *Paste stack* opens a box: paste one
  `name: x multiplier` line per LoRA, press Enter, and that is the stack —
  schedules and all. *Copy current stack* writes the same format back out. See
  [Pasting a stack as text](#pasting-a-stack-as-text).
- **Compare any two videos' recipes.** *Compare* opens the video selected in
  WanGP's gallery beside the newest one — or any two from the gallery, or
  files dropped on either side — both playing together, with their prompts
  and LoRAs lined up underneath: grey where both apply the same, green where
  a side applies more, red where it applies less — per LoRA, and step by
  step. Tune the right one's weights right there and *Apply* its LoRA values,
  its prompt, or both. See [Comparing two videos](#comparing-two-videos).
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
  grouped, so they move as one, and a group can be baked into a single LoRA to
  save memory — and unbaked again. See [Groups and bakes](#groups-and-bakes).
- **Shrink a big stack.** *Shrink* bakes every LoRA that runs at one strength
  into one file, each at its strength, and every set of LoRAs that share a
  schedule into a file of their own that runs it — twenty LoRAs at one
  strength and two schedules shared by five more become three files, applying
  the same. LoRAs whose strengths differ from phase to phase get **a file per
  phase**, each running in its own phase only, so they need not match each
  other phase for phase. *Unshrink* gives them back. See
  [Shrinking a stack](#shrinking-a-stack).
- **Low, Medium or High compression.** Before anything is built, the panel
  measures every layer of the LoRAs and shows what each level keeps of their
  combined change and how much it weighs. Each level is the smallest file that
  keeps its share: every layer gets its own rank, the one that keeps the most
  per byte. See [Compression levels](#compression-levels).
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
| `1,0.5,0.25` | step/time varying schedule | per step: a bar a step, spread over the whole run |
| `0,0,0.5,0.5` | a schedule that is off at first | per step: two bars at 0, two at 0.5 |
| `1,0.8;0.7,0.5` | a schedule inside each phase | per step, in each phase |
| `1;0.7,0.5,0.2` | scalar phase + scheduled phase | phase 1 at one strength, phase 2 per step |
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

A comma in a WanGP multiplier makes it vary over the run. In the panel that is
**Per step**: each phase runs at **one strength** or **per step** — one or the
other — and per step is a bar for every step, each at a value of its own.

```
[Full screen]      [Inspect]       [Remove]

STRENGTH
[ Phase 1  ∿ Per step ]  [ Phase 2  0.5 ]
[    One strength    |   ∿ Per step    ]
Each of the 8 steps runs at a value of its own. One strength goes back to 0.7.

 1    1   0.88 0.88 0.25 0.25  0   −2
 █    █    █    █    ▄    ▄
 ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔   zero
                                   █
 1    2    3    4    5    6    7   [8]
Step 8 selected.                  [All] [None]

[−]  ------------ slider ------------  [−2]  [+]
−2 is outside the slider's 0–1: the number is what applies.
```

- **Tap a bar to select it**, tap again to let it go, or drag across bars to
  select every one you pass; *All* and *None* do the rest. With a keyboard, Tab
  to a bar and press Space. A sideways drag selects; a finger moving up or down
  scrolls the page as it would anywhere else.
- **The selected steps share one control** — the same slider, number field and
  `−`/`+` one strength has. The slider and a typed number set every selected
  step to that value; `−`/`+` move each one by 0.01 from its own. Steps that
  disagree show `mixed` in the field until you set them.
- **The bars are 0 to 1, the values are not.** Type any strength WanGP takes,
  `−10` to `10`: a bar past 1 tops out with a cap, a negative one hangs under
  the zero line (capped past −1), and the line under the control says when a
  value is outside the slider. Its number is printed over its bar, or, on a run
  of more than 12 steps, in its tooltip and the field.
- **Per step starts where the phase was.** Every step starts at the phase's one
  strength, so nothing WanGP applies changes until you set a step.
- **One strength puts the steps aside.** The phase goes back to its one
  strength, and Per step brings the values back exactly as they were, while
  WanGP runs.
- Each phase is its own choice: `∿ Per step` on a phase chip says that phase
  runs per step. Linking phases acts on one strength, so a phase run per step
  is left out of it.

**`∿` means per step.** A phase whose strength changes from step to step shows
a solid `∿ Per step` badge on its chip where a one-strength phase shows its
number, and the list shows how many steps it runs on. The badge is drawn in the
theme's text colour, so it stands out in any theme — including WanGP's dark
one, whose accent is the same white as its text. The chip's own outline still
marks the phase being edited, so the two never share a cue.

**Full screen, when you ask for it.** The open editor is laid out for the width
it has — a phone, a narrow panel or a wide one — so it works where it is. *Full
screen* (or the ⛶ button on a card) opens the same editor in a sheet over the
page, its bars the width of the window and taller. It is live: everything set
there goes straight to WanGP. *Exit full screen*, the backdrop or Escape put it
back in the panel; *Done* closes the LoRA.

**Where per step can be exact.** WanGP stretches each phase's comma list to
fill the steps that phase runs, so a list runs one value per step only when its
length matches them. That decides where the panel offers per step:

- **One Phase, on every model.** Phase 1 covers the whole run, so a 30-value
  schedule at 30 steps runs exactly as set.
- **LTX-2 and MiniMax H3, in either mode.** Their phases are not cut out of one
  run at a switch step: each is a pipeline stage whose length is known while you
  edit, so each phase gets exactly as many bars as its stage runs steps (below).
- **Elsewhere, with two or more phases** — Wan 2.2, say — phase 1 covers only up
  to `model_switch_step`, derived at generation time from the sampler's
  timesteps and the switch threshold and unknowable while editing, so bars
  would show something WanGP is not going to do. The panel offers none there,
  and **does not touch a schedule either**: the row says what the schedule does,
  read-only, and One Phase edits it again exactly as it was.

**Switching phase mode never changes phase 1.** A phase 1 run per step is the
same steps, and a one-strength phase 1 the same strength, in One Phase and in
Two.

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
set. So when such a list enters Two Phases, it stays stage 1's, and **stage 2
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
preset. What stage 2 gets is ordinary steps with no link to stage 1: edit
either stage and the other stays as it is.

**Fill from stage 1.** The same carry, on request: phase 2 at one strength,
while phase 1 runs per step, offers it, and its tooltip shows what it will
write. Disabled, it says why there is nothing to carry. Afterwards phase 2 runs
per step; *One strength* puts back what it was before.

**One strength control, never two.** A phase run per step has no one strength
to show — it is its steps, so a single number would be one WanGP never applies
— and a phase at one strength has no bars.

**One bar per inference step, always.** There is no resolution to choose: the
bars follow the step counter live — change it in WanGP and every schedule in
the panel comes with it, without being asked.

How a schedule follows depends on where it came from, because a slot means two
different things:

- A schedule **set in the panel** is step-aligned — bar *i* is step *i* — so
  the step count only moves where the run ends; nothing on it moves or
  stretches. Going 4 → 5 steps keeps steps 1–4 exactly as they were and adds a
  step 5 at 0. Going 5 → 3 stops the run after step 3, and WanGP is given three
  values.
- **A lower step count hides; it does not delete.** What a shorter run no longer
  reaches is kept — the steps say `+2 steps kept past the end of the run` — and
  raising the count brings it back exactly as it was. So 5 → 3 → 5 is the
  schedule you started with, and typing `12` over `4`, which passes through `1`,
  loses nothing. The memory lasts while WanGP runs; a restart or a model switch
  starts again from the multipliers as they stand.
- A schedule **that arrived from a preset, an `.lset` file or a hand edit** was
  authored at its own resolution, and WanGP spreads it across the whole run.
  Truncating that would change what it renders, so it is resampled into step
  alignment once — and is step-aligned from then on.

Read-only multipliers are never refitted: branch syntax, an ambiguous phase
count, or a list too long for its steps all stay exactly as imported. A list
too long for its steps is offered one way back to editable — **Fit to N
steps**, which resamples it to one value per step and rewrites the multiplier.

This is the one place the panel rewrites a schedule you did not touch. Binding
the steps to the step counter is what makes the two agree, and the trade is
that changing steps writes `loras_multipliers` for every scheduled LoRA.

**Looking writes nothing.** Opening a LoRA, picking a phase or selecting steps
leaves the native multiplier exactly as it was, and an imported schedule is
rendered, never re-serialised, until you make a material edit. Per step itself
writes one value per step, every one the strength the phase already had.

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

### Copying a schedule

Every LoRA's strengths end with **Strength and schedule — Copy · Paste** (on
a card, whose height is its picture's, they are the two icons in its top
line). **Copy** takes that LoRA's multiplier as WanGP holds it — every phase, every
step — and **Paste** on another LoRA gives it exactly that. In **Select**, the
bar's **Paste** gives it to every LoRA selected at once (both sides of one with
separate picture and sound strengths, each of which also has its own Copy and
Paste). Pasting onto a group sets the group's strength and schedule; onto a
shrunk file, the file's — which scales everything in it.

What is pasted is checked and settled the way a pasted stack is: a list
written for another step count is spread over this run's steps as WanGP
spreads it, a phase this mode does not run is kept for when it does, and
WanGP's own LoRAs are never pasted onto. Paste is greyed out on a LoRA that
already has what was copied. The copy lasts the visit (and this browser's
next ones) and goes on the system clipboard too, where the browser allows, to
paste into notes or WanGP's own multiplier box.

LoRAs that share a schedule exactly go into one file when you
[shrink](#shrinking-a-stack): pasting one schedule onto several is also a way
to shrink them further.

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
exactly as written and open per step like any other schedule.

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

### Comparing two videos

Tuning a recipe means rendering variants and looking for what changed.
**Compare**, above the active LoRAs, opens a pop-up over WanGP — most of the
window, with WanGP dimmed round its edges — that puts two recipes side by side:

```
 Compare                          [ Play both ] [⟲] [×]
 Gallery video             [×]   Newest video     Edited [×]
 1 of 12 · one.mp4               12 of 12 · three.mp4
 ┌───────────────────────┐      ┌───────────────────────┐
 │        video          │      │        video          │
 └───────────────────────┘      └───────────────────────┘
 [‹] [›]             [mute]     [‹] [›]             [mute]
 Both 8 steps, One Phase   ▲ more weight   ▼ less weight   grey: the same

 Prompt                                      Differs  [Show all]
 [ a [red] fox in the snow ]      [ a [grey] fox in the snow [at] [dusk] ]
 Detail_Enhancer                                        Per step
 [ ▆ ▆ ▅ ▅ ▂ ▂ _ _    → ]      [ ● ▆ ▆ ▆ ▆ ▃ ▂ _ _ ]   ← gold fade: differs
 Neg_Blur                                            One strength
 [ −1 ▲             → ]      [ ● [−] −2 ▼ [+] ]
 ── THE SAME ON BOTH SIDES (3) ─────────────────────────────────
 Film_Grain                                    One strength · same
 [ 0.6                ]      [ ● [−] 0.6 [+] ]
```

- **It opens on the video selected in the gallery, left, and the newest,
  right** — each taken as it was when it was picked, so a render finishing
  meanwhile does not swap it. With no video yet, the right is your settings
  as they are.
- **Either side can be any video.** ‹ › step a side through the gallery
  without closing the pop-up. × clears a side: drop a video there, or
  *Browse…* for one — MP4, MOV, M4V, MKV or WebM, or a settings JSON — and a
  file dropped on a side that shows a video takes its place. The settings
  WanGP saved in the file are read in the browser, from the bytes that hold
  them, and the video plays from the file itself. ⟲ takes a side from a file
  back to the gallery.
- **The left is read-only; the right is yours to edit.** Another video on
  the right is another recipe, so edits made to the last one are dropped.
- **Both videos play together**: *Play both*, *Restart*, and each its own
  mute. They loop, so a difference has time to show.
- **The prompts come first**, side by side: a few lines each until *Show
  all* (or a tap on one) opens them, with the words one has and the other
  does not marked — in the accent colour, since green and red are about
  weight. Prompts are not edited here.
- **One row per LoRA**, its name once across both sides, both sides
  scrolling as one, and each side's values on one line — the right's switch
  beside them, any tags under them — so they line up across. On a phone, the
  right's value is the target: tap it for the tray, which has −/+ and the
  slider.
- **What differs comes first.** Every LoRA (and the prompt) that differs
  between the two is listed at the top; what is the same on both sides comes
  last, under a divider. The order is set when the two sides are picked, so a
  row never moves away while you edit it.
- **Gold means something differs**: the right side of every LoRA that
  differs fades from gold at its left edge, over its green or red — those say
  which way, gold says that it does. Editing recolours it at once. The gold is
  a CSS variable too, `--lb-changed`. A LoRA is the file you know, whatever carried it: a
  visual-only copy, a sound half, a group or a bake line up with the LoRA
  itself, and say so in a small label.
- **Colours are about weight, not about matching.** Grey: both apply the
  same. Otherwise the side that applies more is green (▲) and the side that
  applies less is red (▼) — so one side is green where the other is red.
  More is the higher number: −1 is more than −2. A LoRA a side does not have,
  or has switched off, counts as 0 there. One strength against per step is
  compared step by step, the one strength counting as that value at every
  step; two phases are compared phase by phase; runs of different lengths at
  the same point of the run. The colours are two CSS variables, `--lb-more`
  and `--lb-less`.
- **Editing the right** works as the editor does, on what the right already
  has: −/+ on a strength, or tap it for the tray — the 0–1 slider, the number
  field (any value, −2 included) and −/+. Tap or drag across steps to set
  several at once; each phase stays one strength or per step. The switch turns
  a LoRA off. → on a left row puts that LoRA's value on the right, adding the
  LoRA if the right does not have it. Every edit recolours both sides at once;
  a group's members move as one.
- **Apply…** asks what to take from the right side:
  - *LoRA values* — its LoRAs and multipliers, edits and all, fitted to the
    guidance mode WanGP is in. Every phase is the right side's: nothing of
    what was set before carries into a phase it does not spell out.
  - *Prompt* — its prompt, into WanGP's prompt box (and its prompt wizard, the
    way WanGP's own presets set one). Off when the right side has none.
  - *Both*.

  The line beside it says what was applied, or why not. Nothing changes
  before that: looking, stepping through the gallery and editing are the
  pop-up's own. *Reset* goes back to the right side's own recipe. Escape puts
  the choice away; Escape again, × or a tap outside closes the pop-up.

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
`.sound.safetensors`), both at the compression you choose. Switching back to one
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
LoRA at the [compression level](#compression-levels) you choose, the one you
last picked to start with. The dialog lists each level with the share it keeps
and its size, measured before anything is built, and **Exact**, every LoRA as it
is (only worth it to hand someone one file). Once it is built, the group says
how much of the change it kept. The originals are never touched: **Unbake**
goes back to them and **Change size…** bakes again at another level.
Strength and schedule still apply live to the baked file, and changing a
member's mode or the group's members bakes it again to match. LoRAs that are not
plain low-rank pairs (DoRA, LoKr, full weight differences) cannot be baked, and
neither can a LoRA WanGP recognises by its file name (see
[Shrinking a stack](#shrinking-a-stack)); the dialog says so.

A group is stored as `#lora_group <letter> <lora>` lines in `loras_multipliers`;
a baked group's lines also name its file
(`_lora_browser/_bakes/<hash>.q999.safetensors` at a compression level keeping
99.9%, `.r<rank>` for one rank in every layer), which is what
`loras_choices` then holds. A group with separate picture and sound strengths
lists each member as the halves it is selected as; nothing else marks it.

### Shrinking a stack

A big stack takes a lot of memory: WanGP keeps every LoRA's factors loaded.
**Shrink**, above the active LoRAs, bakes the stack down into as few files as
run the same:

- **Every LoRA at one strength** — the same value in every phase and at every
  step — goes into **one file**, each baked in at its strength, and that file
  runs at **1**. A LoRA at 0.7 still brings 0.7; a LoRA at −1 still takes away.
- **LoRAs that share exactly the same schedule** (or the same strengths per
  phase) go into **a file per schedule**, each at 1, and that file runs the
  schedule: the sum of LoRAs that each follow a schedule is that sum following
  it. A LoRA with no other on its schedule stays as it is.
- **Strengths that differ from phase to phase** — 0.8 in phase 1 and 0.5 in
  phase 2 — need not match from LoRA to LoRA: those LoRAs get **a file per
  phase**. Phase 1's file holds each of them at its phase 1 strength and runs
  `1;0`; phase 2's holds each at its phase 2 strength and runs `0;1`. See
  [A file per phase](#a-file-per-phase).
- **Each file is built at the [compression level](#compression-levels) you
  pick in the dialog** — Low, Medium or High, each shown with the share it
  keeps and its size before anything is built. Once built, its row says how
  much of the combined change it kept; **Change size…** builds it again at
  another level.
- **Modes count**: an audio-only LoRA goes in as its sound side, a visual-only
  one as its picture side, and a LoRA with separate picture and sound
  strengths as its picture half at the one and its sound half at the other.
  Its two halves go in together or not at all.
- **What goes in** is what runs: LoRAs that are on, on your side of the bar. A
  group's members come in at the group's strength, and anything baked or
  shrunk before comes in as its originals, so shrinking again plans everything
  afresh. **What stays as it is** is listed with why: LoRAs switched off,
  WanGP's own, multipliers with `:` branches, LoRAs not on this machine,
  files a bake cannot combine (DoRA, LoKr, full weight differences), and LoRAs
  WanGP recognises by their file name — LTX-2's IC-LoRAs (`ic-lora`, whose
  `-ref0.5` sets the reference scale), the LoRAs it would otherwise add itself
  (`distilled-lora`, `union-control`, `in-outpainting`, `id-lora-celebvhq`),
  and Wan 2.2's Lightning I2V accelerators. Inside a file with a name of its
  own WanGP would not see them for what they are, and would even add its own
  default beside them.

Why one strength bakes in exactly: WanGP (mmgp) adds each LoRA to a layer as
its change times its multiplier, `W + 0.5·ΔA + 0.7·ΔB`, so a file holding
`0.5·ΔA + 0.7·ΔB` run at 1 is the same sum, at every step. What can differ is
the rank: the file keeps the strongest directions of the combined change, and
its row says how much of it it kept. A subtle LoRA beside strong ones is the
first to fade — check a render with **Compare**, and pick a lower compression
with **Change size…** if it did.

```
 Shrink the stack
 3 LoRAs into 1 file · WanGP loads 2 files instead of 4, about 410 MB of
 LoRA weights instead of 1.2 GB.
 Compression
 (•) Low compression · keeps 99.9%                         410 MB
     Saves 790 MB, 66% smaller than now.
 ( ) Medium compression · keeps 99.0%                      230 MB
 ( ) High compression · keeps 95.1%                        120 MB
 Matching the highest-rank LoRA in every layer, as Shrink did before:
 keeps 93.4%, 300 MB.
 ┌ At one strength   3 LoRAs → 1 file · keeps 99.9% · 410 MB ┐
 │ Detail_Enhancer                                     at 0.8 │
 │ Film_Grain                                          at 0.6 │
 │ Neg_Blur                                             at −1 │
 └────────────────────────────────────────────────────────────┘
 ▸ 1 LoRA stays as it is
                                 [ Cancel ] [ Shrink 3 LoRAs ]
```

The dialog shows the plan first; nothing changes until **Shrink**. Each file
then builds in the background — until it is ready, WanGP reports it missing if
you generate, and the panel warns about exactly that. A shrunk file is a row of
its own, *Shrunk A*, its LoRAs listed under it with the strength each brings.
Its own strength scales them all. `×` takes one out (it runs on its own again,
and the rest are baked again), and **Unshrink** gives every one back as a LoRA
of its own at the strength it had — times the row's strength, if you changed
it.

#### A file per phase

In a run with two or three phases, LoRAs whose strengths differ from phase to
phase are shrunk **a file per phase** — however their phases pair up. Twenty
LoRAs at twenty different pairs of strengths become two files:

- **Phase 1's file** holds every one of them at its phase 1 strength, **phase
  2's file** every one at its phase 2 strength.
- **A phase's file runs in that phase only.** Its multiplier is 0 in every
  other phase, whatever you set, type or paste — the other phases' chips are
  locked on it, and a pasted schedule keeps only its own phase. So phase 1's
  file can never apply in phase 2, nor phase 2's in phase 1.
- **It is exact for the same reason one strength is.** In each phase WanGP
  adds each LoRA's change times that phase's multiplier, and skips a LoRA
  whose multiplier is 0 there, so phase 1 runs `0.8·ΔA + 0.6·ΔB + …` from its
  file alone, and phase 2 `0.5·ΔA + 0.6·ΔB + …` from its own.
- **Steps inside a phase still count.** LoRAs that run the same steps in phase
  1 and differ only in phase 2 get a phase 1 file that runs those steps.
- **Only where it means fewer files.** A phase nothing brings anything to has
  no file; every LoRA the same in each phase is one file for every phase, as
  above; and two LoRAs would be two files either way, so they stay as they
  are.
- **Memory**: each phase's file holds every LoRA, so it takes about as much
  as one file would, once per phase. Each step still runs one file where it
  ran all of them; the dialog says when the files take more memory than the
  LoRAs they replace.

The row, *Shrunk A — 3 LoRAs in a file per phase*, has an editor per file
(*Phase 1 file*, *Phase 2 file*), each set in its own phase. Raising phase 1's
file to 1.2 raises every LoRA's phase 1 strength by a fifth and leaves phase 2
as it was; **Unshrink** gives each LoRA both its phases back, each times its
file's strength. In One Phase there is one phase to bake, so one file.

#### Compression levels

A smaller file keeps less of what the LoRAs change; how much less depends on
the LoRAs. So before anything is built, the panel **measures** them: for every
layer it works out the directions their combined change is made of, and how
much of it each holds (the singular values, from two small matrices per layer;
the change itself is never formed). That takes seconds to a minute or so for a
big stack, in the background, and is remembered for the session. From it the
dialog shows, for each level, the **share of the combined change kept** and
the **size**, exactly as the build will make them:

| Level | Keeps of the combined change | No layer keeps less than |
|---|---|---|
| **Low compression** | 99.9% | 99.5% of its own |
| **Medium compression** | 99% | 95% of its own |
| **High compression** | 95% | 75% of its own |

Each level is the **smallest file that keeps its share**. Rather than one rank
everywhere, every layer gets its own: the directions that hold the most change
per byte go in first, across every layer together, until the share is kept —
and no layer is cut below its floor, so a faint layer is never gutted to save
a few bytes. At the same share that is smaller than one rank for every layer,
and at the same size it keeps more. WanGP reads each layer's rank from that
layer's own tensors, so such a file loads like any other.

The dialog also shows what **matching the highest-rank LoRA in every layer** —
how files were sized before — would keep and weigh, so you can see on your own
stack whether a level saves memory over it. Whether a level is smaller than
that depends on the stack: LoRAs that pull the same way fit in few directions,
while many unrelated LoRAs need many, and keeping 99.9% of them can take more
than matching one LoRA's rank did (which kept less). The level you pick is
remembered and used by the next Shrink and Bake. The share is of the weight
change, which is close to, but not the same as, what you see: **Compare** is
the real check.

A shrunk file is a group whose lines say what each LoRA brings to it:

```
1 1,1,0.5,0.5
#lora_group A Detail_Enhancer.safetensors
#lora_group A Film_Grain.safetensors
#lora_group A _lora_browser/_bakes/3f9c21a07b4e.q999.safetensors
#lora_weight A 0.8 Detail_Enhancer.safetensors
#lora_weight A 0.6 Film_Grain.safetensors
```

A file per phase is the same group with a file per phase and a strength per
phase for each LoRA:

```
1;0 0;1
#lora_group A Detail_Enhancer.safetensors
#lora_group A Film_Grain.safetensors
#lora_group A _lora_browser/_bakes/5d71c0e93a2f.q999.p1.safetensors
#lora_group A _lora_browser/_bakes/a40be6f1d8c7.q999.p2.safetensors
#lora_weight A 0.8;0.5 Detail_Enhancer.safetensors
#lora_weight A 0.6;0.6 Film_Grain.safetensors
```

so presets, the queue and every video's saved settings carry everything needed
to build it again. **Copy current stack** writes the same as tags,
`Detail_Enhancer [group A, baked 99.9%, weight 0.8]: x1` (or `weight 0.8;0.5`
with a file per phase), and pasting it back
gives the shrunk file back; profiles keep it too. The compare view shows each
shrunk LoRA at what it brings, and leaves it as it is: unshrink to change one.

### Built files

Visual-only and audio-only copies, baked groups and shrunk files are kept in
the plugin's own folder, under `derived/<LoRA folder>-<hash>/`, and linked into
the model's LoRA folder as `_lora_browser/` — a symlink, a junction on Windows,
or a plain folder where neither can be made — which is how WanGP finds them
like any other LoRA. They never show in the thumbnail grid. **Built files** in
the status bar lists them with what each was built from, its size and what uses
it (this stack or a saved profile). **Delete** removes one; **Delete unused**
removes those nothing uses; **Delete all but this stack's** removes every one
this stack does not use, saved profiles' included.

**Every built file's recipe is kept** — a copy's original and mode; a bake's
or a shrink's LoRAs, what each brings, and its size — in `recipes.json` beside
the files, and deleting a file keeps it. Whenever a stack, a recalled profile,
a preset or a video's settings names a deleted file again, it is built again
from its originals. That works even where the settings lost the lines saying
what a baked file holds: its recipe gives the group back. The Built files list
shows the recipes kept for deleted files.

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
- No LoRA's absolute path reaches the browser, and all display text is
  inserted as text nodes rather than HTML — prompts in the compare view
  included. The compare view plays gallery videos from the Gradio file URLs
  WanGP's own gallery uses, for the files WanGP lists in it; a dropped file is
  read in the browser and only the settings found in it, and its name, are
  sent.

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
WanGP: multiplier parsing and serialisation, schedule compilation, setting
steps and the regions under them, state reconciliation, phase switching,
preview matching and caching, profile and metadata persistence, and the JSON
bridge in `plugin.py` (against stubbed WanGP/Gradio modules).

The panel itself is also tested in a real browser, against that same Python
bridge rather than a mock of it — the grid's three-row cap, the weight controls,
selecting steps by dragging across them, and the editor at phone width are all
things only layout and pointer events can answer.
Those tests skip unless Playwright and a Chromium build are present:

```bash
pip install playwright && playwright install chromium
python -m pytest tests/test_panel_browser.py
```

```
plugin.py                  WanGP lifecycle, component wiring, JSON bridge
lora_browser/
  multiplier_codec.py      parse/serialise loras_multipliers
  schedule.py              step schedules: compile, reconstruct, set steps
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
  compare.py               two recipes side by side, and edits written back
  shrink.py                a stack baked down: what goes into which file, and
                           the shrunk groups that replace them
assets/                    panel CSS and JavaScript
```

### Decisions worth knowing

Four things the design spec leaves to implementation, settled here:

- **Numeric precision.** Direct entry keeps four decimals — the same precision
  the serialiser emits — rather than the two the editor used to round to. An
  untouched imported token is not rewritten at all, whatever its precision.
- **Linked phases.** Linking moves the one strength of every visible phase
  that has one, skipping any phase run per step. It never copies steps between
  phases; schedules stay independent.
- **What a phase is worth per step.** Its steps and nothing else: there is no
  base hiding behind them. The strength a phase had before Per step starts
  every step at it, and is what *One strength* goes back to.
- **Irregular imported schedules.** Resampled into step alignment the first time
  they are seen, which preserves what they render; after that they follow the
  step counter like any other. A list too long for its steps is preserved and
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
