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
- **Continuous strength sliders** plus exact numeric fields, per guidance phase,
  synced back to WanGP with a short debounce. Values are not clamped to `0..1`.
- **Image previews first, video previews as a still.** A LoRA with only a video
  preview gets its *first frame* decoded server-side and cached; the video itself
  is never loaded in the browser.
- **Stack profiles** — lightweight LoRA-only recalls, plus favourites, tags and a
  persistent thumbnail zoom.
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
| `0.8` | one multiplier | slider |
| `0.4;0.9` | per guidance phase | one slider per visible phase |
| `1,0.5,0.25` | step/time varying schedule | shown read-only, preserved verbatim |
| `0.5:0.9` | LoRA multiplier branches | shown read-only, preserved verbatim |
| `1 1\|0.8` | accelerator boundary | LoRAs left of the bar are marked *WanGP-managed* |

Schedules the simple sliders cannot represent are **never silently flattened**.
They round-trip byte for byte, and replacing one with a plain multiplier takes an
explicit confirmation.

Strengths are stored against a stable LoRA identity (the relative path WanGP
itself uses), never against a row index — so removing the middle LoRA of a stack
cannot shift another LoRA's multiplier onto the wrong file.

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

### Security notes

- The LoRA directory is **never** exposed via `gr.set_static_paths()`. Tiles
  receive a small derived WebP as a `data:` URI.
- The frontend addresses LoRAs by stable ID only; paths are resolved server-side
  and validated against the current model's LoRA root.
- Absolute paths never reach the browser, and all display text is inserted as
  text nodes rather than HTML.

## Preferences

Favourites, tags, zoom and stack profiles are stored in
`wan2gp_lora_browser.json` next to WanGP's own config file (falling back to a
`data/` folder inside the plugin if that location cannot be discovered). Writes
are atomic, and a corrupt file is backed up rather than crashing WanGP.

## Development

```bash
pip install pytest Pillow
python -m pytest
```

The test suite covers the parts that can be reasoned about without a running
WanGP: multiplier parsing and serialisation, state reconciliation, phase
switching, preview matching and caching, profile and metadata persistence, and
the JSON bridge in `plugin.py` (against stubbed WanGP/Gradio modules).

```
plugin.py                  WanGP lifecycle, component wiring, JSON bridge
lora_browser/
  multiplier_codec.py      parse/serialise loras_multipliers
  ui_payloads.py           state model, phase resolution, action application
  inventory.py             native LoRA list -> displayable entries
  thumbnails.py            preview matching, first-frame decode, cache
  metadata_store.py        favourites, tags, zoom (atomic JSON)
  profile_store.py         stack profiles
assets/                    panel CSS and JavaScript
```

## Status and compatibility

Built against **WanGP 12.642** — the plugin API, `refresh_lora_list`,
`get_lora_dir`, the `loras_choices` / `loras_multipliers` components and the
multiplier grammar in `shared/utils/loras_mutipliers.py` were all verified
against that source. Every WanGP global is read through a guarded lookup, so a
build that renames or drops one degrades gracefully rather than breaking the
tab.

The automated suite runs without WanGP; the in-app acceptance pass (installing
from GitHub into a live WanGP instance and walking the checklist in the design
spec) still needs to be done on a machine with WanGP running.

## Licence

MIT — see [LICENSE](LICENSE).
