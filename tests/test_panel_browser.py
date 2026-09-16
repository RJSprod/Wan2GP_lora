"""The panel driven in a real browser, against the real Python bridge.

Everything below the DOM is already covered by the other suites.  What only a
browser can answer is whether the rules survive contact with layout and pointer
events: that the grid really stops at three rows, that a tile's Inspector button
does not toggle the LoRA underneath it, that an imported 0.71 is still 0.71
after rendering, and that dragging a region ends up as the multiplier WanGP will
actually be given.

Skipped unless Playwright and a Chromium build are both present, the same way
the thumbnail tests skip without Pillow.
"""

import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

from . import test_plugin_bridge as bridge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCE = "browsertest"
NS = "wgp-lora-browser-browsertest"

#: Enough LoRAs that the grid needs more than the three rows it may show.
LORAS = [f"lora_{index:02d}.safetensors" for index in range(1, 41)]
MODEL_DEF = {"guidance_max_phases": 2, "lora_multiplier_phases": 2}
STATE = {"model_type": "browser_model", "loras": list(LORAS)}

#: A simple token with values the slider cannot represent, a two-phase
#: schedule, and a branch token that must stay read-only.
START = {
    "selected": ["lora_01.safetensors", "lora_02.safetensors", "lora_03.safetensors"],
    "multipliers": "0.71;1.23 1,0.8,0.4,0;0.7,0.5,0.2,0 0.5:0.9",
}

#: What the step counter says, so slot counts can be checked against it.
STEPS = 30

IDS = {
    "__INST__": INSTANCE,
    "__NS__": NS,
    "__ROOT__": "lb_root_b",
    "__PAYLOAD_ID__": "lb_payload_b",
    "__ACTION_ID__": "lb_action_b",
    "__ACTION_BTN__": "lb_action_btn_b",
    "__THUMB_REQ_ID__": "lb_thumb_req_b",
    "__THUMB_BTN__": "lb_thumb_btn_b",
    "__THUMB_RES_ID__": "lb_thumb_res_b",
    "__REFRESH_BTN__": "lb_refresh_btn_b",
    "__NATIVE_CHOICES__": "lb_native_choices_b",
    "__NATIVE_MULTIPLIERS__": "lb_native_multipliers_b",
}

# Gradio's job, reduced to what the panel actually depends on: hidden text
# fields for the payload and the action, a button that submits the action, and
# the callback that pushes the answer back in.
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>panel</title><style>
body { margin:0; font-family:system-ui,sans-serif; background:#111216; color:#ececf1; font-size:13px;
  --border-color-primary:#3a3c42; --input-background-fill:#222328; --background-fill-primary:#18191d;
  --background-fill-secondary:#1d1e23; --body-text-color:#ececf1; --body-text-color-subdued:#a8abb2;
  --color-accent:#4c8dff; --button-secondary-background-fill:#222328;
  --button-secondary-text-color:#ececf1; --input-text-color:#ececf1; }
.wrap { max-width:1180px; margin:16px auto; padding:0 12px; }
.bridge { position:absolute; left:-9999px; }
__CSS__
</style></head><body><div class="wrap">
<div id="__ROOT__"></div>
<div class="bridge">
  <div id="__PAYLOAD_ID__"><textarea></textarea></div>
  <div id="__ACTION_ID__"><textarea></textarea></div>
  <div id="__ACTION_BTN__"><button type="button"></button></div>
  <div id="__THUMB_REQ_ID__"><textarea></textarea></div>
  <div id="__THUMB_BTN__"><button type="button"></button></div>
  <div id="__THUMB_RES_ID__"><textarea></textarea></div>
  <div id="__REFRESH_BTN__"><button type="button"></button></div>
  <div id="__NATIVE_CHOICES__" class="block"><select multiple></select></div>
  <div id="__NATIVE_MULTIPLIERS__" class="block"><textarea></textarea></div>
</div></div>
<script>
(function () {
  var action = document.querySelector("#__ACTION_ID__ textarea");
  var payload = document.querySelector("#__PAYLOAD_ID__ textarea");
  function push(text) {
    payload.value = text;
    var api = window.wgpLoraBrowser && window.wgpLoraBrowser["__INST__"];
    if (api) { api.apply(text); }
  }
  window.__push = push;
  document.querySelector("#__ACTION_BTN__ button").addEventListener("click", function () {
    fetch("/act", { method: "POST", body: action.value })
      .then(function (response) { return response.json(); })
      .then(function (data) { window.__native = data.multipliers; push(data.payload); });
  });
  document.querySelector("#__THUMB_BTN__ button").addEventListener("click", function () {
    var api = window.wgpLoraBrowser && window.wgpLoraBrowser["__INST__"];
    if (api) { api.applyMedia(JSON.stringify({ kind: "thumbs", thumbs: {} })); }
  });
})();
</script>
<script>__SCRIPT__</script>
<script>
fetch("/payload").then(function (r) { return r.json(); }).then(function (data) {
  window.__native = data.multipliers;
  window.__push(data.payload);
});
</script></body></html>"""


def _asset(name):
    with open(os.path.join(ROOT, "assets", name), encoding="utf-8") as handle:
        return handle.read()


class _Host:
    """The plugin plus the native state a browser session mutates."""

    def __init__(self, tmp_path):
        module = bridge._load_plugin_module()
        self.plugin = module.LoraBrowserPlugin()
        self.plugin.server_config_filename = str(tmp_path / "wgp_config.json")
        self.plugin.get_state_model_type = lambda state: state.get("model_type", "")
        self.plugin.get_model_def = lambda model_type: MODEL_DEF
        self.plugin.get_lora_dir = lambda model_type: str(tmp_path / "loras")
        self.plugin._init_stores()
        self.selected = list(START["selected"])
        self.multipliers = START["multipliers"]
        self.steps = STEPS
        #: Guidance phases, as WanGP's own dropdown would report them. Step
        #: schedules exist only at 1, so that is where this starts; the tests
        #: about more phases switch and switch back.
        self.guidance = 1

    def payload(self):
        return self.plugin._build_payload(
            self.plugin._instance(INSTANCE), STATE, self.selected, self.multipliers,
            self.guidance, self.steps,
        )

    def act(self, action_json):
        choices, mults, text = self.plugin._apply_action(
            self.plugin._instance(INSTANCE), action_json, STATE, self.selected,
            self.multipliers, self.guidance, self.steps,
        )
        if isinstance(choices, dict) and "value" in choices:
            self.selected = list(choices["value"])
        if isinstance(mults, dict) and "value" in mults:
            self.multipliers = mults["value"]
        return text


def _serve(host):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, content_type="application/json"):
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/steps"):
                host.steps = int(self.path.rsplit("=", 1)[-1])
                self._send(json.dumps({"steps": host.steps}))
                return
            if self.path.startswith("/guidance"):
                host.guidance = int(self.path.rsplit("=", 1)[-1])
                self._send(json.dumps({"guidance": host.guidance}))
                return
            if self.path == "/":
                page = PAGE.replace("__CSS__", _asset("lora_browser.css").replace("__NS__", NS))
                page = page.replace("__SCRIPT__", _asset("lora_browser.js"))
                for key, value in IDS.items():
                    page = page.replace(key, value)
                self._send(page, "text/html; charset=utf-8")
                return
            if self.path == "/payload":
                self._send(json.dumps({"payload": host.payload(), "multipliers": host.multipliers}))
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8")
            text = host.act(body)
            self._send(json.dumps({"payload": text, "multipliers": host.multipliers}))

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{port}/"


def _chromium():
    """Playwright's own build, or the one this image ships."""
    candidates = [
        os.environ.get("CHROMIUM_PATH", ""),
        "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    root = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
    if root and os.path.isdir(root):
        for entry in sorted(os.listdir(root)):
            path = os.path.join(root, entry, "chrome-linux", "chrome")
            if entry.startswith("chromium-") and os.path.exists(path):
                return path
    return ""


@pytest.fixture(scope="module")
def panel(tmp_path_factory):
    executable = _chromium()
    if not executable:
        pytest.skip("no Chromium build available for Playwright")

    host = _Host(tmp_path_factory.mktemp("browser"))
    server, url = _serve(host)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=executable, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1100, "height": 1000})
        problems = []
        page.on("console", lambda msg: problems.append(msg.text)
                if msg.type == "error" and "favicon" not in msg.text and "404" not in msg.text
                else None)
        page.on("pageerror", lambda error: problems.append(str(error)))
        page.on("dialog", lambda dialog: dialog.accept())
        page.goto(url)
        page.wait_for_selector(".lb-row", timeout=15000)
        page.wait_for_timeout(400)
        yield page, host, problems
        browser.close()
    server.shutdown()


def native(page):
    return page.evaluate("window.__native")


def expected_values(schedule):
    """What the token should say, compiled from the regions on screen."""
    values = [schedule["gap_strength"]] * schedule["slots"]
    for region in schedule["regions"]:
        for slot in range(region["start"], region["end"] + 1):
            values[slot - 1] = region["strength"]
    return [("%g" % value) for value in values]


def settle(page, ms=500):
    page.wait_for_timeout(ms)


def schedule_of(page, lora_id, phase=0):
    """The schedule the panel is currently showing, as the payload gave it."""
    return page.evaluate(
        """(args) => window.wgpLoraBrowser[args.instance].state.rows
             .find(r => r.id === args.id).phase_schedules[args.phase]""",
        {"instance": INSTANCE, "id": lora_id, "phase": phase},
    )


def drag_by_slots(page, node, timeline, slots, count, handle=None):
    """Drag a region (or one of its handles) a whole number of slots."""
    target = handle if handle is not None else node
    box = target.bounding_box()
    step = timeline.bounding_box()["width"] / float(slots)
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(
        box["x"] + box["width"] / 2 + step * count, box["y"] + box["height"] / 2, steps=8
    )
    return step


def repaint(page, ms=1800):
    """Push a fresh payload, as any WanGP change event would."""
    page.evaluate("""() => fetch('/payload').then(r => r.json()).then(d => {
        window.__native = d.multipliers; window.__push(d.payload); })""")
    settle(page, ms)


def set_guidance(page, count):
    """Move WanGP's guidance-phase dropdown."""
    page.evaluate("(n) => fetch('/guidance?n=' + n)", count)
    settle(page, 250)
    repaint(page)


def open_timeline(page, row):
    if row.locator(".lb-timeline").count() == 0:
        row.locator("button.lb-schedule-toggle").click()
        settle(page, 800)


def draw_region(page, row, at=0.3):
    """Open the timeline and tap once, so the row has a schedule to look at."""
    open_timeline(page, row)
    timeline = row.locator(".lb-timeline")
    timeline.scroll_into_view_if_needed()
    settle(page, 200)
    box = timeline.bounding_box()
    page.mouse.click(box["x"] + box["width"] * at, box["y"] + box["height"] / 2)
    settle(page, 1300)


def select_phase(page, row, index):
    """These tests share one page, so never inherit whichever chip was left on."""
    chips = row.locator(".lb-phase-strip .lb-chip")
    if chips.count() > index:
        chips.nth(index).click()
        settle(page, 500)


class TestPanelInTheBrowser:
    def test_the_grid_shows_exactly_three_rows_and_scrolls(self, panel):
        page, _, _ = panel
        measured = page.evaluate("""() => {
          const grid = document.querySelector('.lb-grid');
          const wrap = document.querySelector('.lb-grid-wrap');
          const tiles = grid.querySelectorAll('.lb-tile');
          const gs = getComputedStyle(grid), ws = getComputedStyle(wrap);
          const cols = gs.gridTemplateColumns.split(' ').filter(Boolean).length;
          return {
            rows: Math.ceil(tiles.length / cols),
            expected: 3 * tiles[0].getBoundingClientRect().height + 2 * parseFloat(gs.rowGap)
              + parseFloat(ws.paddingTop) + parseFloat(ws.paddingBottom)
              + parseFloat(ws.borderTopWidth) + parseFloat(ws.borderBottomWidth),
            height: wrap.getBoundingClientRect().height,
            scrollHeight: wrap.scrollHeight,
          };
        }""")
        assert measured["rows"] > 3
        assert abs(measured["height"] - measured["expected"]) < 1.5
        assert measured["scrollHeight"] > measured["height"] + 10

    def test_the_tile_inspector_opens_without_toggling_the_lora(self, panel):
        page, _, _ = panel
        before = native(page)
        page.locator(".lb-tile").first.locator("button.lb-tile-inspect").click()
        settle(page)
        assert page.locator(".lb-modal-overlay").count() == 1
        assert native(page) == before

        # The flyout is still the near-full-page view it was.
        box = page.locator(".lb-modal").bounding_box()
        assert box["width"] > 900 and box["height"] > 700
        page.keyboard.press("Escape")
        settle(page, 200)

    def test_imported_values_survive_rendering(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_01.safetensors"]')
        assert row.locator(".lb-weight-main input[type=number]").input_value() == "0.71"
        # The slider cannot say 0.71, so it shows the nearest grid stop and the
        # token keeps the exact value.
        assert row.locator(".lb-weight-main input[type=range]").input_value() == "0.7"
        assert native(page).split()[0] == "0.71;1.23"

    def test_a_value_outside_the_slider_range_is_kept(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_01.safetensors"]')
        number = row.locator(".lb-weight-main input[type=number]")
        number.fill("1.23")
        number.press("Enter")
        settle(page, 1200)

        # The slider pins at its maximum and the exact value stays in the token.
        assert number.input_value() == "1.23"
        assert row.locator(".lb-weight-main input[type=range]").input_value() == "1"
        assert native(page).split()[0].split(";")[0] == "1.23"

        number.fill("0.71")
        number.press("Enter")
        settle(page, 1200)

    def test_the_three_weight_controls_have_three_semantics(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_01.safetensors"]')
        number = row.locator(".lb-weight-main input[type=number]")

        # One Phase guidance, so these act on phase 1; phase 2 rides along in
        # the token untouched.
        row.locator(".lb-weight-main button.lb-step").nth(1).click()
        settle(page)
        assert native(page).split()[0] == "0.72;1.23"

        number.fill("0.855")
        number.press("Enter")
        settle(page)
        assert native(page).split()[0] == "0.855;1.23"

        slider = row.locator(".lb-weight-main input[type=range]")
        slider.click()
        slider.press("ArrowRight")
        settle(page)
        value = float(native(page).split()[0].split(";")[0])
        assert round(value * 100) % 5 == 0

    def test_the_scheduler_is_offered_in_one_phase(self, panel):
        page, _, _ = panel
        assert page.evaluate(
            "() => window.wgpLoraBrowser['%s'].state.schedulingEnabled" % INSTANCE
        ) is True
        assert page.locator(
            '.lb-row[data-id="lora_01.safetensors"] button.lb-schedule-toggle'
        ).count() == 1

    def test_opening_a_timeline_changes_nothing(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        before = native(page)
        row.locator("button.lb-schedule-toggle").click()
        settle(page, 700)
        assert row.locator(".lb-timeline").count() == 1
        assert row.locator(".lb-region").count() == 3
        assert native(page) == before

    def test_a_phase_specific_schedule_never_claims_global_steps(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        assert "phase-relative" in row.locator(".lb-sched-head .lb-pill").last.inner_text()

    def test_dragging_a_region_commits_one_canonical_update(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        before = schedule_of(page, "lora_02.safetensors")
        moved = before["regions"][0]

        region = row.locator(".lb-region").first
        region.scroll_into_view_if_needed()
        settle(page, 200)
        start_x = region.bounding_box()["x"]
        drag_by_slots(page, region, row.locator(".lb-timeline"), before["slots"], 2)
        # The preview follows the pointer without a round trip.
        assert row.locator(".lb-region").first.bounding_box()["x"] > start_x
        page.mouse.up()
        settle(page, 700)

        after = schedule_of(page, "lora_02.safetensors")
        assert after["regions"][0]["start"] == moved["start"] + 2
        assert after["regions"][0]["end"] == moved["end"] + 2
        # The token is exactly the picture: regions where they are, zero elsewhere.
        assert native(page).split()[1].split(";")[0].split(",") == expected_values(after)

    def test_a_resize_moves_only_the_dragged_edge(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        before = schedule_of(page, "lora_02.safetensors")
        resized = before["regions"][0]

        region = row.locator(".lb-region").first
        region.scroll_into_view_if_needed()
        settle(page, 200)
        drag_by_slots(
            page, region, row.locator(".lb-timeline"), before["slots"], -1,
            handle=region.locator(".lb-handle-left"),
        )
        page.mouse.up()
        settle(page, 700)

        after = schedule_of(page, "lora_02.safetensors")
        assert after["regions"][0]["start"] == resized["start"] - 1
        assert after["regions"][0]["end"] == resized["end"]
        assert native(page).split()[1].split(";")[0].split(",") == expected_values(after)

    def test_scheduler_mode_replaces_the_plain_strength_control(self, panel):
        """The scrubber is removed, not disabled: it is not what WanGP applies."""
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        open_timeline(page, row)
        assert row.locator(".lb-timeline").count() == 1
        assert row.locator(".lb-weight-main").count() == 0
        # Exactly one strength control on the row, and it belongs to the region.
        assert row.locator("input[type=range]").count() == 1
        assert row.locator(".lb-region-editor input[type=range]").count() == 1
        # Scheduling needs One Phase, so there are no phase chips to link.
        assert row.locator(".lb-phase-strip").count() == 0

        # A row that is not in scheduler mode still has its plain control.
        plain = page.locator('.lb-row[data-id="lora_01.safetensors"]')
        assert plain.locator(".lb-weight-main input[type=range]").count() == 1

    def test_a_collapsed_scheduled_phase_never_offers_a_scrubber(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        before = native(page)
        row.locator("button.lb-schedule-toggle").click()
        settle(page, 600)

        assert row.locator(".lb-timeline").count() == 0
        assert row.locator("input[type=range]").count() == 0
        summary = row.locator(".lb-sched-summary")
        assert summary.count() == 1
        assert native(page) == before

        # And the summary is the way back in.
        summary.click()
        settle(page, 600)
        assert row.locator(".lb-timeline").count() == 1

    def test_closing_the_timeline_keeps_the_schedule(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        before = native(page)
        regions = row.locator(".lb-region").count()
        row.locator(".lb-sched-head button:has-text('Close')").click()
        settle(page)
        assert row.locator(".lb-timeline").count() == 0
        assert native(page) == before

        row.locator("button.lb-schedule-toggle").click()
        settle(page, 700)
        assert row.locator(".lb-region").count() == regions

    def test_clear_phase_returns_it_to_a_plain_strength(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        hidden = native(page).split()[1].split(";")[1]
        row.locator(".lb-sched-head button:has-text('Clear phase')").click()
        settle(page, 900)
        token = native(page).split()[1]
        assert "," not in token.split(";")[0]
        # The phase this guidance mode hides is untouched.
        assert token.split(";")[1] == hidden
        assert row.locator(".lb-weight-main input[type=range]").count() == 1

    def test_an_advanced_token_is_preserved_and_explained(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_03.safetensors"]')
        assert row.locator("code.lb-advanced").inner_text() == "0.5:0.9"
        assert "branch" in row.locator(".lb-preserved-note").inner_text().lower()
        assert row.locator("button.lb-schedule-toggle").count() == 0
        assert native(page).split()[2] == "0.5:0.9"

    def test_fast_taps_all_land(self, panel):
        """Each answer used to be built from state the previous tap had not
        written yet, so a burst sprang back to an older value."""
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_01.safetensors"]')
        select_phase(page, row, 0)
        number = row.locator(".lb-weight-main input[type=number]")
        number.fill("0.50")
        number.press("Enter")
        settle(page, 1200)

        plus = row.locator(".lb-weight-main button.lb-step").nth(1)
        for _ in range(6):
            plus.click(delay=0)
        settle(page, 2500)
        assert number.input_value() == "0.56"
        assert native(page).split()[0].startswith("0.56;")

        # And with no gap at all between them.
        page.evaluate("""() => {
          const row = document.querySelector('.lb-row[data-id="lora_01.safetensors"]');
          const minus = row.querySelectorAll('.lb-weight-main button.lb-step')[0];
          for (let i = 0; i < 8; i++) { minus.click(); }
        }""")
        settle(page, 2500)
        assert number.input_value() == "0.48"
        assert native(page).split()[0].startswith("0.48;")

    def test_a_typed_value_is_not_overwritten_by_an_earlier_answer(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_01.safetensors"]')
        select_phase(page, row, 0)
        number = row.locator(".lb-weight-main input[type=number]")
        row.locator(".lb-weight-main button.lb-step").nth(1).click()
        number.fill("0.37")
        number.press("Enter")
        settle(page, 2000)
        assert number.input_value() == "0.37"
        assert native(page).split()[0].startswith("0.37;")

    def test_a_new_timeline_has_one_slot_per_step(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_01.safetensors"]')
        select_phase(page, row, 0)
        row.locator("button.lb-schedule-toggle").click()
        settle(page, 900)
        slots = page.evaluate("""() => window.wgpLoraBrowser['%s'].state.rows
            .find(r => r.id === 'lora_01.safetensors').phase_schedules[0].slots""" % INSTANCE)
        assert slots == STEPS
        assert row.locator(".lb-tick").last.inner_text() == str(STEPS)

    def test_a_region_can_be_drawn_on_empty_timeline_space(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_01.safetensors"]')
        timeline = row.locator(".lb-timeline")
        timeline.scroll_into_view_if_needed()
        settle(page, 200)
        box = timeline.bounding_box()
        slot = box["width"] / float(STEPS)

        page.mouse.move(box["x"] + slot * 7.5, box["y"] + box["height"] / 2)
        page.mouse.down()
        page.mouse.move(box["x"] + slot * 13.5, box["y"] + box["height"] / 2, steps=10)
        assert row.locator(".lb-region.lb-ghost").count() == 1
        page.mouse.up()
        settle(page, 1200)

        assert row.locator('[data-lb="region-range"]').inner_text() == "Slots 8\u201314"
        # A new region starts at the strength the row already had.
        assert row.locator(".lb-region-editor input[type=number]").input_value() == "0.37"

    def test_the_drawn_region_is_what_wangp_is_given(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_01.safetensors"]')
        phase_two = native(page).split()[0].split(";")[1]

        strength = row.locator(".lb-region-editor input[type=number]")
        strength.fill("0.9")
        strength.press("Enter")
        settle(page, 1500)

        # One value per step, the drawn slots at the region's strength and
        # every slot no region covers at zero -- exactly the picture on screen.
        token = native(page).split()[0]
        phase_one = token.split(";")[0].split(",")
        assert len(phase_one) == STEPS
        assert phase_one[7:14] == ["0.9"] * 7
        assert set(phase_one[:7] + phase_one[14:]) == {"0"}
        # The phase that was not being edited is untouched.
        assert token.split(";")[1] == phase_two

    def test_a_tap_on_empty_space_draws_a_region(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_01.safetensors"]')
        timeline = row.locator(".lb-timeline")
        box = timeline.bounding_box()
        slot = box["width"] / float(STEPS)
        page.mouse.click(box["x"] + slot * 20.5, box["y"] + box["height"] / 2)
        settle(page, 1200)
        assert row.locator(".lb-region").count() == 2
        assert row.locator('[data-lb="region-range"]').inner_text().startswith("Slots 21")

        row.locator("button.lb-schedule-toggle").click()   # tidy up for later tests
        settle(page, 400)

    def test_a_gradio_remount_rebuilds_the_panel(self, panel):
        """Gradio can wipe the HTML host; the panel must come back by itself."""
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        if row.locator(".lb-region").count() == 0:
            draw_region(page, row)
        regions = row.locator(".lb-region").count()
        before = native(page)
        assert regions > 0

        page.evaluate("""() => {
          const root = document.querySelector('[id^=lb_root_]');
          root.innerHTML = '';   // what Gradio does when it re-renders the tab
        }""")
        page.wait_for_timeout(1600)   # the integrity check runs on a timer

        assert page.locator(".lb-tile").count() > 0
        assert page.locator(".lb-row").count() == 3
        assert native(page) == before
        # View state survives with it, because it never left the frontend.
        assert page.locator('.lb-row[data-id="lora_02.safetensors"] .lb-region').count() == regions

    def test_the_timeline_follows_the_step_counter(self, panel):
        """The reported issue: change the steps in WanGP, live, and the
        schedule has to come with it."""
        page, host, _ = panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')

        def set_steps(count):
            """Move the counter, then push a payload as WanGP's change does."""
            page.evaluate("(n) => fetch('/steps?n=' + n)", count)
            settle(page, 250)
            page.evaluate("""() => fetch('/payload').then(r => r.json()).then(d => {
                window.__native = d.multipliers; window.__push(d.payload); })""")
            settle(page, 1800)

        def phase_one():
            return native(page).split()[1].split(";")[0].split(",")

        try:
            # A schedule drawn against a four-step run, as the user described.
            set_steps(4)
            draw_region(page, row, at=0.1)
            drawn = schedule_of(page, "lora_02.safetensors")
            assert drawn["slots"] == 4
            assert drawn["regions"]
            at_four = phase_one()
            assert at_four == expected_values(drawn)
            assert row.locator(".lb-tick").last.inner_text() == "4"

            # Longer: what was there is inherited, the new step arrives undefined.
            set_steps(5)
            grown = schedule_of(page, "lora_02.safetensors")
            assert grown["slots"] == 5
            assert phase_one() == at_four + ["0"]
            assert grown["regions"] == drawn["regions"]

            # Shorter: the steps that are gone are dropped, the rest untouched.
            set_steps(3)
            assert schedule_of(page, "lora_02.safetensors")["slots"] == 3
            assert phase_one() == at_four[:3]
        finally:
            set_steps(STEPS)

    def test_leaving_one_phase_resets_a_scheduled_lora(self, panel):
        """No single strength carries a schedule over, so it switches off."""
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_01.safetensors"]')
        row.locator("button.lb-schedule-toggle").click()
        settle(page, 800)
        timeline = row.locator(".lb-timeline")
        timeline.scroll_into_view_if_needed()
        settle(page, 200)
        box = timeline.bounding_box()
        slots = schedule_of(page, "lora_01.safetensors")["slots"]
        page.mouse.click(box["x"] + box["width"] * 0.3, box["y"] + box["height"] / 2)
        settle(page, 1200)
        assert row.locator(".lb-region").count() >= 1
        assert "," in native(page).split()[0]

        try:
            set_guidance(page, 2)
            settle(page, 900)
            assert native(page).split()[0] == "0;0"
            assert row.locator("button.lb-schedule-toggle").count() == 0
            assert row.locator(".lb-timeline").count() == 0
        finally:
            set_guidance(page, 1)

    def test_the_panel_fits_a_phone(self, panel):
        page, _, _ = panel
        page.set_viewport_size({"width": 400, "height": 900})
        settle(page, 400)
        try:
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
        finally:
            page.set_viewport_size({"width": 1100, "height": 1000})
            settle(page, 300)

    def test_nothing_raised_in_the_browser(self, panel):
        page, _, problems = panel
        assert problems == []
