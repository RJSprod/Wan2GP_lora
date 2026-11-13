"""The panel driven in a real browser, against the real Python bridge.

Everything below the DOM is already covered by the other suites.  What only a
browser can answer is whether the rules survive contact with layout and pointer
events: that the grid really stops at three rows, that a tile's Inspector button
does not toggle the LoRA underneath it, that an imported 0.71 is still 0.71
after rendering, and that setting a phase's steps ends up as the multiplier
WanGP will actually be given.

Skipped unless Playwright and a Chromium build are both present, the same way
the thumbnail tests skip without Pillow.
"""

import json
import os
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import unquote

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
  // The media bridge as Gradio runs it: the request reaches the plugin's real
  // handler, and the reply only reaches the panel when its text differs from
  // the last one -- a Textbox raises no change event for an identical value.
  var mediaReq = document.querySelector("#__THUMB_REQ_ID__ textarea");
  var mediaRes = document.querySelector("#__THUMB_RES_ID__ textarea");
  document.querySelector("#__THUMB_BTN__ button").addEventListener("click", function () {
    fetch("/media", { method: "POST", body: mediaReq.value })
      .then(function (response) { return response.text(); })
      .then(function (text) {
        if (text === mediaRes.value) { return; }
        mediaRes.value = text;
        var api = window.wgpLoraBrowser && window.wgpLoraBrowser["__INST__"];
        if (api) { api.applyMedia(text); }
      });
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
    """The plugin plus the native state a browser session mutates.

    The active LoRAs start in Cards, where every row's editor is open, which
    is what most of these tests drive; the list view's tests ask for "list".
    """

    def __init__(self, tmp_path, view="cards", state=None):
        module = bridge._load_plugin_module()
        self.plugin = module.LoraBrowserPlugin()
        self.plugin.server_config_filename = str(tmp_path / "wgp_config.json")
        self.plugin.get_state_model_type = lambda state: state.get("model_type", "")
        self.plugin.get_model_def = lambda model_type: MODEL_DEF
        self.plugin.get_lora_dir = lambda model_type: str(tmp_path / "loras")
        self.plugin._init_stores()
        self.plugin._metadata.set_stack_view(view)
        self.state = state or STATE
        self.selected = list(START["selected"])
        self.multipliers = START["multipliers"]
        self.steps = STEPS
        #: Guidance phases, as WanGP's own dropdown would report them. Step
        #: schedules exist only at 1, so that is where this starts; the tests
        #: about more phases switch and switch back.
        self.guidance = 1

    def payload(self):
        return self.plugin._build_payload(
            self.plugin._instance(INSTANCE), self.state, self.selected, self.multipliers,
            self.guidance, self.steps,
        )

    def act(self, action_json):
        choices, mults, text = self.plugin._apply_action(
            self.plugin._instance(INSTANCE), action_json, self.state, self.selected,
            self.multipliers, self.guidance, self.steps,
        )
        if isinstance(choices, dict) and "value" in choices:
            self.selected = list(choices["value"])
        if isinstance(mults, dict) and "value" in mults:
            self.multipliers = mults["value"]
        return text

    def media(self, request_json):
        return self.plugin._serve_media(self.plugin._instance(INSTANCE), request_json, self.state)


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
            if self.path.startswith("/multipliers"):
                host.multipliers = unquote(self.path.split("v=", 1)[-1])
                self._send(json.dumps({"multipliers": host.multipliers}))
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
            if self.path == "/media":
                self._send(host.media(body))
                return
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
def chromium():
    """One browser for the module: Playwright's sync API allows a single
    session per thread, so every panel opens a page in this one."""
    executable = _chromium()
    if not executable:
        pytest.skip("no Chromium build available for Playwright")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=executable, args=["--no-sandbox"])
        yield browser
        browser.close()


@contextmanager
def _open_panel(browser, host):
    """A page on the harness, serving ``host``."""
    server, url = _serve(host)
    page = browser.new_page(viewport={"width": 1100, "height": 1000})
    problems = []
    page.on("console", lambda msg: problems.append(msg.text)
            if msg.type == "error" and "favicon" not in msg.text and "404" not in msg.text
            else None)
    page.on("pageerror", lambda error: problems.append(str(error)))
    page.on("dialog", lambda dialog: dialog.accept())
    try:
        page.goto(url)
        page.wait_for_selector(".lb-row, .lb-li", timeout=15000)
        page.wait_for_timeout(400)
        yield page, host, problems
    finally:
        page.close()
        server.shutdown()


@pytest.fixture(scope="module")
def panel(chromium, tmp_path_factory):
    with _open_panel(chromium, _Host(tmp_path_factory.mktemp("browser"))) as opened:
        yield opened


#: LTX-2.3 Distilled as WanGP's get_model_def describes it: two phases at most,
#: and they are the pipeline's stages.
LTX_MODEL_DEF = {
    "architecture": "ltx2_22B", "ltx2_pipeline": "distilled", "guidance_max_phases": 2,
    "visible_phases": 0, "lock_inference_steps": True, "ltx2_22B_class": True,
}


#: MiniMax H3 as its handler defines it: its phase 2 is a pass of its own, so
#: both phases are stages and phase 1 is the same full run in either mode.
H3_MODEL_DEF = {
    "guidance_max_phases": 2, "lora_multiplier_phases": 2, "phase_2_spatial_tiling": True,
    "inference_steps2": True,
}


@pytest.fixture(scope="module")
def h3_panel(chromium, tmp_path_factory):
    """The reported setup: H3 in One Phase, a four-step run, phase 1 scheduled."""
    host = _Host(tmp_path_factory.mktemp("browser-h3"))
    host.plugin.get_model_def = lambda model_type: H3_MODEL_DEF
    host.steps, host.guidance = 4, 1
    host.selected = ["lora_01.safetensors", "lora_02.safetensors"]
    host.multipliers = "0,-2,0,0;1 0.7;0.5"
    with _open_panel(chromium, host) as opened:
        yield opened


@pytest.fixture(scope="module")
def off_panel(chromium, tmp_path_factory):
    """Three LoRAs, the middle one scheduled, for switching them off and on."""
    host = _Host(tmp_path_factory.mktemp("browser-off"))
    host.steps = 4
    host.selected = ["lora_01.safetensors", "lora_02.safetensors", "lora_03.safetensors"]
    host.multipliers = "0.7;0.7 1,1,0,0;0.5 0.4;0.4"
    with _open_panel(chromium, host) as opened:
        yield opened


@pytest.fixture(scope="module")
def ltx_panel(chromium, tmp_path_factory):
    """The same harness on LTX-2.3 Distilled, in WanGP's default Two Phases."""
    host = _Host(tmp_path_factory.mktemp("browser-ltx"))
    host.plugin.get_model_def = lambda model_type: LTX_MODEL_DEF
    host.steps, host.guidance = 8, 2
    host.multipliers = "1;1 1;1 0.5:0.9"
    with _open_panel(chromium, host) as opened:
        yield opened


def native(page):
    return page.evaluate("window.__native")


def settle(page, ms=500):
    page.wait_for_timeout(ms)


def schedule_of(page, lora_id, phase=0):
    """The schedule the panel is currently showing, as the payload gave it."""
    return page.evaluate(
        """(args) => window.wgpLoraBrowser[args.instance].state.rows
             .find(r => r.id === args.id).phase_schedules[args.phase]""",
        {"instance": INSTANCE, "id": lora_id, "phase": phase},
    )


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


def set_steps(page, count):
    """Move WanGP's step counter, then push a payload as its change event does."""
    page.evaluate("(n) => fetch('/steps?n=' + n)", count)
    settle(page, 250)
    repaint(page)


def set_token(page, position, token):
    """Write one LoRA's multiplier the way a preset or a hand edit would."""
    tokens = native(page).split()
    tokens[position] = token
    page.evaluate(
        "(v) => fetch('/multipliers?v=' + encodeURIComponent(v))", " ".join(tokens)
    )
    settle(page, 250)
    repaint(page)


def bars(row):
    return row.locator(".lb-bars .lb-bar")


def bar_values(row):
    """What each step's bar says it is worth, as WanGP's token spells it."""
    return row.locator(".lb-bars .lb-bar").evaluate_all("bars => bars.map(bar => bar.dataset.value)")


def picked(row):
    return row.locator(".lb-bars .lb-bar-on").evaluate_all("bars => bars.map(bar => bar.dataset.step)")


def per_step(row):
    """Whether the phase the row shows runs per step."""
    return row.locator("button.lb-kind-steps").first.get_attribute("aria-checked") == "true"


def set_kind(page, row, steps):
    """One strength or Per step, for the phase the row shows."""
    if per_step(row) != steps:
        row.locator("button.lb-kind-steps" if steps else "button.lb-kind-flat").first.click()
        settle(page, 900)


def pick(page, row, *steps):
    """Select exactly ``steps`` (1-based), a tap on each bar."""
    row.locator("button.lb-pick-none").click()
    settle(page, 150)
    for step in steps:
        row.locator(f'.lb-bar[data-step="{step}"]').click()
        settle(page, 150)


def type_value(page, row, value, ms=1300):
    """Type a value for the selected steps, as a person would."""
    number = row.locator(".lb-weight-steps input.lb-number")
    number.fill(str(value))
    number.press("Enter")
    settle(page, ms)


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

    def test_reopening_the_inspector_is_answered_at_once(self, panel):
        """A LoRA with no catalogue gets the same Inspect reply every time, and
        preview-less thumbnail batches all reply alike. Gradio drops a reply
        whose text equals the last one, so without the echoed sequence number
        the view sat on "Loading..." until the 30 s timeout, then claimed the
        LoRA could not be read."""
        page, _, _ = panel
        inspect = page.locator(".lb-tile").first.locator("button.lb-tile-inspect")
        for _ in range(2):
            inspect.click()
            page.wait_for_selector(".lb-modal .lb-fetch-btn", timeout=5000)
            assert page.locator(".lb-modal-title").inner_text() != "Loading..."
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

    def test_each_phase_is_one_strength_or_per_step(self, panel):
        page, _, _ = panel
        assert page.evaluate(
            "() => window.wgpLoraBrowser['%s'].state.schedulingEnabled" % INSTANCE
        ) is True
        plain = page.locator('.lb-row[data-id="lora_01.safetensors"]')
        stepped = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        for row, steps in ((plain, False), (stepped, True)):
            seg = row.locator(".lb-kind-seg")
            assert seg.get_attribute("role") == "radiogroup"
            labels = [text.replace("∿", "").strip() for text in seg.locator("button").all_inner_texts()]
            assert labels == ["One strength", "Per step"]
            assert per_step(row) is steps
        # One or the other: a phase run per step has no one-strength scrubber,
        # and a phase at one strength has no bars.
        assert plain.locator(".lb-bars").count() == 0
        assert plain.locator(".lb-weight-main input[type=range]").count() == 1
        assert stepped.locator(".lb-weight-main").count() == 0
        assert stepped.locator("input[type=range]").count() == 1
        # Scheduling needs One Phase, so there are no phase chips to pick from.
        assert stepped.locator(".lb-phase-strip").count() == 0

    def test_a_per_step_phase_shows_a_bar_for_every_step(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        before = native(page)
        assert schedule_of(page, "lora_02.safetensors")["slots"] == STEPS
        assert bars(row).count() == STEPS
        # Each bar says what WanGP is given for its step.
        assert bar_values(row) == native(page).split()[1].split(";")[0].split(",")
        # A long run numbers every fifth step, and its ends.
        numbered = row.locator(".lb-bar:not(.lb-bar-quiet) .lb-bar-n").all_inner_texts()
        assert numbered == ["1", "5", "10", "15", "20", "25", "30"]
        # Looking changes nothing; nothing is selected, so nothing can be set.
        assert native(page) == before
        assert picked(row) == []
        assert row.locator(".lb-weight-steps input[type=range]").is_disabled()

    def test_tapping_bars_selects_steps_and_one_value_sets_them_all(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        token = native(page).split()[1]
        before, hidden = token.split(";")[0].split(","), token.split(";")[1]

        pick(page, row, 3, 4)
        assert picked(row) == ["3", "4"]
        assert row.locator(".lb-steps-picked").inner_text() == "2 steps selected."
        # A second tap unselects.
        row.locator('.lb-bar[data-step="4"]').click()
        settle(page, 200)
        assert row.locator(".lb-steps-picked").inner_text() == "Step 3 selected."
        row.locator('.lb-bar[data-step="4"]').click()
        settle(page, 200)

        type_value(page, row, 0.33)
        token = native(page).split()[1]
        after = token.split(";")[0].split(",")
        assert after[2:4] == ["0.33", "0.33"]
        assert after[:2] + after[4:] == before[:2] + before[4:]
        # The phase this guidance mode hides is untouched.
        assert token.split(";")[1] == hidden
        assert bar_values(row)[2:4] == ["0.33", "0.33"]

    def test_dragging_across_bars_selects_each_one_passed(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        row.locator("button.lb-pick-none").click()
        settle(page, 200)
        row.locator(".lb-bars").scroll_into_view_if_needed()
        first = row.locator('.lb-bar[data-step="5"]').bounding_box()
        last = row.locator('.lb-bar[data-step="9"]').bounding_box()
        page.mouse.move(first["x"] + first["width"] / 2, first["y"] + first["height"] / 2)
        page.mouse.down()
        page.mouse.move(last["x"] + last["width"] / 2, last["y"] + last["height"] / 2, steps=12)
        # Each bar passed lights up under the pointer, before anything is sent.
        assert row.locator(".lb-bar-on").count() == 5
        page.mouse.up()
        settle(page, 300)
        assert picked(row) == ["5", "6", "7", "8", "9"]
        # The steps disagree, so the field says so rather than show one of them.
        number = row.locator(".lb-weight-steps input.lb-number")
        assert number.input_value() == ""
        assert number.get_attribute("placeholder") == "mixed"

    def test_minus_and_plus_move_each_step_from_its_own_value(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        assert picked(row) == ["5", "6", "7", "8", "9"]
        before = native(page).split()[1].split(";")[0].split(",")
        row.locator(".lb-weight-steps button.lb-down").click()
        settle(page, 1300)
        after = native(page).split()[1].split(";")[0].split(",")
        assert after[4:9] == ["%g" % round(float(value) - 0.01, 4) for value in before[4:9]]
        assert after[:4] + after[9:] == before[:4] + before[9:]

    def test_a_step_takes_a_value_the_slider_cannot_show(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        pick(page, row, 1)
        type_value(page, row, -2)
        assert native(page).split()[1].split(",")[0] == "-2"
        bar = row.locator('.lb-bar[data-step="1"]')
        # Under the zero line, and capped: it is past -1.
        assert "lb-bar-under" in bar.get_attribute("class")
        assert bar.locator(".lb-bar-value").inner_text() == "−2"
        assert bar.locator(".lb-bar-down i").evaluate("i => i.getBoundingClientRect().height") > 10
        # The slider stays 0..1; the number holds the truth and says so.
        assert row.locator(".lb-weight-steps input[type=range]").input_value() == "0"
        note = row.locator(".lb-range-note")
        assert note.is_visible() and "−2" in note.inner_text()

        type_value(page, row, 1.5)
        assert "lb-bar-over" in bar.get_attribute("class")
        type_value(page, row, 1)
        assert not row.locator(".lb-range-note").is_visible()
        assert "lb-bar-odd" not in bar.get_attribute("class")

    def test_one_strength_puts_the_steps_aside_and_per_step_brings_them_back(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        stepped = native(page).split()[1]
        set_kind(page, row, False)
        token = native(page).split()[1]
        assert "," not in token.split(";")[0]
        assert token.split(";")[1] == stepped.split(";")[1]
        assert row.locator(".lb-bars").count() == 0
        assert row.locator(".lb-weight-main input[type=range]").count() == 1
        assert "brings back" in row.locator(".lb-kind-seg").get_attribute("title")

        set_kind(page, row, True)
        assert native(page).split()[1] == stepped
        assert bars(row).count() == STEPS

    def test_the_bars_never_scroll(self, panel):
        """Every step is in view at whatever width the panel is given."""
        page, _, _ = panel
        for width in (1100, 700, 400):
            page.set_viewport_size({"width": width, "height": 1000})
            settle(page, 400)
            sizes = page.evaluate("""() => {
              const strip = document.querySelector('.lb-row[data-id="lora_02.safetensors"] .lb-bars');
              return {scroll: strip.scrollWidth, client: strip.clientWidth,
                      last: strip.lastElementChild.getBoundingClientRect().right,
                      edge: strip.getBoundingClientRect().right};
            }""")
            assert sizes["scroll"] <= sizes["client"], (width, sizes)
            assert sizes["last"] <= sizes["edge"] + 0.5, (width, sizes)
        page.set_viewport_size({"width": 1100, "height": 1000})
        settle(page, 400)

    def test_full_screen_is_the_same_editor_with_room(self, panel):
        page, _, _ = panel
        # The panel's row, not the sheet's copy of it.
        row = page.locator('.lb-rows .lb-row[data-id="lora_02.safetensors"]')
        before = native(page)
        inline = row.locator(".lb-bars").bounding_box()

        try:
            row.locator(".lb-row-top button.lb-full").click()
            settle(page, 600)
            sheet = page.locator(".lb-sheet")
            assert sheet.count() == 1
            assert sheet.get_attribute("aria-modal") == "true"
            assert native(page) == before        # opening it is a view change
            # The panel keeps the LoRA's place, and says where it went.
            assert row.locator(".lb-full-note").count() == 1
            assert row.locator(".lb-bars").count() == 0

            # As wide as the row's, and taller.
            box = sheet.locator(".lb-bars").bounding_box()
            assert box["width"] >= inline["width"] - 8
            assert box["height"] > inline["height"]
            # The way back is in its header; the tools are the rest.
            assert sheet.locator(".lb-sheet-head button.lb-full").inner_text().strip() == "Exit full screen"
            assert sheet.locator(".lb-toolbar-row button").all_inner_texts() == ["Inspect", "Remove"]

            # It is live: an edit here reaches WanGP like any other.
            sheet.locator("button.lb-pick-none").click()
            settle(page, 200)
            sheet.locator('.lb-bar[data-step="2"]').click()
            settle(page, 200)
            number = sheet.locator(".lb-weight-steps input.lb-number")
            number.fill("0.44")
            number.press("Enter")
            settle(page, 1500)
            assert native(page).split()[1].split(",")[1] == "0.44"
        finally:
            # A left-open sheet would swallow every later test's clicks.
            page.keyboard.press("Escape")
            settle(page, 400)

        assert page.locator(".lb-sheet").count() == 0
        # Back in the panel, with the edit, and the same step selected.
        assert bars(row).count() == STEPS
        assert bar_values(row)[1] == "0.44"
        assert picked(row) == ["2"]

    def test_an_advanced_token_is_preserved_and_explained(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_03.safetensors"]')
        assert row.locator("code.lb-advanced").inner_text() == "0.5:0.9"
        assert "branch" in row.locator(".lb-preserved-note").inner_text().lower()
        assert row.locator(".lb-kind-seg").count() == 0
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

    def test_per_step_starts_with_every_step_at_the_strength_it_had(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_01.safetensors"]')
        select_phase(page, row, 0)
        phase_two = native(page).split()[0].split(";")[1]
        set_kind(page, row, True)
        assert schedule_of(page, "lora_01.safetensors")["slots"] == STEPS
        assert bars(row).count() == STEPS
        # One value per step, every one what the phase was worth: nothing
        # WanGP applies has changed.
        token = native(page).split()[0]
        assert token.split(";")[0].split(",") == ["0.37"] * STEPS
        assert token.split(";")[1] == phase_two
        assert set(bar_values(row)) == {"0.37"}
        assert "goes back to 0.37" in row.locator(".lb-kind-seg").get_attribute("title")

    def test_the_keyboard_selects_one_step_at_a_time(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_01.safetensors"]')
        row.locator("button.lb-pick-none").click()
        settle(page, 200)
        row.locator('.lb-bar[data-step="8"]').focus()
        page.keyboard.press("Space")
        settle(page, 300)
        assert picked(row) == ["8"]
        # The rebuilt bar has the focus back, so the keyboard can go on.
        assert page.evaluate("document.activeElement.dataset.step") == "8"
        page.keyboard.press("Tab")
        page.keyboard.press("Enter")
        settle(page, 300)
        assert picked(row) == ["8", "9"]
        assert row.locator('.lb-bar[data-step="9"]').get_attribute("aria-pressed") == "true"

        row.locator("button.lb-pick-all").click()
        settle(page, 200)
        assert len(picked(row)) == STEPS
        assert row.locator(".lb-steps-picked").inner_text() == f"{STEPS} steps selected."
        row.locator("button.lb-pick-none").click()
        settle(page, 200)
        assert picked(row) == []

    def test_a_gradio_remount_rebuilds_the_panel(self, panel):
        """Gradio can wipe the HTML host; the panel must come back by itself."""
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        pick(page, row, 6)
        count = bars(row).count()
        before = native(page)
        assert count > 0

        page.evaluate("""() => {
          const root = document.querySelector('[id^=lb_root_]');
          root.innerHTML = '';   // what Gradio does when it re-renders the tab
        }""")
        page.wait_for_timeout(1600)   # the integrity check runs on a timer

        assert page.locator(".lb-tile").count() > 0
        assert page.locator(".lb-row").count() == 3
        assert native(page) == before
        # View state survives with it, because it never left the frontend.
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        assert bars(row).count() == count
        assert picked(row) == ["6"]

    def test_the_steps_follow_the_step_counter(self, panel):
        """The reported issue: change the steps in WanGP, live, and the
        schedule has to come with it -- with nothing on it moving."""
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')

        def phase_one():
            return native(page).split()[1].split(";")[0].split(",")

        try:
            # A value per step of a four-step run, the last step different so
            # a stretch or a lost step would show.
            set_steps(page, 4)
            set_token(page, 1, "0.5,0.5,0.3,0.8;0.7")
            drawn = schedule_of(page, "lora_02.safetensors")
            assert drawn["slots"] == 4
            assert phase_one() == ["0.5", "0.5", "0.3", "0.8"]
            assert bar_values(row) == phase_one()

            # Longer: everything stays where it was, the new step arrives empty.
            set_steps(page, 5)
            grown = schedule_of(page, "lora_02.safetensors")
            assert grown["slots"] == 5
            assert phase_one() == ["0.5", "0.5", "0.3", "0.8", "0"]
            assert bar_values(row) == phase_one()
            assert grown["regions"] == drawn["regions"]
            assert row.locator(".lb-kept").count() == 0

            # Shorter: the run stops sooner, and the step it no longer reaches
            # is kept -- and the steps say so.
            set_steps(page, 3)
            assert schedule_of(page, "lora_02.safetensors")["slots"] == 3
            assert phase_one() == ["0.5", "0.5", "0.3"]
            assert bars(row).count() == 3
            kept = row.locator(".lb-steps .lb-kept")
            assert kept.count() == 1
            assert kept.inner_text() == "+1 step kept past the end of the run"

            # Back up: exactly what was there at four steps.
            set_steps(page, 4)
            assert phase_one() == ["0.5", "0.5", "0.3", "0.8"]
            assert schedule_of(page, "lora_02.safetensors")["regions"] == drawn["regions"]
            assert row.locator(".lb-kept").count() == 0
        finally:
            set_steps(page, STEPS)

    def test_leaving_one_phase_keeps_a_schedule_it_cannot_show(self, panel):
        """This model's Two Phases cannot show a schedule truthfully, so it
        shows what the schedule does instead -- and changes nothing."""
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_01.safetensors"]')
        assert per_step(row)
        pick(page, row, 10, 11, 12)
        type_value(page, row, 0.9)
        drawn = native(page).split()[0]
        assert drawn.split(";")[0].split(",")[9:12] == ["0.9"] * 3

        try:
            set_guidance(page, 2)
            settle(page, 900)
            assert native(page).split()[0] == drawn
            assert row.locator(".lb-kind-seg").count() == 0
            assert row.locator(".lb-bars").count() == 0
            assert "∿" in row.locator(".lb-kept-summary").inner_text()
            assert "One Phase" in row.locator(".lb-preserved-note").inner_text()
        finally:
            set_guidance(page, 1)
        assert native(page).split()[0] == drawn
        assert per_step(row)
        assert bars(row).count() == STEPS

    def test_the_panel_fits_a_phone(self, panel):
        page, _, _ = panel
        page.set_viewport_size({"width": 400, "height": 900})
        settle(page, 400)
        try:
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
        finally:
            page.set_viewport_size({"width": 1100, "height": 1000})
            settle(page, 300)

    def test_full_screen_fits_a_phone(self, panel):
        page, _, _ = panel
        page.set_viewport_size({"width": 400, "height": 900})
        settle(page, 400)
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        try:
            row.locator(".lb-row-top button.lb-full").click()
            settle(page, 600)
            sheet = page.locator(".lb-sheet")
            assert sheet.bounding_box()["width"] <= 400.5
            assert page.evaluate("""() => {
              const body = document.querySelector('.lb-sheet-body');
              return body.scrollWidth <= body.clientWidth + 1;
            }""")
            # A narrow sheet gives the slider a line of its own, with the
            # number and -/+ under it.
            weight = sheet.locator(".lb-weight-steps")
            slider = weight.locator("input[type=range]").bounding_box()
            number = weight.locator("input.lb-number").bounding_box()
            assert number["y"] >= slider["y"] + slider["height"] - 1
            assert number["height"] >= 44
            # Exit full screen puts it back in the panel.
            sheet.locator(".lb-sheet-head button.lb-full").click()
            settle(page, 500)
            assert page.locator(".lb-sheet").count() == 0
            assert row.locator(".lb-bars").count() == 1
        finally:
            page.keyboard.press("Escape")
            page.set_viewport_size({"width": 1100, "height": 1000})
            settle(page, 300)

    def test_nothing_raised_in_the_browser(self, panel):
        page, _, problems = panel
        assert problems == []


class TestLtxInTheBrowser:
    """LTX-2 in the real frontend: its phases are stages, each its own steps."""

    def test_each_stage_runs_per_step_over_its_own_length(self, ltx_panel):
        page, _, problems = ltx_panel
        row = page.locator('.lb-row[data-id="lora_01.safetensors"]')
        assert row.locator(".lb-phase-strip .lb-chip").count() >= 2

        select_phase(page, row, 0)
        set_kind(page, row, True)
        first = schedule_of(page, "lora_01.safetensors", 0)
        assert (first["stage"], first["stage_steps"]) == (1, 8)
        assert bars(row).count() == 8
        assert native(page).split()[0] == "1,1,1,1,1,1,1,1;1"
        set_kind(page, row, False)
        assert native(page).split()[0] == "1;1"

        select_phase(page, row, 1)
        set_kind(page, row, True)
        second = schedule_of(page, "lora_01.safetensors", 1)
        assert (second["stage"], second["stage_steps"]) == (2, 3)
        assert bars(row).count() == 3

        # The refinement's first two steps off, in the token's second part.
        pick(page, row, 1, 2)
        type_value(page, row, 0)
        assert native(page).split()[0] == "1;0,0,1"
        assert problems == []

    def test_one_phase_and_back_starts_the_refinement_empty(self, ltx_panel):
        page, _, problems = ltx_panel
        set_token(page, 1, "1,1,1,1,0,0,0,0;0,0.5,1")
        try:
            set_guidance(page, 1)
            settle(page, 900)
            # The only token WanGP accepts in One Phase: stage 1's schedule.
            assert native(page).split()[1] == "1,1,1,1,0,0,0,0"
            schedule = schedule_of(page, "lora_02.safetensors")
            assert (schedule["slots"], schedule["coordinate_mode"]) == (8, "global_exact")
        finally:
            set_guidance(page, 2)
            settle(page, 900)
        # Stage 1 kept, stage 2 empty -- not the old refinement schedule, and
        # not stage 1's list squeezed into three steps.
        assert native(page).split()[1] == "1,1,1,1,0,0,0,0;0"

        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        assert row.locator(".lb-chip-static").count() == 0     # no shared schedule
        select_phase(page, row, 1)
        # Empty: one strength, and it is 0.
        assert not per_step(row)
        assert row.locator(".lb-weight-main input[type=number]").input_value() == "0"

        # Per step starts every step there; the last one on, in the token's
        # second part.
        set_kind(page, row, True)
        assert schedule_of(page, "lora_02.safetensors", 1)["stage_steps"] == 3
        assert bar_values(row) == ["0", "0", "0"]
        pick(page, row, 3)
        type_value(page, row, 1)
        assert native(page).split()[1] == "1,1,1,1,0,0,0,0;0,0,1"
        assert problems == []

    def test_switching_to_two_phases_carries_the_last_steps_into_stage_2(self, ltx_panel):
        page, _, problems = ltx_panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        try:
            set_guidance(page, 1)
            settle(page, 900)
            # On for the second half of the run, so on at steps 6-8.
            set_token(page, 1, "0,0,0,0,1,1,1,1")
            assert native(page).split()[1] == "0,0,0,0,1,1,1,1"
        finally:
            set_guidance(page, 2)
            settle(page, 1200)
        assert native(page).split()[1] == "0,0,0,0,1,1,1,1;1,1,1"

        select_phase(page, row, 1)
        second = schedule_of(page, "lora_02.safetensors", 1)
        assert second["stage_steps"] == 3
        assert second["values"] == [1, 1, 1]
        assert per_step(row)
        assert bar_values(row) == ["1", "1", "1"]
        assert row.locator("button.lb-fill-stage-2").count() == 0
        assert problems == []

    def test_fill_from_stage_1_runs_the_refinement_per_step_once(self, ltx_panel):
        page, _, problems = ltx_panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')

        # Off where stage 2's noise levels are: the button says why, and is off.
        set_token(page, 1, "1,1,1,1,0,0,0,0;0")
        select_phase(page, row, 1)
        assert not per_step(row)
        button = row.locator("button.lb-fill-stage-2")
        assert button.count() == 1 and button.is_disabled()
        assert "nothing to carry" in button.get_attribute("title")

        set_token(page, 1, "1,1,1,1,0.8,0.6,0.4,0.2;0")
        assert button.is_enabled()
        assert button.get_attribute("title").endswith("0.6, 0.4, 0.2")
        button.click()
        settle(page, 1300)
        assert native(page).split()[1] == "1,1,1,1,0.8,0.6,0.4,0.2;0.6,0.4,0.2"
        # Filled once: stage 2 runs per step now, with nothing left to fill.
        assert per_step(row)
        assert bar_values(row) == ["0.6", "0.4", "0.2"]
        assert row.locator("button.lb-fill-stage-2").count() == 0
        assert problems == []


class TestH3InTheBrowser:
    """MiniMax H3: phase 1 is the same full run in One Phase and Two, so a
    switch between them leaves it -- steps and all -- exactly as it was."""

    def test_two_phases_keep_phase_1_and_its_steps(self, h3_panel):
        page, _, problems = h3_panel
        token = "0,-2,0,0;1 0.7;0.5"
        row = page.locator('.lb-row[data-id="lora_01.safetensors"]')
        assert per_step(row)
        assert bar_values(row) == ["0", "-2", "0", "0"]
        assert "lb-bar-under" in row.locator('.lb-bar[data-step="2"]').get_attribute("class")
        try:
            set_guidance(page, 2)
            settle(page, 900)
            assert native(page) == token
            assert row.locator(".lb-phase-strip .lb-chip").count() >= 2
            select_phase(page, row, 0)
            first = schedule_of(page, "lora_01.safetensors", 0)
            assert first["slots"] == 4
            assert first["values"] == [0, -2, 0, 0]
            assert bar_values(row) == ["0", "-2", "0", "0"]
            # Phase 2 is the enhancement pass, at one strength: three steps long.
            select_phase(page, row, 1)
            assert not per_step(row)
            assert page.evaluate(
                "() => window.wgpLoraBrowser['%s'].state.phaseSteps" % INSTANCE)[1] == 3
            assert native(page) == token
        finally:
            set_guidance(page, 1)
            settle(page, 900)
        assert native(page) == token
        assert schedule_of(page, "lora_01.safetensors", 0)["values"][1] == -2
        assert problems == []


class TestSwitchingOffInTheBrowser:
    """Off keeps a LoRA where it was, with everything it had, and applies none
    of it: it leaves WanGP's selection for a comment line WanGP never reads."""

    def test_a_switched_off_lora_keeps_its_place_and_its_settings(self, off_panel):
        page, _, problems = off_panel
        page.locator('.lb-row[data-id="lora_02.safetensors"] button.lb-switch').click()
        settle(page, 1200)
        assert native(page) == "0.7;0.7 0.4;0.4\n#lora_off 1 1,1,0,0;0.5 lora_02.safetensors"

        rows = page.evaluate("() => [...document.querySelectorAll('.lb-row')].map(r => r.dataset.id)")
        assert rows == ["lora_01.safetensors", "lora_02.safetensors", "lora_03.safetensors"]
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        assert "lb-row-off" in row.get_attribute("class")
        assert row.locator("button.lb-switch").get_attribute("aria-checked") == "false"
        # Nothing to edit while it is off -- only the way back, and out.
        assert row.locator("input, .lb-kind-seg, .lb-bars").count() == 0
        assert "\u223f 1 at steps 1\u20132" in row.locator(".lb-off-summary").inner_text()
        assert "1 off" in page.locator('[data-lb="active-title"]').inner_text()
        tile = page.locator('.lb-tile[data-id="lora_02.safetensors"]')
        assert tile.locator(".lb-check.lb-check-off").count() == 1

        # Its tile brings it back, exactly as it was.
        tile.click()
        settle(page, 1200)
        assert native(page) == "0.7;0.7 1,1,0,0;0.5 0.4;0.4"
        assert page.locator(".lb-row.lb-row-off").count() == 0
        assert problems == []

    def test_disable_all_and_enable_all(self, off_panel):
        page, _, problems = off_panel
        before = native(page)
        page.locator('[data-lb="disable-all"]').click()
        settle(page, 1200)
        assert native(page).count("#lora_off") == 3
        assert page.locator(".lb-row.lb-row-off").count() == 3
        assert page.locator('[data-lb="disable-all"]').is_disabled()

        page.locator('[data-lb="enable-all"]').click()
        settle(page, 1200)
        assert native(page) == before
        assert page.locator(".lb-row.lb-row-off").count() == 0
        assert page.locator('[data-lb="enable-all"]').is_disabled()
        assert problems == []


#: Three LoRAs as text, as two-phase LTX takes them: one scheduled in stage 1,
#: one in stage 2, one plain.
PASTED = "lora_05: x0,0,0,0,0.325,0.325,0,0;0\nlora_06: x0.8;0.375,0.25,0.25\nlora_07: x1;0.25"


@pytest.fixture(scope="module")
def paste_panel(chromium, tmp_path_factory):
    """LTX-2.3 Distilled in Two Phases, one LoRA on and one switched off."""
    host = _Host(tmp_path_factory.mktemp("browser-paste"))
    host.plugin.get_model_def = lambda model_type: LTX_MODEL_DEF
    host.steps, host.guidance = 8, 2
    host.selected = ["lora_01.safetensors"]
    host.multipliers = "1;0\n#lora_off 1 0.45;0.25 lora_02.safetensors"
    with _open_panel(chromium, host) as opened:
        yield opened


def open_paste(page):
    """The paste box, open, and its field."""
    if page.locator(".lb-paste").count() == 0:
        page.locator('[data-lb="paste"]').click()
        settle(page, 200)
    return page.locator(".lb-paste textarea")


def close_paste(page):
    if page.locator(".lb-paste").count():
        page.keyboard.press("Escape")
        settle(page, 200)


def row_ids(page):
    return page.evaluate("() => [...document.querySelectorAll('.lb-row')].map(r => r.dataset.id)")


class TestPastingInTheBrowser:
    """Open the box, paste a stack, press Enter: it is the stack."""

    def test_the_box_opens_ready_to_paste_into_and_enter_loads_it(self, paste_panel):
        page, _, problems = paste_panel
        open_paste(page)
        assert page.evaluate("document.activeElement.matches('.lb-paste textarea')")
        assert page.locator('[data-lb="paste"]').get_attribute("aria-expanded") == "true"
        page.keyboard.insert_text(PASTED)
        page.keyboard.press("Enter")
        settle(page, 1500)

        assert page.locator(".lb-paste").count() == 0
        assert native(page) == "0,0,0,0,0.325,0.325,0,0;0 0.8;0.375,0.25,0.25 1;0.25"
        # The LoRA that was on and the one that was off are both replaced.
        assert row_ids(page) == ["lora_05.safetensors", "lora_06.safetensors", "lora_07.safetensors"]
        assert page.locator('[data-lb="status"]').inner_text() == "Loaded 3 LoRAs from text."
        # Back to the button that opened it, with nothing left to type over.
        assert page.evaluate("document.activeElement.dataset.lb") == "paste"
        assert open_paste(page).input_value() == ""
        close_paste(page)
        assert problems == []

    def test_shift_enter_starts_a_line_and_escape_keeps_the_draft(self, paste_panel):
        page, _, _ = paste_panel
        before = native(page)
        field = open_paste(page)
        field.fill("lora_01: x1")
        page.keyboard.press("Shift+Enter")
        page.keyboard.insert_text("lora_02: x0.5")
        settle(page, 300)
        assert field.input_value() == "lora_01: x1\nlora_02: x0.5"
        assert native(page) == before

        page.keyboard.press("Escape")
        settle(page, 200)
        assert page.locator(".lb-paste").count() == 0
        assert native(page) == before
        # Selected when the box opens again, so a paste replaces it.
        field = open_paste(page)
        assert field.input_value() == "lora_01: x1\nlora_02: x0.5"
        assert page.evaluate(
            "() => { const f = document.activeElement; return f.selectionEnd - f.selectionStart; }"
        ) == len("lora_01: x1\nlora_02: x0.5")
        field.fill("")
        close_paste(page)

    def test_a_line_that_cannot_load_keeps_the_box_open_and_says_why(self, paste_panel):
        page, _, _ = paste_panel
        field = open_paste(page)
        field.fill("lora_08: x0.5\nnot_a_lora: x1")
        page.keyboard.press("Enter")
        settle(page, 1500)

        assert page.locator(".lb-paste").count() == 1
        report = page.locator(".lb-paste-report").inner_text()
        assert "Loaded 1 of 2 LoRAs." in report
        assert "Line 2" in report and "no LoRA named 'not_a_lora'" in report
        # The text stays to be fixed; what could load, did.
        assert field.input_value() == "lora_08: x0.5\nnot_a_lora: x1"
        assert native(page) == "0.5;0.5"
        field.fill("")
        close_paste(page)

    def test_load_with_nothing_pasted_says_so(self, paste_panel):
        page, _, _ = paste_panel
        before = native(page)
        open_paste(page).fill("   ")
        page.locator('[data-lb="paste-load"]').click()
        settle(page, 400)
        assert "Paste a stack first" in page.locator(".lb-paste-report").inner_text()
        assert native(page) == before
        close_paste(page)

    def test_a_click_elsewhere_closes_it(self, paste_panel):
        page, _, _ = paste_panel
        open_paste(page)
        page.locator('[data-lb="active-title"]').click()
        settle(page, 200)
        assert page.locator(".lb-paste").count() == 0
        assert page.locator('[data-lb="paste"]').get_attribute("aria-expanded") == "false"

    def test_the_current_stack_copies_as_text_that_pastes_back(self, paste_panel):
        page, _, problems = paste_panel
        page.context.grant_permissions(["clipboard-read", "clipboard-write"])
        open_paste(page).fill(PASTED)
        page.keyboard.press("Enter")
        settle(page, 1500)

        open_paste(page)
        page.locator('[data-lb="paste-copy"]').click()
        settle(page, 400)
        assert page.evaluate("navigator.clipboard.readText()") == PASTED
        assert page.locator('[data-lb="status"]').inner_text() == "Copied the current stack (3 LoRAs)"
        close_paste(page)
        assert problems == []


#: WanGP's dark theme, read off a screenshot of it: panel, fields and rows all
#: one colour, and a white accent.
WANGP_DARK = {
    "--background-fill-primary": "#171717", "--background-fill-secondary": "#282828",
    "--input-background-fill": "#282828", "--border-color-primary": "#424242",
    "--color-accent": "#ffffff", "--border-color-accent": "#ffffff",
    "--body-text-color": "#e6e6e6", "--button-secondary-background-fill": "#282828",
}


def contrast(first, second):
    """WCAG contrast ratio between two sRGB colours."""
    def luminance(rgb):
        channels = [value / 255 for value in rgb]
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def colour(page, selector, prop="backgroundColor", pseudo=None):
    """A computed colour as [r, g, b], whatever syntax the browser reports it
    in -- color-mix() comes back as color(srgb ...)."""
    return page.evaluate(
        """(args) => {
          const node = document.querySelector(args.selector);
          const value = getComputedStyle(node, args.pseudo)[args.prop];
          const context = document.createElement('canvas').getContext('2d');
          context.fillStyle = value;
          context.fillRect(0, 0, 1, 1);
          return Array.from(context.getImageData(0, 0, 1, 1).data.slice(0, 3));
        }""",
        {"selector": selector, "prop": prop, "pseudo": pseudo},
    )


class TestRowsStandApart:
    """In a theme where the panel, its fields and its rows are one colour, a
    row was told apart from the next by a 1px line. Each is now a card on a
    tray of a different colour, with a band of it between every two."""

    @pytest.fixture
    def themed(self, off_panel):
        page, _, _ = off_panel
        page.evaluate(
            "(vars) => { for (const [k, v] of Object.entries(vars)) document.body.style.setProperty(k, v); }",
            WANGP_DARK,
        )
        settle(page, 100)
        yield page
        page.evaluate(
            "(vars) => { for (const k of Object.keys(vars)) document.body.style.removeProperty(k); }",
            WANGP_DARK,
        )

    def test_every_row_is_a_card_on_a_tray(self, themed):
        page = themed
        tray = colour(page, ".lb-rows")
        card = colour(page, ".lb-row")
        assert card != colour(page, "#lb_root_b")
        assert contrast(card, tray) >= 1.4

        boxes = page.evaluate(
            "() => [...document.querySelectorAll('.lb-row')].map(r => r.getBoundingClientRect().toJSON())"
        )
        assert len(boxes) == 3
        for upper, lower in zip(boxes, boxes[1:]):
            assert lower["top"] - upper["bottom"] >= 6
        rows = page.locator(".lb-rows").bounding_box()
        assert boxes[0]["top"] - rows["y"] >= 6
        assert boxes[0]["left"] - rows["x"] >= 6

    def test_a_switched_off_row_is_a_card_too(self, themed):
        page = themed
        page.locator('.lb-row[data-id="lora_02.safetensors"] button.lb-switch').click()
        settle(page, 1200)
        try:
            off = colour(page, '.lb-row[data-id="lora_02.safetensors"]')
            assert contrast(off, colour(page, ".lb-rows")) >= 1.3
        finally:
            page.locator('.lb-row[data-id="lora_02.safetensors"] button.lb-switch').click()
            settle(page, 1200)
        assert page.locator(".lb-row.lb-row-off").count() == 0

    def test_a_white_accent_hides_neither_the_switch_knob_nor_the_tile_tick(self, themed):
        page = themed
        switch = '.lb-row[data-id="lora_01.safetensors"] button.lb-switch'
        assert "lb-on" in page.locator(switch).get_attribute("class")
        knob = colour(page, switch, pseudo="::after")
        track = colour(page, switch, pseudo="::before")
        assert contrast(knob, track) >= 3

        tick = '.lb-tile[data-id="lora_01.safetensors"] .lb-check'
        assert contrast(colour(page, tick, "color"), colour(page, tick)) >= 3


@pytest.fixture(scope="module")
def pictures_panel(chromium, tmp_path_factory):
    """LTX-2.3 in Two Phases: two LoRAs with a picture, one whose preview is
    only a video, one with none; a phase 1 and a phase 2 scheduled."""
    pillow = pytest.importorskip("PIL.Image")
    root = tmp_path_factory.mktemp("browser-pictures")
    loras = root / "loras"
    loras.mkdir()
    for name in ("lora_01", "lora_02", "lora_03", "lora_04"):
        (loras / f"{name}.safetensors").write_bytes(b"")
    pillow.new("RGB", (64, 96), (200, 60, 40)).save(str(loras / "lora_01.png"))
    pillow.new("RGB", (64, 96), (40, 160, 90)).save(str(loras / "lora_03.png"))
    (loras / "lora_02.mp4").write_bytes(b"\x00")
    host = _Host(root)
    host.plugin.get_model_def = lambda model_type: LTX_MODEL_DEF
    host.steps, host.guidance = 8, 2
    host.selected = ["lora_01.safetensors", "lora_02.safetensors", "lora_03.safetensors"]
    host.multipliers = (
        "0,0,0,0,1,1,0,0;0 0.8;0.375,0.25,0.25 1;0.25\n#lora_off 3 0.5;0.5 lora_04.safetensors"
    )
    with _open_panel(chromium, host) as opened:
        settle(opened[0], 800)
        yield opened


def row_cells(page):
    """Each row's picture cell and where the rest of the row starts. ``inner``
    is the row's height above the band its editor adds under the picture."""
    return {cell["id"]: cell for cell in page.evaluate("""() =>
        [...document.querySelectorAll('.lb-row')].map(row => {
          const cell = row.firstElementChild;
          const box = cell.getBoundingClientRect();
          const band = row.querySelector(':scope > .lb-row-extra');
          return {
            id: row.dataset.id, first: cell.className, width: box.width, height: box.height,
            inner: row.clientHeight - (band ? band.offsetHeight : 0),
            main: row.querySelector('.lb-row-main').getBoundingClientRect().left,
            image: !!cell.querySelector('img.lb-thumb'),
            placeholder: !!cell.querySelector('.lb-placeholder'),
            play: !!cell.querySelector('.lb-play'),
          };
        })""")}


class TestRowPictures:
    """Every row starts with its LoRA's picture, in one column."""

    def test_every_row_starts_with_its_picture_and_the_rows_stay_aligned(self, pictures_panel):
        page, _, problems = pictures_panel
        cells = row_cells(page)
        assert list(cells) == [f"lora_0{index}.safetensors" for index in range(1, 5)]
        assert {cell["first"] for cell in cells.values()} == {"lb-row-thumb"}
        # One width for every row, so every row's controls start in one place...
        assert len({round(cell["width"]) for cell in cells.values()}) == 1
        assert len({round(cell["main"]) for cell in cells.values()}) == 1
        # ...and each picture is as tall as its own row, switched off or not.
        for cell in cells.values():
            assert abs(cell["height"] - cell["inner"]) <= 1
        # A row at rest, its strengths showing, is a square (the 1px is the
        # line between picture and controls).
        rest = cells["lora_03.safetensors"]
        assert abs((rest["width"] - 1) - rest["height"]) <= 2

        assert [(cells[f"lora_0{i}.safetensors"]["image"], cells[f"lora_0{i}.safetensors"]["play"])
                for i in range(1, 5)] == [(True, False), (False, True), (True, False), (False, False)]
        assert cells["lora_04.safetensors"]["placeholder"]
        assert problems == []

    def test_pressing_the_picture_opens_inspect(self, pictures_panel):
        page, host, _ = pictures_panel
        before = native(page)
        page.locator('.lb-row[data-id="lora_01.safetensors"] .lb-row-thumb-open').click()
        page.wait_for_selector(".lb-modal .lb-fetch-btn", timeout=5000)
        assert page.locator(".lb-modal-overlay").count() == 1
        assert native(page) == before
        page.keyboard.press("Escape")
        settle(page, 200)

    def test_a_per_step_phase_wears_a_badge_and_a_plain_one_its_number(self, pictures_panel):
        page, _, _ = pictures_panel
        chips = page.evaluate("""() => Object.fromEntries([...document.querySelectorAll('.lb-row')]
          .filter(row => row.querySelector('.lb-phase-strip'))
          .map(row => [row.dataset.id, [...row.querySelectorAll('.lb-phase-strip .lb-chip')].slice(0, 2)
            .map(chip => [chip.classList.contains('lb-chip-sched'),
                          chip.querySelector('.lb-sched-badge')?.textContent ?? null,
                          chip.querySelector('strong')?.textContent ?? null])]))""")
        assert chips["lora_01.safetensors"] == [[True, "∿Per step", None], [False, None, "0"]]
        assert chips["lora_02.safetensors"] == [[False, None, "0.8"], [True, "∿Per step", None]]
        assert chips["lora_03.safetensors"] == [[False, None, "1"], [False, None, "0.25"]]
        # The switch says the same of the phase each row shows.
        kinds = {lora: per_step(page.locator(f'.lb-row[data-id="{lora}"]'))
                 for lora in ("lora_01.safetensors", "lora_02.safetensors", "lora_03.safetensors")}
        assert kinds == {"lora_01.safetensors": True, "lora_02.safetensors": False,
                         "lora_03.safetensors": False}

    def test_the_badge_contrasts_with_its_pill_in_wangps_dark_theme(self, pictures_panel):
        page, _, _ = pictures_panel
        page.evaluate(
            "(vars) => { for (const [k, v] of Object.entries(vars)) document.body.style.setProperty(k, v); }",
            WANGP_DARK,
        )
        settle(page, 100)
        try:
            badge = '.lb-row[data-id="lora_02.safetensors"] .lb-chip-sched .lb-sched-badge'
            pill = '.lb-row[data-id="lora_02.safetensors"] .lb-chip-sched'
            plain = '.lb-row[data-id="lora_03.safetensors"] .lb-phase-strip .lb-chip:not(.lb-on)'
            # Text on the badge reads as text; the badge reads as a mark on
            # its pill, and on a plain pill's colour too.
            assert contrast(colour(page, badge, "color"), colour(page, badge)) >= 4.5
            assert contrast(colour(page, badge), colour(page, pill)) >= 3
            assert contrast(colour(page, badge), colour(page, plain)) >= 3
        finally:
            page.evaluate(
                "(vars) => { for (const k of Object.keys(vars)) document.body.style.removeProperty(k); }",
                WANGP_DARK,
            )

    def test_cards_have_nothing_to_collapse_and_full_screen_closes_alone(self, pictures_panel):
        page, _, problems = pictures_panel
        before = native(page)
        collapse = page.locator('[data-lb="collapse-all"]')
        # Every card's editor is open, by design: there is nothing to close.
        assert collapse.is_disabled()
        row = page.locator('.lb-row[data-id="lora_01.safetensors"]')
        row.locator(".lb-row-top button.lb-full").click()
        settle(page, 600)
        assert page.locator(".lb-sheet").count() == 1
        # The backdrop is the way out, as Escape is; neither changes anything.
        page.mouse.click(5, 5)
        settle(page, 500)
        assert page.locator(".lb-sheet").count() == 0
        assert native(page) == before
        assert collapse.is_disabled()
        assert problems == []

    def test_a_video_preview_plays_in_its_row(self, pictures_panel):
        page, _, _ = pictures_panel
        cell = page.locator('.lb-row[data-id="lora_02.safetensors"] .lb-row-thumb')
        cell.locator(".lb-play").click()
        settle(page, 1000)
        assert cell.locator("video.lb-thumb").count() == 1
        cell.locator(".lb-play").click()
        settle(page, 300)
        assert cell.locator("video").count() == 0
        assert cell.locator(".lb-placeholder").count() == 1


# ------------------------------------------- list view, modes, groups, bakes

from . import test_av_bridge as av  # noqa: E402

#: Real (tiny) LTX-2 LoRAs: one changes picture and sound, one only the
#: picture, one only the sound, and one runs a schedule in phase 1.
AV_LORAS = {
    "both.safetensors": (["attn1.to_q", "audio_attn1.to_q", "video_to_audio_attn.to_k"], 4),
    "picture.safetensors": (["attn1.to_q", "ff.net.2"], 8),
    "sound.safetensors": (["audio_attn1.to_q"], 2),
    "sched.safetensors": (["attn1.to_q", "audio_ff.net.2"], 4),
}
AV_SELECTED = list(AV_LORAS)
AV_MULTIPLIERS = "0.8;0.5 0.6;0.6 0.7;0.7 1,1,0.875,0.875,0.25,0.25,0,0;0"


@pytest.fixture(scope="module")
def av_panel(chromium, tmp_path_factory):
    """LTX-2.3 Distilled in Two Phases, in the list view, on real LoRA files.
    Builds wait in the queue until a test runs them."""
    root = tmp_path_factory.mktemp("browser-av")
    loras = root / "loras"
    loras.mkdir()
    for seed, (name, (modules, rank)) in enumerate(AV_LORAS.items(), 1):
        av._write(loras / name, modules, rank, seed)
    with pytest.MonkeyPatch.context() as patch:
        # Built files go under the plugin's folder: a temporary one here.
        patch.setattr(bridge.plugin_module, "PLUGIN_DIR", str(root / "plugin"))
        state = {"model_type": "ltx2_22B_distilled", "loras": list(AV_LORAS)}
        host = _Host(root, view="list", state=state)
        host.plugin.get_model_def = lambda model_type: LTX_MODEL_DEF
        host.plugin._builder = av.SyncBuilder()
        host.steps, host.guidance = 8, 2
        host.selected = list(AV_SELECTED)
        host.multipliers = AV_MULTIPLIERS
        with _open_panel(chromium, host) as opened:
            yield opened


def reset_av(page, host, multipliers=AV_MULTIPLIERS):
    """Each test starts from the same stack, in the list, nothing open."""
    host.plugin._builder.run_pending()
    host.plugin._metadata.set_stack_view("list")
    host.selected = list(AV_SELECTED)
    host.multipliers = multipliers
    page.reload()
    page.wait_for_selector(".lb-li", timeout=15000)
    settle(page, 400)


def line(page, lora):
    return page.locator(f'.lb-li[data-key="{lora}"]')


def tokens_by_lora(host):
    return dict(zip(host.selected, host.multipliers.split("\n")[0].split()))


def wait_until_built(page, host):
    host.plugin._builder.run_pending()
    page.wait_for_selector(".lb-buildbar[hidden]", state="attached", timeout=8000)
    settle(page, 300)


class TestListView:
    """The active LoRAs one line each; a line opens the full editor."""

    def test_a_line_shows_each_phase_and_counts_scheduled_steps(self, av_panel):
        page, host, problems = av_panel
        reset_av(page, host)
        assert page.locator(".lb-li").count() == 4
        assert page.locator(".lb-row").count() == 0
        assert "Phase 1 · 8 steps" in page.locator(".lb-li-head").inner_text()
        assert line(page, "both.safetensors").locator(".lb-li-ph code").all_inner_texts() == ["0.8", "0.5"]

        scheduled = line(page, "sched.safetensors")
        assert scheduled.locator(".lb-li-ph code").all_inner_texts() == ["6 steps", "0"]
        assert "steps 1–6 of 8" in scheduled.locator(".lb-li-ph").first.get_attribute("title")
        # How many steps, never each step's strength.
        assert "0.875" not in scheduled.inner_text() and "0.25" not in scheduled.inner_text()
        assert problems == []

    def test_only_one_line_is_open_at_a_time(self, av_panel):
        page, host, problems = av_panel
        reset_av(page, host)
        line(page, "both.safetensors").locator(".lb-li-name").click()
        settle(page, 300)
        assert page.locator(".lb-li-editor .lb-row").count() == 1
        assert page.locator('.lb-li-editor .lb-row[data-id="both.safetensors"]').count() == 1
        assert line(page, "both.safetensors").locator(".lb-li-name").get_attribute("aria-expanded") == "true"

        line(page, "picture.safetensors").locator(".lb-li-name").click()
        settle(page, 300)
        assert page.locator(".lb-li-editor .lb-row").count() == 1
        assert page.locator('.lb-li-editor .lb-row[data-id="picture.safetensors"]').count() == 1

        page.locator(".lb-li-editor button.lb-done").click()
        settle(page, 300)
        assert page.locator(".lb-li-editor").count() == 0
        assert problems == []

    def test_the_switch_on_a_line_does_not_open_it(self, av_panel):
        page, host, problems = av_panel
        reset_av(page, host)
        line(page, "picture.safetensors").locator("button.lb-switch").click()
        settle(page, 700)
        assert "#lora_off" in host.multipliers and "picture.safetensors" in host.multipliers
        assert page.locator(".lb-li-editor").count() == 0
        assert "lb-li-off" in line(page, "picture.safetensors").get_attribute("class")
        assert problems == []

    def test_visual_only_builds_a_copy_and_warns_until_it_is_ready(self, av_panel):
        page, host, problems = av_panel
        reset_av(page, host)
        line(page, "both.safetensors").locator(".lb-li-name").click()
        settle(page, 300)
        page.locator(".lb-li-editor .lb-seg-visual").click()
        page.wait_for_selector(".lb-buildbar:not([hidden])", timeout=5000)
        bar = page.locator(".lb-buildbar").inner_text()
        assert "unexpected result" in bar and "missing" in bar
        assert host.selected[0] == "_lora_browser/both.visual.safetensors"
        assert "queued" in line(page, "both.safetensors").locator(".lb-mode-pill").inner_text()
        # The same LoRA's editor stays open through the rename.
        assert page.locator('.lb-li-editor .lb-seg-visual[aria-checked="true"]').count() == 1

        wait_until_built(page, host)
        assert line(page, "both.safetensors").locator(".lb-mode-pill").inner_text().strip() == "Visual only"
        assert "exact copy of its picture tensors" in page.locator(".lb-li-editor .lb-load").inner_text()
        assert problems == []

    def test_every_lora_audio_only_switches_off_those_with_no_sound(self, av_panel):
        page, host, problems = av_panel
        reset_av(page, host)
        page.locator('[data-lb="av"]').click()
        page.locator(".lb-menu-portal button", has_text="audio only").click()
        settle(page, 800)
        assert "#lora_skip" in host.multipliers and "picture.audio.safetensors" in host.multipliers
        picture = line(page, "picture.safetensors")
        assert "lb-li-off" in picture.get_attribute("class")
        assert "nothing to apply" in picture.locator(".lb-mode-pill").inner_text()

        page.locator('[data-lb="av"]').click()
        page.locator(".lb-menu-portal button", has_text="normal").click()
        settle(page, 800)
        assert host.selected == AV_SELECTED and host.multipliers == AV_MULTIPLIERS
        assert page.locator(".lb-li.lb-li-off").count() == 0
        assert problems == []

    def test_grouping_lines_that_differ_asks_whose_settings_to_keep(self, av_panel):
        page, host, problems = av_panel
        reset_av(page, host)
        page.locator('[data-lb="select"]').click()
        line(page, "picture.safetensors").click()
        line(page, "sound.safetensors").click()
        assert "2 rows selected" in page.locator(".lb-bulk").inner_text()
        page.locator(".lb-bulk button", has_text="Group").click()
        page.wait_for_selector(".lb-dialog .lb-opt")
        page.locator(".lb-dialog .lb-opt").nth(1).click()          # sound's 0.7;0.7
        page.locator(".lb-dialog button.lb-primary").click()
        settle(page, 800)

        assert "#lora_group A picture.safetensors" in host.multipliers
        assert "#lora_group A sound.safetensors" in host.multipliers
        tokens = tokens_by_lora(host)
        assert tokens["picture.safetensors"] == tokens["sound.safetensors"] == "0.7;0.7"
        group = page.locator(".lb-li-group")
        assert group.count() == 1 and group.locator(".lb-li-member").count() == 2
        assert page.locator(".lb-li").count() == 5             # 3 lines + 2 members
        assert problems == []

    def test_a_group_bakes_at_its_highest_members_rank_and_unbakes(self, av_panel):
        page, host, problems = av_panel
        reset_av(page, host, "0.8;0.5 0.6;0.6 0.6;0.6 1,1,0.875,0.875,0.25,0.25,0,0;0"
                             "\n#lora_group A picture.safetensors\n#lora_group A sound.safetensors")
        page.locator(".lb-li-ghead .lb-li-name").click()
        settle(page, 300)
        page.locator(".lb-li-editor button", has_text="Bake to save memory").click()
        page.wait_for_selector(".lb-dialog .lb-opt")
        chosen = page.locator(".lb-dialog .lb-opt:has(input:checked)").inner_text()
        # Picture's rank, not sound's 2 -- and exact here: they share no layer.
        assert "rank 8" in chosen.lower() and "default" in chosen
        page.locator(".lb-dialog button.lb-primary").click()
        settle(page, 800)
        assert any(value.startswith("_lora_browser/_bakes/") and value.endswith(".r8.safetensors")
                   for value in host.selected)
        assert "picture.safetensors" not in host.selected

        wait_until_built(page, host)
        assert page.locator(".lb-li-ghead .lb-bake-pill").inner_text().startswith("Baked r8")

        page.locator(".lb-li-editor button", has_text="Unbake").click()
        settle(page, 800)
        assert "picture.safetensors" in host.selected and "sound.safetensors" in host.selected
        assert page.locator(".lb-li-ghead .lb-bake-pill").count() == 0
        assert problems == []

    def test_built_files_lists_them_and_deletes_what_nothing_uses(self, av_panel):
        page, host, problems = av_panel
        reset_av(page, host)
        line(page, "both.safetensors").locator(".lb-li-name").click()
        page.locator(".lb-li-editor .lb-seg-visual").click()
        settle(page, 600)
        wait_until_built(page, host)
        page.locator(".lb-li-editor .lb-seg-normal").click()
        settle(page, 800)

        link = page.locator('[data-lb="built"]')
        assert link.is_visible() and link.inner_text().startswith("Built files:")
        link.click()
        page.wait_for_selector(".lb-dialog .lb-table")
        table = page.locator(".lb-dialog .lb-table").inner_text()
        assert "both.visual.safetensors" in table and "nothing" in table
        page.locator(".lb-dialog button.lb-left").click()
        page.wait_for_selector(".lb-dialog :text('Nothing is built')", timeout=5000)
        page.locator(".lb-dialog button.lb-primary").click()
        assert page.locator(".lb-dialog").count() == 0
        settle(page, 800)
        assert not page.locator('[data-lb="built"]').is_visible()
        assert problems == []

    def test_the_view_is_remembered(self, av_panel):
        page, host, problems = av_panel
        reset_av(page, host)
        page.locator('[data-lb="viewsw"] button[data-view="cards"]').click()
        settle(page, 700)
        assert page.locator(".lb-row").count() == 4 and page.locator(".lb-li").count() == 0
        assert host.plugin._metadata.stack_view == "cards"
        page.reload()
        page.wait_for_selector(".lb-row", timeout=15000)
        settle(page, 300)
        assert page.locator('[data-lb="viewsw"] button[data-view="cards"]').get_attribute("aria-checked") == "true"
        assert problems == []

    def test_the_list_fits_a_phone(self, av_panel):
        page, host, problems = av_panel
        reset_av(page, host)
        page.set_viewport_size({"width": 390, "height": 900})
        try:
            settle(page, 400)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
            assert not page.locator(".lb-li-head").is_visible()
            # Each phase carries its own label once the columns are gone.
            label = page.evaluate("""() => getComputedStyle(
                document.querySelector('.lb-li[data-key="both.safetensors"] .lb-li-ph'), '::before').content""")
            assert "P1" in label
        finally:
            page.set_viewport_size({"width": 1100, "height": 1000})
            settle(page, 300)
        assert problems == []

    def test_changing_side_keeps_the_editor_as_it_was(self, av_panel):
        page, host, problems = av_panel
        reset_av(page, host)
        line(page, "sched.safetensors").locator(".lb-li-name").click()
        settle(page, 300)
        editor = page.locator(".lb-li-editor")
        assert per_step(editor)
        pick(page, editor, 3)
        # The file it loads is renamed; the row, and the step selected on it,
        # are not.
        page.locator(".lb-li-editor .lb-seg-visual").click()
        settle(page, 900)
        assert host.selected[3] == "_lora_browser/sched.visual.safetensors"
        assert per_step(editor)
        assert picked(editor) == ["3"]
        assert problems == []

    def test_an_open_line_edits_on_its_own_tools_and_done_closes_it(self, av_panel):
        page, host, problems = av_panel
        reset_av(page, host)
        line(page, "both.safetensors").locator(".lb-li-name").click()
        settle(page, 300)
        editor = page.locator(".lb-li-editor")
        # The line above has the picture, name and switch; the editor opens on
        # its tools, and has a section for the strength.
        assert editor.locator(".lb-row-thumb, .lb-row-top, button.lb-switch").count() == 0
        tools = [text.strip() for text in editor.locator(".lb-toolbar-row button").all_inner_texts()]
        assert tools == ["Full screen", "Inspect", "Remove"]
        assert editor.locator(".lb-ed-title").first.inner_text().lower() == "strength"
        # Collapse all closes it, as Done does; neither changes anything.
        before = native(page)
        collapse = page.locator('[data-lb="collapse-all"]')
        assert not collapse.is_disabled()
        collapse.click()
        settle(page, 300)
        assert page.locator(".lb-li-editor").count() == 0
        assert collapse.is_disabled()
        assert native(page) == before
        assert problems == []

    def test_full_screen_from_the_list_comes_back_to_the_open_line(self, av_panel):
        page, host, problems = av_panel
        reset_av(page, host)
        line(page, "sched.safetensors").locator(".lb-li-name").click()
        settle(page, 300)
        before = native(page)
        page.locator(".lb-li-editor .lb-toolbar-row button.lb-full").click()
        settle(page, 600)
        sheet = page.locator(".lb-sheet")
        assert sheet.count() == 1
        assert sheet.locator(".lb-sheet-name").inner_text() == "sched.safetensors"
        assert page.locator(".lb-li-editor .lb-full-note").count() == 1
        # Focus starts on the way out.
        assert "lb-full" in page.evaluate("document.activeElement.className")
        sheet.locator(".lb-sheet-head button.lb-full").click()
        settle(page, 500)
        assert page.locator(".lb-sheet").count() == 0
        assert page.locator('.lb-li-editor .lb-row[data-id="sched.safetensors"] .lb-bars').count() == 1

        # Done in full screen closes the LoRA altogether.
        page.locator(".lb-li-editor .lb-toolbar-row button.lb-full").click()
        settle(page, 600)
        page.locator(".lb-sheet .lb-sheet-head button.lb-done").click()
        settle(page, 500)
        assert page.locator(".lb-sheet").count() == 0
        assert page.locator(".lb-li-editor").count() == 0
        assert native(page) == before
        assert problems == []

    def test_an_open_line_fits_a_phone(self, av_panel):
        page, host, problems = av_panel
        reset_av(page, host)
        page.set_viewport_size({"width": 390, "height": 900})
        try:
            settle(page, 300)
            line(page, "sched.safetensors").locator(".lb-li-name").click()
            settle(page, 400)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
            editor = page.locator(".lb-li-editor")
            # Big targets, and the slider on a line of its own above the number.
            for button in editor.locator(".lb-toolbar-row button").all():
                assert button.bounding_box()["height"] >= 44
            weight = editor.locator(".lb-weight-steps")
            slider = weight.locator("input[type=range]").bounding_box()
            number = weight.locator("input.lb-number").bounding_box()
            assert number["y"] >= slider["y"] + slider["height"] - 1
            # One strength and Per step share the width.
            seg = editor.locator(".lb-kind-seg").bounding_box()
            assert seg["width"] >= editor.locator(".lb-strength").bounding_box()["width"] - 24
        finally:
            page.set_viewport_size({"width": 1100, "height": 1000})
            settle(page, 300)
        assert problems == []

    def test_separate_strengths_edit_picture_and_sound_apart(self, av_panel):
        page, host, problems = av_panel
        reset_av(page, host)
        line(page, "both.safetensors").locator(".lb-li-name").click()
        settle(page, 300)
        page.locator(".lb-li-editor .lb-seg-separate").click()
        settle(page, 900)
        visual, sound = "_lora_browser/both.visual.safetensors", "_lora_browser/both.sound.safetensors"
        assert host.selected[:2] == [visual, sound]
        # One line in the list, one editor with a section per side.
        assert page.locator(".lb-li").count() == 4
        sections = page.locator(".lb-li-editor .lb-track")
        assert sections.count() == 2
        assert sections.nth(0).locator(".lb-track-name").inner_text().lower() == "picture"
        assert sections.nth(1).locator(".lb-track-name").inner_text().lower() == "sound"
        # The sound side's own strength moves only the sound half.
        number = sections.nth(1).locator(".lb-weight-main input[type=number]")
        number.fill("0.3")
        number.press("Enter")
        settle(page, 900)
        tokens = tokens_by_lora(host)
        assert tokens[visual] == "0.8;0.5" and tokens[sound].startswith("0.3")
        cells = line(page, "both.safetensors").locator(".lb-li-pair").first.locator(".lb-li-side code")
        assert cells.all_inner_texts() == ["0.8", "0.3"]
        host.plugin._builder.run_pending()
        assert problems == []

    def test_a_group_takes_picture_and_sound_strengths(self, av_panel):
        page, host, problems = av_panel
        reset_av(page, host, "0.8;0.5 0.6;0.6 0.6;0.6 1,1,0.875,0.875,0.25,0.25,0,0;0"
                             "\n#lora_group A both.safetensors\n#lora_group A sound.safetensors")
        page.locator(".lb-li-ghead .lb-li-name").click()
        settle(page, 300)
        page.locator(".lb-li-editor .lb-seg-separate").click()
        settle(page, 900)
        assert "#lora_group A _lora_browser/both.sound.safetensors" in host.multipliers
        assert page.locator(".lb-li-editor .lb-track").count() == 2
        assert "picture and sound" in page.locator(".lb-li-ghead .lb-li-sub").inner_text()
        strengths = page.locator(".lb-li-editor .lb-extra-line", has_text="Strengths")
        strengths.locator(".lb-seg-normal").click()
        settle(page, 900)
        assert "sound.safetensors" in host.selected and ".sound.safetensors" not in host.multipliers
        host.plugin._builder.run_pending()
        assert problems == []
