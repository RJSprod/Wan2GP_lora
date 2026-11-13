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
        """This model's phases are cut from one run, so a slot is not a step."""
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        assert schedule_of(page, "lora_02.safetensors")["coordinate_mode"] == "phase_relative"
        assert row.locator(".lb-region .lb-region-body span").first.inner_text().startswith("Slot")

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
        # Exactly one strength control on the row -- the plain row's, under
        # the timeline -- and it belongs to the selected region.
        assert row.locator("input[type=range]").count() == 1
        assert row.locator(".lb-schedule .lb-weight input[type=range]").count() == 1
        selected = row.locator(".lb-region.lb-on strong").inner_text()
        assert row.locator(".lb-schedule .lb-weight input[type=number]").input_value() == selected
        # The old header line, region panel, "+ Region" and "Slots" are gone;
        # one row of tools is left, with Delete for the selected region.
        tools = row.locator(".lb-sched-tools button")
        assert [tools.nth(i).inner_text() for i in range(tools.count())] == [
            "\u2922 Full screen", "Clear schedule", "Delete",
        ]
        for gone in (".lb-sched-head", ".lb-region-editor", "button:has-text('+ Region')",
                     "button:has-text('Slots')"):
            assert row.locator(gone).count() == 0, gone
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
        # The Schedule button closes it as it opened it; nothing else does.
        row.locator("button.lb-schedule-toggle").click()
        settle(page)
        assert row.locator(".lb-timeline").count() == 0
        assert native(page) == before

        row.locator("button.lb-schedule-toggle").click()
        settle(page, 700)
        assert row.locator(".lb-region").count() == regions

    def test_the_inline_timeline_never_scrolls(self, panel):
        """Every step is in view at whatever width the panel is given."""
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        open_timeline(page, row)

        for width in (1100, 700, 400):
            page.set_viewport_size({"width": width, "height": 1000})
            settle(page, 400)
            sizes = page.evaluate("""() => {
              const w = document.querySelector('.lb-timeline-wrap');
              const t = w.querySelector('.lb-timeline');
              return {scroll: w.scrollWidth, client: w.clientWidth,
                      timeline: t.getBoundingClientRect().width};
            }""")
            assert sizes["scroll"] <= sizes["client"], (width, sizes)
            assert sizes["timeline"] <= sizes["client"] + 1, (width, sizes)
        page.set_viewport_size({"width": 1100, "height": 1000})
        settle(page, 400)

    def test_full_screen_is_the_same_editor_with_room(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        open_timeline(page, row)
        before = native(page)
        inline = row.locator(".lb-timeline").bounding_box()

        try:
            row.locator(".lb-sched-tools button:has-text('Full screen')").click()
            settle(page, 700)
            modal = page.locator(".lb-sched-modal")
            assert modal.count() == 1
            assert native(page) == before        # opening it is a view change

            # Full width like the row, and taller: the room is vertical, which
            # is what makes a region easy to aim at.
            box = modal.locator(".lb-timeline").bounding_box()
            assert box["width"] >= inline["width"] - 8
            assert box["height"] > inline["height"]

            # It is live: an edit here reaches WanGP like any other.
            modal.locator(".lb-region").first.click()
            settle(page, 400)
            # The same tools, minus the way into full screen.
            assert modal.locator(".lb-sched-tools button:has-text('Full screen')").count() == 0
            strength = modal.locator(".lb-weight input[type=number]")
            strength.fill("0.44")
            strength.press("Enter")
            settle(page, 1500)
            assert "0.44" in native(page).split()[1]
        finally:
            # A left-open overlay would swallow every later test's clicks.
            page.keyboard.press("Escape")
            settle(page, 400)

        assert page.locator(".lb-sched-modal").count() == 0
        # The row it came from is untouched, and still shows the edit.
        assert row.locator(".lb-timeline").count() == 1
        assert "0.44" in native(page).split()[1]

    def test_clear_phase_returns_it_to_a_plain_strength(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')
        hidden = native(page).split()[1].split(";")[1]
        row.locator(".lb-sched-tools button:has-text('Clear schedule')").click()
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

        assert row.locator(".lb-region.lb-on .lb-region-body span").inner_text() == "Slots 8\u201314"
        # A new region starts at the strength the row already had, and the
        # strength control is now the region's.
        assert row.locator(".lb-schedule .lb-weight input[type=number]").input_value() == "0.37"
        assert row.locator(".lb-sched-tools button:has-text('Delete')").count() == 1

    def test_the_drawn_region_is_what_wangp_is_given(self, panel):
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_01.safetensors"]')
        phase_two = native(page).split()[0].split(";")[1]

        strength = row.locator(".lb-schedule .lb-weight input[type=number]")
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
        assert row.locator(".lb-region.lb-on .lb-region-body span").inner_text().startswith("Slots 21")

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
        schedule has to come with it -- with nothing on it moving."""
        page, _, _ = panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')

        def phase_one():
            return native(page).split()[1].split(";")[0].split(",")

        try:
            # A schedule one value per step of a four-step run, the last step
            # different so a stretch or a lost step would show.
            set_steps(page, 4)
            set_token(page, 1, "0.5,0.5,0.3,0.8;0.7")
            open_timeline(page, row)
            drawn = schedule_of(page, "lora_02.safetensors")
            assert drawn["slots"] == 4
            assert phase_one() == ["0.5", "0.5", "0.3", "0.8"]
            assert phase_one() == expected_values(drawn)
            assert row.locator(".lb-tick").last.inner_text() == "4"

            # Longer: everything stays where it was, the new step arrives empty.
            set_steps(page, 5)
            grown = schedule_of(page, "lora_02.safetensors")
            assert grown["slots"] == 5
            assert phase_one() == ["0.5", "0.5", "0.3", "0.8", "0"]
            assert grown["regions"] == drawn["regions"]
            assert row.locator(".lb-kept").count() == 0

            # Shorter: the run stops sooner, and the step it no longer reaches
            # is kept -- and the timeline says so.
            set_steps(page, 3)
            assert schedule_of(page, "lora_02.safetensors")["slots"] == 3
            assert phase_one() == ["0.5", "0.5", "0.3"]
            kept = row.locator(".lb-timeline .lb-kept")
            assert kept.count() == 1
            assert kept.inner_text() == "+1 step kept"

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

        drawn = native(page).split()[0]
        try:
            set_guidance(page, 2)
            settle(page, 900)
            assert native(page).split()[0] == drawn
            assert row.locator("button.lb-schedule-toggle").count() == 0
            assert row.locator(".lb-timeline").count() == 0
            assert "\u223f" in row.locator(".lb-kept-summary").inner_text()
            assert "One Phase" in row.locator(".lb-preserved-note").inner_text()
        finally:
            set_guidance(page, 1)
        assert native(page).split()[0] == drawn
        assert row.locator("button.lb-schedule-toggle").count() == 1

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


class TestLtxInTheBrowser:
    """LTX-2 in the real frontend: its phases are stages, each a timeline."""

    def test_each_stage_is_a_timeline_of_its_own_length(self, ltx_panel):
        page, _, problems = ltx_panel
        row = page.locator('.lb-row[data-id="lora_01.safetensors"]')
        assert row.locator(".lb-phase-strip .lb-chip").count() >= 2

        select_phase(page, row, 0)
        open_timeline(page, row)
        first = schedule_of(page, "lora_01.safetensors", 0)
        assert (first["stage"], first["stage_steps"]) == (1, 8)
        assert row.locator(".lb-tick").last.inner_text() == "8"

        select_phase(page, row, 1)
        settle(page, 900)
        second = schedule_of(page, "lora_01.safetensors", 1)
        assert (second["stage"], second["stage_steps"]) == (2, 3)
        assert row.locator(".lb-tick").last.inner_text() == "3"

        # A tap on the refinement's last step lands in the token's second part.
        timeline = row.locator(".lb-timeline")
        timeline.scroll_into_view_if_needed()
        settle(page, 200)
        box = timeline.bounding_box()
        page.mouse.click(box["x"] + box["width"] * 0.85, box["y"] + box["height"] / 2)
        settle(page, 1300)
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
        open_timeline(page, row)
        settle(page, 600)
        assert schedule_of(page, "lora_02.safetensors", 1)["stage_steps"] == 3
        assert row.locator(".lb-region").count() == 0
        # Nothing drawn: the strength control is what the stage is worth, 0.
        assert row.locator(".lb-schedule .lb-weight input[type=number]").input_value() == "0"
        assert "draw" in row.locator(".lb-baseline").inner_text().lower()

        # Drawing on it starts at full strength, in the token's second part.
        timeline = row.locator(".lb-timeline")
        timeline.scroll_into_view_if_needed()
        settle(page, 200)
        box = timeline.bounding_box()
        page.mouse.click(box["x"] + box["width"] * 0.85, box["y"] + box["height"] / 2)
        settle(page, 1300)
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

        select_phase(page, row, 0)
        open_timeline(page, row)
        select_phase(page, row, 1)
        settle(page, 900)
        second = schedule_of(page, "lora_02.safetensors", 1)
        assert second["stage_steps"] == 3
        assert [(r["start"], r["end"], r["strength"]) for r in second["regions"]] == [(1, 3, 1)]
        assert row.locator(".lb-region").count() == 1
        assert row.locator("button.lb-fill-stage-2").count() == 0
        assert problems == []

    def test_fill_from_stage_1_draws_the_refinement_once(self, ltx_panel):
        page, _, problems = ltx_panel
        row = page.locator('.lb-row[data-id="lora_02.safetensors"]')

        def show_stage_2():
            # A chip only asks for a timeline when it changes the selection.
            select_phase(page, row, 0)
            open_timeline(page, row)
            select_phase(page, row, 1)
            settle(page, 900)
            return row.locator("button.lb-fill-stage-2")

        # Off where stage 2's noise levels are: the button says why, and is off.
        set_token(page, 1, "1,1,1,1,0,0,0,0;0")
        button = show_stage_2()
        assert button.count() == 1 and button.is_disabled()
        assert "nothing to carry" in button.get_attribute("title")

        set_token(page, 1, "1,1,1,1,0.8,0.6,0.4,0.2;0")
        button = show_stage_2()
        assert button.is_enabled()
        assert button.get_attribute("title").endswith("0.6, 0.4, 0.2")
        button.click()
        settle(page, 1300)
        assert native(page).split()[1] == "1,1,1,1,0.8,0.6,0.4,0.2;0.6,0.4,0.2"
        assert row.locator(".lb-region").count() == 3
        # Drawn once: stage 2 is regions now, with nothing left to fill.
        assert row.locator("button.lb-fill-stage-2").count() == 0
        assert problems == []


class TestH3InTheBrowser:
    """MiniMax H3: phase 1 is the same full run in One Phase and Two, so a
    switch between them leaves it -- schedule and all -- exactly as it was."""

    def test_two_phases_keep_phase_1_and_its_timeline(self, h3_panel):
        page, _, problems = h3_panel
        token = "0,-2,0,0;1 0.7;0.5"
        row = page.locator('.lb-row[data-id="lora_01.safetensors"]')
        open_timeline(page, row)
        assert row.locator(".lb-region").count() == 1
        try:
            set_guidance(page, 2)
            settle(page, 900)
            assert native(page) == token
            assert row.locator(".lb-phase-strip .lb-chip").count() >= 2
            select_phase(page, row, 0)
            open_timeline(page, row)
            first = schedule_of(page, "lora_01.safetensors", 0)
            assert first["slots"] == 4
            assert [(r["start"], r["end"], r["strength"]) for r in first["regions"]] == [(2, 2, -2)]
            assert row.locator(".lb-region").count() == 1
            # Phase 2 is the enhancement pass: a timeline as long as it runs.
            select_phase(page, row, 1)
            settle(page, 900)
            assert schedule_of(page, "lora_01.safetensors", 1)["slots"] == 3
            assert native(page) == token
        finally:
            set_guidance(page, 1)
            settle(page, 900)
        assert native(page) == token
        assert schedule_of(page, "lora_01.safetensors", 0)["regions"][0]["strength"] == -2
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
        assert row.locator("input, button.lb-schedule-toggle").count() == 0
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

    def test_a_scheduled_phase_wears_a_badge_and_a_plain_one_its_number(self, pictures_panel):
        page, _, _ = pictures_panel
        chips = page.evaluate("""() => Object.fromEntries([...document.querySelectorAll('.lb-row')]
          .filter(row => row.querySelector('.lb-phase-strip'))
          .map(row => [row.dataset.id, [...row.querySelectorAll('.lb-phase-strip .lb-chip')].slice(0, 2)
            .map(chip => [chip.classList.contains('lb-chip-sched'),
                          chip.querySelector('.lb-sched-badge')?.textContent ?? null,
                          chip.querySelector('strong')?.textContent ?? null])]))""")
        assert chips["lora_01.safetensors"] == [[True, "∿Scheduled", None], [False, None, "0"]]
        assert chips["lora_02.safetensors"] == [[False, None, "0.8"], [True, "∿Scheduled", None]]
        assert chips["lora_03.safetensors"] == [[False, None, "1"], [False, None, "0.25"]]
        # The Schedule button says the same of the LoRA as a whole.
        toggles = {
            lora: "lb-scheduled" in page.locator(
                f'.lb-row[data-id="{lora}"] button.lb-schedule-toggle').get_attribute("class")
            for lora in ("lora_01.safetensors", "lora_02.safetensors", "lora_03.safetensors")
        }
        assert toggles == {"lora_01.safetensors": True, "lora_02.safetensors": True,
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

    def test_collapse_all_closes_every_open_schedule_and_changes_nothing(self, pictures_panel):
        page, _, problems = pictures_panel
        before = native(page)
        collapse = page.locator('[data-lb="collapse-all"]')
        assert collapse.is_disabled()
        for lora in ("lora_01.safetensors", "lora_02.safetensors"):
            page.locator(f'.lb-row[data-id="{lora}"] button.lb-schedule-toggle').click()
            settle(page, 1000)
        assert page.locator(".lb-timeline").count() == 2
        assert not collapse.is_disabled()
        assert "2 open schedules" in collapse.get_attribute("title")

        collapse.click()
        settle(page, 500)
        assert page.locator(".lb-timeline").count() == 0
        assert collapse.is_disabled()
        assert native(page) == before
        # Closed, not cleared: the schedule is still there, said in one line.
        assert page.locator('.lb-row[data-id="lora_01.safetensors"] .lb-sched-summary').count() == 1
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
        page.locator(".lb-li-editor button.lb-schedule-toggle").click()
        settle(page, 700)
        assert page.locator(".lb-li-editor .lb-timeline").count() == 1
        # The file it loads is renamed; the row, and its open timeline, are not.
        page.locator(".lb-li-editor .lb-seg-visual").click()
        settle(page, 900)
        assert host.selected[3] == "_lora_browser/sched.visual.safetensors"
        assert page.locator(".lb-li-editor .lb-timeline").count() == 1
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
