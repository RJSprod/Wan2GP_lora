/* WanGP LoRA Browser -- frontend.
 *
 * Contract with plugin.py:
 *   - Python pushes a JSON payload into a hidden textbox; its change event
 *     calls apply().
 *   - The frontend sends typed action messages back through a second hidden
 *     textbox plus a hidden button click.
 *   - Thumbnails are requested lazily in batches for tiles that scroll into
 *     view, so a library of hundreds does not block first render.
 *
 * Placeholders (__NS__, __INST__, ...) are substituted per media-generator tab,
 * because WanGP can build the tab more than once.
 */
(function () {
  "use strict";

  var INSTANCE = "__INST__";
  var NS = "__NS__";
  var IDS = {
    root: "__ROOT__",
    payload: "__PAYLOAD_ID__",
    action: "__ACTION_ID__",
    actionBtn: "__ACTION_BTN__",
    thumbReq: "__THUMB_REQ_ID__",
    thumbBtn: "__THUMB_BTN__",
    thumbRes: "__THUMB_RES_ID__",
    refreshBtn: "__REFRESH_BTN__",
    nativeChoices: "__NATIVE_CHOICES__",
    nativeMultipliers: "__NATIVE_MULTIPLIERS__"
  };

  var MAX_VISIBLE_ROWS = 3;
  var SYNC_DEBOUNCE_MS = 110;

  window.wgpLoraBrowser = window.wgpLoraBrowser || {};
  if (window.wgpLoraBrowser[INSTANCE]) { return; }

  var S = {
    revision: 0,
    modelKey: "",
    phases: { capacity: 1, effective: 1 },
    phaseLabels: ["Strength"],
    items: [],
    rows: [],
    byId: {},
    search: "",
    sortMode: "name",
    nameMode: "civitai",
    defaultProfile: "",
    autoDefaultApplied: {},
    zoomPx: 104,
    sliderMin: 0,
    sliderMax: 1,
    sliderStep: 0.05,
    nudgeStep: 0.01,
    valueMin: -10,
    valueMax: 10,
    valueDecimals: 4,
    steps: 0,
    //: Steps each LoRA phase runs, where the model's phases are stages of
    //  known length (LTX-2, MiniMax H3); empty elsewhere.
    phaseSteps: [],
    //: Step schedules only mean what they show in One Phase guidance, or where
    //  every phase is a stage of known length (LTX-2, MiniMax H3).
    schedulingEnabled: true,
    schedulingDisabledReason: "",
    profiles: [],
    profilesIncomplete: [],
    civitaiKeySet: false,
    bulkFetch: null,
    activeProfile: "",
    status: "",
    statusWarn: false,
    //: Whether "Disable all" / "Enable all" would change anything right now.
    canDisableAll: false,
    canEnableAll: false,
    thumbs: {},
    requested: {},
    pendingThumbs: [],
    linked: {},
    // View state, not data: which phase chip is selected, whether a row's
    // timeline is expanded, and which region is highlighted. Losing any of it
    // costs a click; it never costs a schedule.
    selectedPhaseById: {},
    scheduleOpenById: {},
    selectedRegionByKey: {},
    //: Transient pointer state for a region drag; never anything canonical.
    drag: null,
    //: The full-screen schedule editor, when one is open.
    scheduleModal: null,
    ready: false,
    mounted: false,
    forceNative: false,
    layoutBroken: false,
    remountQueued: false,
    dragging: false,
    //: The user's latest intent per control, held until the exchange settles.
    //  Without it, the answer to an earlier tap repaints over a later one.
    localValues: {},
    //: One batch of edits is in flight at a time; see flushSync.
    inFlight: false,
    inFlightTimer: null,
    signature: "",
    rowsKey: "",
    syncTimer: null,
    thumbTimer: null,
    mediaQueue: [],
    mediaSeq: 0,
    mediaInFlight: null,
    mediaTimer: null,
    bulkTimer: null,
    playingTile: null,
    searchTerms: [],
    modal: null,
    pendingValues: {},
    //: The LoRAs that run, as text the paste box reads back.
    stackText: "",
    //: The paste box, while it is open, and what was typed into it -- kept
    //  when it closes so a click elsewhere does not cost the text.
    paste: null,
    pasteDraft: "",
    pasteSeq: 0,
    //: The active LoRAs as a read-only list ("list") or as every row's
    //  editor ("cards"); in the list, at most one row is open.
    view: "list",
    openRow: "",
    //: Select mode: rows picked for a bulk change, by row key.
    selecting: false,
    selected: {},
    //: LTX-2: LoRAs have a picture side and a sound side to choose between.
    avSupported: false,
    //: What is building for this stack, from the payload and the poll.
    builds: null,
    buildTimer: null,
    built: { count: 0, bytes: 0 },
    dialog: null,
    dialogReturn: null
  };

  var el = {};
  var observer = null;
  var resizeObserver = null;
  var anchorObserver = null;
  var integrityTimer = null;
  var menu = null;

  /* ------------------------------------------------------------ helpers */

  function byId(id) { return document.getElementById(id); }

  function text(value) { return value === undefined || value === null ? "" : String(value); }

  function fmt(value) {
    var number = Math.round(Number(value) * 10000) / 10000;
    if (!isFinite(number)) { return "1"; }
    return String(number);
  }

  /* Gradio owns these inputs; writing the value alone is not enough, the
     framework only notices after a bubbling input event. */
  function setHidden(id, value) {
    var host = byId(id);
    if (!host) { return false; }
    var field = host.querySelector("textarea, input");
    if (!field) { return false; }
    var setter = Object.getOwnPropertyDescriptor(field.__proto__, "value");
    if (setter && setter.set) { setter.set.call(field, value); } else { field.value = value; }
    field.dispatchEvent(new Event("input", { bubbles: true }));
    return true;
  }

  function clickHidden(id) {
    var host = byId(id);
    var button = host && (host.tagName === "BUTTON" ? host : host.querySelector("button"));
    if (button) { button.click(); return true; }
    return false;
  }

  function send(action) {
    if (!setHidden(IDS.action, JSON.stringify(action))) { return; }
    clickHidden(IDS.actionBtn);
  }

  function setStatus(message, warn) {
    S.status = text(message);
    S.statusWarn = !!warn;
    if (el.status) {
      el.status.textContent = S.status;
      el.status.classList.toggle("lb-warn", S.statusWarn);
    }
  }

  /* --------------------------------------------------------- native UI */

  /* Locate the two native controls the panel stands in for.
     Preferred route is the elem_id the plugin assigns them.  The fallback uses
     position: the panel is injected directly after the multiplier textbox, so
     the two blocks preceding our host are the native controls. */
  /* Pick the element to hide for one native control.
     Climbing to the component's wrapper is what makes it disappear cleanly,
     but the wrapper must never be an ancestor of our own panel: WanGP groups
     loras_choices and loras_multipliers into a Gradio <div class="form">, and
     the panel is injected INSIDE that same form. Hiding the form would hide
     the panel with it. When the wrapper is unsafe, hide just the element that
     carries the elem_id. */
  function hideTargetFor(id) {
    var node = byId(id);
    if (!node) { return null; }
    var root = byId(IDS.root);
    var candidate = node.closest(".block, .form") || node;
    if (root && candidate.contains(root)) { candidate = node; }
    if (root && candidate.contains(root)) { return null; }
    return candidate;
  }

  function nativeBlocks() {
    var blocks = [];
    [IDS.nativeChoices, IDS.nativeMultipliers].forEach(function (id) {
      var target = hideTargetFor(id);
      if (target) { blocks.push(target); }
    });
    if (blocks.length === 2) { return blocks; }

    var host = byId(IDS.root);
    var panel = host && host.closest(".wgp-lora-browser-host");
    var sibling = panel && panel.previousElementSibling;
    var found = [];
    while (sibling && found.length < 2) {
      // Only take blocks that actually hold a control, so a stray label or
      // markdown block is never hidden by mistake.
      if (sibling.querySelector("textarea, input, select")) { found.push(sibling); }
      sibling = sibling.previousElementSibling;
    }
    return found.length === 2 ? found : blocks;
  }

  /* Has the panel actually drawn something a user can work with?  Layout box
     size is deliberately not part of this: the LoRAs tab may not be the
     selected tab, and a hidden tab legitimately measures zero. */
  function panelHasContent() {
    var root = byId(IDS.root);
    if (!root) { return false; }
    return !!(root.querySelector(".lb-tile") || root.querySelector(".lb-row") || root.querySelector(".lb-empty"));
  }

  /* Fail open: the native selector and multiplier box are only ever hidden
     while the panel is genuinely standing in for them. */
  function hideNative(hide) {
    var conceal = !!hide && panelHasContent() && !S.forceNative && !S.layoutBroken;
    var root = byId(IDS.root);
    var heightBefore = root ? root.getBoundingClientRect().height : 0;
    var targets = nativeBlocks();

    targets.forEach(function (block) {
      block.style.display = conceal ? "none" : "";
    });

    // Backstop for any DOM shape we did not anticipate: if concealing the
    // native controls just collapsed the panel, we hid one of our own
    // ancestors. Put it back rather than leaving an empty tab.
    if (conceal && root && heightBefore > 0 && root.getBoundingClientRect().height === 0) {
      targets.forEach(function (block) { block.style.display = ""; });
      console.warn("[LoRA Browser] hiding the native controls collapsed the panel; leaving them visible.");
    }
  }

  /* ------------------------------------------------------------- mount */

  /* The panel lives inside a gr.HTML component, and Gradio renders that with
     Svelte's {@html value}: any re-render of the component reassigns its
     innerHTML and destroys everything mounted into it.  So "mounted" can never
     be a one-shot flag -- the skeleton has to be rebuildable at any time. */
  function panelIntact() {
    var root = byId(IDS.root);
    return !!(root && el.root === root && root.querySelector('[data-lb="grid"]'));
  }

  function mount(force) {
    var root = byId(IDS.root);
    if (!root) { return false; }
    if (!force && panelIntact()) { return true; }

    // Tear down observers bound to the previous, now-detached skeleton.
    if (observer) { observer.disconnect(); observer = null; }
    if (resizeObserver) { resizeObserver.disconnect(); resizeObserver = null; }

    root.classList.add(NS);
    root.innerHTML =
      '<div class="lb-header">' +
        '<div class="lb-title"><span>LoRAs</span>' +
          '<span class="lb-pill" data-lb="count">0 active</span>' +
          '<span class="lb-pill" data-lb="phase"></span>' +
        '</div>' +
        '<div class="lb-actions">' +
          '<select class="lb-select" data-lb="sort" aria-label="Sort LoRAs" title="Sort order">' +
            '<option value="name">Name A-Z</option>' +
            '<option value="civitai">Civitai name</option>' +
            '<option value="recent">Recently added</option>' +
            '<option value="active">Active first</option>' +
            '<option value="favorite">Favourites first</option>' +
          '</select>' +
          '<button type="button" class="lb-btn lb-icon" data-lb="naming" ' +
            'title="Switch between Civitai names and filenames">Aa</button>' +
          '<select class="lb-select" data-lb="profiles" aria-label="Stack profiles"></select>' +
          '<button type="button" class="lb-btn" data-lb="save">Save as...</button>' +
          '<button type="button" class="lb-btn lb-icon" data-lb="manage" title="Manage profiles" aria-label="Manage profiles">\u22EF</button>' +
          '<button type="button" class="lb-btn" data-lb="refresh">Refresh</button>' +
        '</div>' +
      '</div>' +
      '<div class="lb-toolbar">' +
        '<input class="lb-input lb-search" type="search" data-lb="search" ' +
          'placeholder="Filter - or kw:trigger, name:civitai, file:name, tag:x" ' +
          'title="Free text searches names, trigger words and tags. Prefix a term with kw:, name:, file: or tag: to target one field." ' +
          'aria-label="Filter LoRAs">' +
        '<div class="lb-zoom">' +
          '<button type="button" class="lb-btn lb-icon" data-lb="zoom-out" aria-label="Smaller thumbnails">-</button>' +
          '<input type="range" data-lb="zoom" min="72" max="176" step="1" aria-label="Thumbnail size">' +
          '<button type="button" class="lb-btn lb-icon" data-lb="zoom-in" aria-label="Larger thumbnails">+</button>' +
          '<span class="lb-zoom-value" data-lb="zoom-value">100%</span>' +
        '</div>' +
      '</div>' +
      '<div class="lb-grid-wrap" data-lb="grid-wrap"><div class="lb-grid" data-lb="grid" role="listbox" aria-multiselectable="true"></div></div>' +
      '<div class="lb-active-head"><span data-lb="active-title">Active LoRAs (0)</span>' +
        '<span class="lb-actions">' +
          '<span class="lb-viewsw" data-lb="viewsw" role="radiogroup" aria-label="Show the active LoRAs as">' +
            '<button type="button" role="radio" data-view="list" title="One row per LoRA; open one to edit it">List</button>' +
            '<button type="button" role="radio" data-view="cards" title="Every LoRA\'s editor, open">Cards</button>' +
          '</span>' +
          '<button type="button" class="lb-btn" data-lb="select" aria-pressed="false" ' +
            'title="Pick several LoRAs to change or group together">Select</button>' +
          '<button type="button" class="lb-btn" data-lb="av" aria-haspopup="menu" hidden ' +
            'title="Make every LoRA visual only, audio only, or normal again">Audio / visual \u25BE</button>' +
          '<button type="button" class="lb-btn" data-lb="collapse-all" disabled ' +
            'title="Close every open schedule - nothing in them changes">Collapse all</button>' +
          '<button type="button" class="lb-btn" data-lb="paste" aria-haspopup="dialog" aria-expanded="false" ' +
            'title="Load a LoRA stack from text - one name: x multiplier per line">Paste stack</button>' +
          '<button type="button" class="lb-btn" data-lb="disable-all" ' +
            'title="Switch every LoRA off - each keeps its strength and schedule">Disable all</button>' +
          '<button type="button" class="lb-btn" data-lb="enable-all" disabled ' +
            'title="Switch every LoRA back on">Enable all</button>' +
        '</span>' +
      '</div>' +
      '<div class="lb-buildbar" data-lb="buildbar" role="status" hidden></div>' +
      '<div class="lb-bulk" data-lb="bulk" hidden></div>' +
      '<div class="lb-rows" data-lb="rows"></div>' +
      '<div class="lb-statusbar">' +
        '<span class="lb-status" data-lb="status"></span>' +
        '<button type="button" class="lb-built-link" data-lb="built" hidden ' +
          'title="The visual-only and audio-only copies and the baked groups this folder holds">Built files</button>' +
        '<button type="button" class="lb-native-toggle" data-lb="native" ' +
          'title="Show or hide WanGP\'s built-in LoRA controls">native controls</button>' +
      '</div>';

    var pick = function (name) { return root.querySelector('[data-lb="' + name + '"]'); };
    el = {
      root: root, count: pick("count"), phase: pick("phase"), profiles: pick("profiles"),
      sort: pick("sort"), naming: pick("naming"),
      save: pick("save"), manage: pick("manage"), refresh: pick("refresh"),
      search: pick("search"), zoom: pick("zoom"), zoomValue: pick("zoom-value"),
      zoomIn: pick("zoom-in"), zoomOut: pick("zoom-out"),
      gridWrap: pick("grid-wrap"), grid: pick("grid"),
      activeTitle: pick("active-title"), collapseAll: pick("collapse-all"),
      paste: pick("paste"), disableAll: pick("disable-all"),
      enableAll: pick("enable-all"), rows: pick("rows"), status: pick("status"),
      nativeToggle: pick("native"),
      viewSwitch: pick("viewsw"), select: pick("select"), av: pick("av"),
      buildBar: pick("buildbar"), bulk: pick("bulk"), built: pick("built")
    };

    var sharedForm = root.parentElement && root.closest(".form");
    if (sharedForm) { sharedForm.classList.add("wgp-lora-browser-form-host"); }

    wireHeader();
    wireGrid();
    observeVisibility();
    observeResize();
    watchAnchor();
    S.mounted = true;
    return true;
  }

  /* Recover from a wipe as soon as it happens, rather than waiting for the
     next native change event to push a payload. */
  function watchAnchor() {
    var root = byId(IDS.root);
    var holder = root && root.parentElement;
    if (!holder) { return; }
    if (anchorObserver) { anchorObserver.disconnect(); }
    if (typeof MutationObserver !== "function") { return; }
    anchorObserver = new MutationObserver(function () { scheduleRemount(); });
    anchorObserver.observe(holder, { childList: true, subtree: true });
  }

  function scheduleRemount() {
    if (S.remountQueued) { return; }
    S.remountQueued = true;
    requestAnimationFrame(function () {
      S.remountQueued = false;
      checkIntegrity();
    });
  }

  /* If the skeleton is gone, rebuild it and repaint from cached state.  If it
     cannot be rebuilt, give the native controls back rather than leaving the
     user with no LoRA UI at all. */
  function checkIntegrity() {
    if (panelIntact()) { return; }
    if (!mount(true)) {
      hideNative(false);
      return;
    }
    try {
      S.applyZoom(S.zoomPx, false);
      renderHeader();
      renderGrid();
      renderRows(true);
      setStatus(S.status, S.statusWarn);
      hideNative(S.ready && panelHasContent());
    } catch (error) {
      console.error("[LoRA Browser] remount failed", error);
      hideNative(false);
    }
  }

  function wireHeader() {
    el.search.addEventListener("input", function () {
      S.search = el.search.value;
      S.searchTerms = parseSearch(S.search);
      renderGrid();
    });

    var applyZoom = function (value, persist) {
      S.zoomPx = Math.max(72, Math.min(176, Math.round(Number(value) || 104)));
      el.zoom.value = String(S.zoomPx);
      el.zoomValue.textContent = Math.round((S.zoomPx / 104) * 100) + "%";
      el.root.style.setProperty("--lb-tile", S.zoomPx + "px");
      // Zoom is pure CSS reflow: never round-trip to the backend for it.
      requestAnimationFrame(capHeight);
      if (persist) { send({ type: "zoom", value: S.zoomPx }); }
    };
    S.applyZoom = applyZoom;

    el.zoom.addEventListener("input", function () { applyZoom(el.zoom.value, false); });
    el.zoom.addEventListener("change", function () { applyZoom(el.zoom.value, true); });
    el.zoomIn.addEventListener("click", function () { applyZoom(S.zoomPx + 8, true); });
    el.zoomOut.addEventListener("click", function () { applyZoom(S.zoomPx - 8, true); });

    el.sort.addEventListener("change", function () {
      S.sortMode = el.sort.value;
      renderGrid();                       // reorder locally, no backend call
      send({ type: "sort", value: S.sortMode });
    });

    el.naming.addEventListener("click", function () {
      S.nameMode = S.nameMode === "civitai" ? "file" : "civitai";
      renderHeader();
      renderGrid();
      renderRows(true);
      send({ type: "name_mode", value: S.nameMode });
    });

    el.refresh.addEventListener("click", function () {
      el.refresh.disabled = true;
      setStatus("Refreshing...");
      // Reuse WanGP's own refresh so newly copied files become selectable.
      if (!clickHidden(IDS.refreshBtn)) {
        el.refresh.disabled = false;
        setStatus("Refresh is unavailable in this WanGP build.", true);
      }
      setTimeout(function () { el.refresh.disabled = false; }, 4000);
    });

    el.profiles.addEventListener("change", function () {
      var name = el.profiles.value;
      if (!name) { return; }
      send({ type: "profile_recall", name: name });
    });

    el.save.addEventListener("click", function () {
      var name = window.prompt("Save the current LoRA stack as:", S.activeProfile || "");
      if (name === null) { return; }
      send({ type: "profile_save", name: name });
    });

    el.manage.addEventListener("click", function (event) {
      openMenu(event.clientX, event.clientY, profileMenuItems());
    });

    el.collapseAll.addEventListener("click", collapseAll);
    el.paste.addEventListener("click", togglePaste);
    el.disableAll.addEventListener("click", function () { send({ type: "disable_all" }); });
    el.enableAll.addEventListener("click", function () { send({ type: "enable_all" }); });

    el.viewSwitch.querySelectorAll("button").forEach(function (button) {
      button.addEventListener("click", function () { setView(button.dataset.view, true); });
    });
    el.select.addEventListener("click", function () { setSelecting(!S.selecting); });
    el.av.addEventListener("click", function (event) {
      var rect = el.av.getBoundingClientRect();
      openMenu(rect.left, rect.bottom + 4, avMenuItems());
    });
    el.built.addEventListener("click", openBuiltDialog);

    // Last-resort escape hatch. Automatic fail-open covers the failures we can
    // detect; this covers the ones we cannot.
    el.nativeToggle.addEventListener("click", function () {
      S.forceNative = !S.forceNative;
      el.nativeToggle.setAttribute("aria-pressed", S.forceNative ? "true" : "false");
      hideNative(S.ready);
    });
  }

  function profileMenuItems() {
    var selected = el.profiles.value;
    var isDefault = selected && selected === S.defaultProfile;
    return [
      { label: "Save as new profile...", run: function () {
          var name = window.prompt("Save the current LoRA stack as:", "");
          if (name === null) { return; }
          send({ type: "profile_save", name: name });
        } },
      { label: "Update to current stack", disabled: !selected, run: function () {
          send({ type: "profile_update", name: selected });
        } },
      { label: "Recall", disabled: !selected, run: function () {
          send({ type: "profile_recall", name: selected });
        } },
      { separator: true },
      { label: isDefault ? "Clear default for this model" : "Set as default for this model",
        disabled: !selected,
        run: function () { send({ type: "profile_default", name: selected }); } },
      { separator: true },
      { label: "Rename...", disabled: !selected, run: function () {
          var name = window.prompt("Rename profile:", selected);
          if (name === null) { return; }
          send({ type: "profile_rename", name: selected, new_name: name });
        } },
      { label: "Delete", disabled: !selected, run: function () {
          if (!window.confirm("Delete profile '" + selected + "'?")) { return; }
          send({ type: "profile_delete", name: selected });
        } },
      { separator: true },
      { label: S.civitaiKeySet ? "Civitai API key (set)..." : "Set Civitai API key...",
        run: function () {
          var key = window.prompt(
            "Civitai API key, used when fetching LoRA info.\n" +
            "Leave empty to clear it. Public models do not need one.\n" +
            "It is stored in the plugin's settings file in plain text; " +
            "export CIVITAI_API_KEY instead to keep it out of that file.",
            ""
          );
          if (key === null) { return; }
          send({ type: "civitai_key", value: key });
        } },
      { separator: true },
      bulkFetchMenuItem()
    ];
  }

  /* One item, two jobs: start a run, or stop the one already going. The label
     carries the progress so the menu is also where you check on it. */
  function bulkFetchMenuItem() {
    var bulk = S.bulkFetch;
    if (bulk && bulk.running) {
      return { label: "Stop fetching (" + bulk.done + "/" + bulk.total + ")...",
        run: function () { requestBulkFetch("stop"); } };
    }
    return { label: "Fetch all missing info...",
      run: function () { requestBulkFetch("start"); } };
  }

  /* Fetching every missing catalogue is minutes of hashing and downloading, so
     the server runs it on a worker thread and we poll for progress. The status
     line is the progress bar. */
  function requestBulkFetch(action) {
    if (action === "start") { setStatus("Looking for LoRAs with no catalogue..."); }
    requestMedia({ kind: "fetch_all", action: action }, function (response) {
      if (!response) { return; }
      S.bulkFetch = response;
      if (response.running) {
        showBulkProgress(response);
        if (!S.bulkTimer) { S.bulkTimer = setTimeout(pollBulkFetch, 1500); }
        return;
      }
      finishBulkFetch(response);
    });
  }

  function pollBulkFetch() {
    S.bulkTimer = null;
    requestMedia({ kind: "fetch_all", action: "status" }, function (response) {
      if (!response) { return; }
      S.bulkFetch = response;
      if (response.running) {
        showBulkProgress(response);
        S.bulkTimer = setTimeout(pollBulkFetch, 1500);
        return;
      }
      finishBulkFetch(response);
    });
  }

  function showBulkProgress(bulk) {
    // Cancelling is cooperative: the LoRA in flight is allowed to finish so it
    // never leaves a half-written folder behind.
    if (bulk.cancelled) { setStatus("Stopping after the current LoRA..."); return; }
    setStatus("Fetching catalogues " + bulk.done + "/" + bulk.total +
      (bulk.current ? " - " + bulk.current : "") + "...");
  }

  function finishBulkFetch() {
    if (S.bulkTimer) { clearTimeout(S.bulkTimer); S.bulkTimer = null; }
    // New previews and Civitai names only reach the grid on the next payload,
    // and tiles that had no preview were cached as having none.
    S.thumbs = {};
    S.requested = {};
    // The server left the outcome as this instance's status, and a fresh
    // payload is what both collects it and redraws the enriched tiles.
    send({ type: "ready" });
  }

  /* ------------------------------------------------------------- grid */

  function wireGrid() {
    el.grid.addEventListener("click", function (event) {
      var tile = event.target.closest(".lb-tile");
      if (!tile || !el.grid.contains(tile)) { return; }
      toggle(tile.dataset.id);
    });

    el.grid.addEventListener("keydown", function (event) {
      if (event.key !== "Enter" && event.key !== " ") { return; }
      // A tile's own buttons (Inspect, play) handle their own keys; reaching
      // them with the keyboard must not also toggle the LoRA.
      if (event.target.closest("button")) { return; }
      var tile = event.target.closest(".lb-tile");
      if (!tile) { return; }
      event.preventDefault();
      toggle(tile.dataset.id);
    });

    el.grid.addEventListener("contextmenu", function (event) {
      var tile = event.target.closest(".lb-tile");
      if (!tile) { return; }
      event.preventDefault();
      openMenu(event.clientX, event.clientY, tileMenuItems(tile.dataset.id));
    });

    var pressTimer = null;
    el.grid.addEventListener("touchstart", function (event) {
      var tile = event.target.closest(".lb-tile");
      if (!tile) { return; }
      var touch = event.touches[0];
      pressTimer = setTimeout(function () {
        pressTimer = null;
        openMenu(touch.clientX, touch.clientY, tileMenuItems(tile.dataset.id));
      }, 550);
    }, { passive: true });
    var cancelPress = function () { if (pressTimer) { clearTimeout(pressTimer); pressTimer = null; } };
    el.grid.addEventListener("touchend", cancelPress);
    el.grid.addEventListener("touchmove", cancelPress);

    el.gridWrap.addEventListener("scroll", closeMenu, { passive: true });
  }

  function toggle(id) {
    var item = S.byId[id];
    if (!item) { return; }
    send({ type: "toggle", id: id, enabled: !item.active });
  }

  function tileMenuItems(id) {
    var item = S.byId[id] || {};
    return [
      // Never disabled: with no catalogue yet, Inspect is where you fetch one.
      { label: item.has_catalogue ? "Inspect" : "Inspect / fetch info...",
        run: function () { openInspect(id); } },
      { separator: true },
      { label: item.favorite ? "Unfavorite" : "Favorite", run: function () { send({ type: "favorite", id: id, value: !item.favorite }); } },
      { label: item.tag ? "Edit tag..." : "Add tag...", run: function () {
          var value = window.prompt("Tag for " + item.name + ":", item.tag || "");
          if (value === null) { return; }
          send({ type: "tag", id: id, value: value });
        } },
      { separator: true },
      { label: "Copy filename", run: function () { copyText(item.name || ""); } },
      item.disabled
        ? { label: "Switch on", run: function () { send({ type: "set_enabled", id: id, value: true }); } }
        : { label: item.active ? "Exclude LoRA" : "Include LoRA", run: function () { send({ type: "toggle", id: id, enabled: !item.active }); } }
    ];
  }

  /* ``label`` names what was copied in the status line, for text too long to
     repeat there. */
  function copyText(value, label) {
    // Only the basename is ever copied; absolute paths stay on the Python side.
    var done = function () { setStatus("Copied " + (label || value)); };
    var fallback = function () {
      var field = document.createElement("textarea");
      field.value = value;
      document.body.appendChild(field);
      field.select();
      try { document.execCommand("copy"); done(); } catch (error) { /* ignore */ }
      document.body.removeChild(field);
    };
    // Plain http on a LAN address has no async clipboard, and a refusal is
    // still worth one more try the old way.
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(value).then(done, fallback);
      return;
    }
    fallback();
  }

  /* Civitai names are usually more coherent than the downloaded filename, so
     they win when the enrichment script supplied one. */
  function displayName(item) {
    if (S.nameMode === "civitai" && item.civitai_name) { return item.civitai_name; }
    return item.label || item.name;
  }

  function searchFields(item) {
    return {
      file: (item.name || "") + " " + (item.parent || ""),
      name: item.civitai_name || "",
      kw: (item.words || []).join(" "),
      tag: item.tag || ""
    };
  }

  /* Free text matches every field. A "field:term" prefix restricts the term to
     one field, so `kw:zoom` finds trigger words without matching filenames that
     merely contain "zoom". Terms are ANDed. */
  function matchesSearch(item) {
    if (!S.searchTerms || !S.searchTerms.length) { return true; }
    var fields = searchFields(item);
    var all = (fields.file + " " + fields.name + " " + fields.kw + " " + fields.tag).toLowerCase();

    return S.searchTerms.every(function (term) {
      if (term.field) {
        return (fields[term.field] || "").toLowerCase().indexOf(term.text) >= 0;
      }
      return all.indexOf(term.text) >= 0;
    });
  }

  var SEARCH_FIELDS = { kw: "kw", key: "kw", word: "kw", trigger: "kw",
                        name: "name", civitai: "name", file: "file",
                        filename: "file", tag: "tag" };

  function parseSearch(text) {
    var terms = [];
    String(text || "").trim().toLowerCase().split(/\s+/).forEach(function (chunk) {
      if (!chunk) { return; }
      var split = chunk.indexOf(":");
      if (split > 0) {
        var field = SEARCH_FIELDS[chunk.slice(0, split)];
        var value = chunk.slice(split + 1);
        if (field && value) { terms.push({ field: field, text: value }); return; }
      }
      terms.push({ field: null, text: chunk });
    });
    return terms;
  }

  function sortItems(items) {
    var mode = S.sortMode;
    var byName = function (a, b) {
      return displayName(a).localeCompare(displayName(b), undefined, { sensitivity: "base" });
    };
    var copy = items.slice();

    if (mode === "recent") {
      copy.sort(function (a, b) { return (b.mtime || 0) - (a.mtime || 0) || byName(a, b); });
    } else if (mode === "active") {
      copy.sort(function (a, b) { return (b.active ? 1 : 0) - (a.active ? 1 : 0) || byName(a, b); });
    } else if (mode === "favorite") {
      copy.sort(function (a, b) { return (b.favorite ? 1 : 0) - (a.favorite ? 1 : 0) || byName(a, b); });
    } else if (mode === "civitai") {
      // LoRAs with no catalogue entry sink below the named ones.
      copy.sort(function (a, b) {
        var an = a.civitai_name || "", bn = b.civitai_name || "";
        if (!an !== !bn) { return an ? -1 : 1; }
        return (an || a.name).localeCompare(bn || b.name, undefined, { sensitivity: "base" });
      });
    } else {
      copy.sort(byName);
    }
    return copy;
  }

  function renderGrid() {
    if (!el.grid) { return; }
    var visible = sortItems(S.items.filter(matchesSearch));

    if (!visible.length) {
      el.grid.innerHTML = "";
      var empty = document.createElement("div");
      empty.className = "lb-empty";
      empty.textContent = S.items.length
        ? "No LoRAs match this filter."
        : "No LoRAs available for this model yet. Copy files into the model's LoRA folder and press Refresh.";
      el.grid.appendChild(empty);
      capHeight();
      return;
    }

    var fragment = document.createDocumentFragment();
    visible.forEach(function (item) { fragment.appendChild(buildTile(item)); });
    el.grid.innerHTML = "";
    el.grid.appendChild(fragment);
    capHeight();
    scheduleThumbnails();
  }

  function buildTile(item) {
    var tile = document.createElement("div");
    tile.className = "lb-tile" + (item.disabled ? " lb-tile-off" : "");
    tile.dataset.id = item.id;
    tile.tabIndex = 0;
    tile.setAttribute("role", "option");
    tile.setAttribute("aria-selected", item.active ? "true" : "false");
    var tooltip = [displayName(item), item.name];
    if (item.disabled) { tooltip.push("Switched off - click to switch it back on"); }
    if (item.parent) { tooltip.push(item.parent); }
    if ((item.words || []).length) { tooltip.push("Triggers: " + item.words.join(", ")); }
    tile.title = tooltip.filter(Boolean).join("\n");

    tile.appendChild(thumbMedia(item.id));

    var badges = document.createElement("div");
    badges.className = "lb-badges";
    var check = document.createElement("span");
    if (item.active || item.disabled) {
      // In the stack either way; hollow while it is switched off.
      check.className = "lb-check" + (item.disabled ? " lb-check-off" : "");
      check.textContent = "\u2713";
    }
    badges.appendChild(check);
    if (item.favorite) {
      var star = document.createElement("span");
      star.className = "lb-star";
      star.textContent = "\u2605";
      star.title = "Favourite";
      badges.appendChild(star);
    }
    tile.appendChild(badges);

    // The tile's own action is Inspect, not Favorite: it is the one thing worth
    // a dedicated target, and Favorite stays in the Inspector and context menu.
    var inspect = document.createElement("button");
    inspect.type = "button";
    inspect.className = "lb-btn lb-tile-inspect";
    inspect.textContent = "i";
    inspect.title = (item.has_catalogue ? "Inspect " : "Inspect / fetch info for ")
      + displayName(item);
    inspect.setAttribute("aria-label", inspect.title);
    inspect.addEventListener("click", function (event) {
      event.stopPropagation();          // inspect, do not toggle the LoRA
      event.preventDefault();
      openInspect(item.id);
    });
    tile.appendChild(inspect);

    // A tile whose preview is a video gets a play control. The still frame
    // stays the thumbnail; the video itself is only fetched on click.
    if (item.has_video) { tile.appendChild(playButton(item.id, tile, displayName(item))); }

    // textContent, never innerHTML: filenames and tags are user-controlled.
    var name = document.createElement("div");
    name.className = "lb-tile-name";
    name.textContent = displayName(item);
    tile.appendChild(name);

    if (item.tag) {
      var tag = document.createElement("span");
      tag.className = "lb-tag";
      tag.textContent = item.tag;
      tile.appendChild(tag);
    }
    return tile;
  }

  /* A tile's or a row's picture: the cached preview, or the placeholder a
     LoRA with none shows. */
  function thumbMedia(id) {
    var thumb = S.thumbs[id];
    if (thumb) {
      var image = document.createElement("img");
      image.className = "lb-thumb";
      image.loading = "lazy";
      image.alt = "";
      image.src = thumb;
      return image;
    }
    var placeholder = document.createElement("div");
    placeholder.className = "lb-thumb lb-placeholder";
    placeholder.setAttribute("aria-hidden", "true");
    placeholder.textContent = "\u25A6";
    return placeholder;
  }

  /* Plays ``holder``'s preview video in place of its still, or stops it. */
  function playButton(id, holder, name) {
    var play = document.createElement("button");
    play.type = "button";
    play.className = "lb-play";
    play.title = "Play preview";
    play.setAttribute("aria-label", "Play preview for " + name);
    play.textContent = "\u25B6";
    play.addEventListener("click", function (event) {
      event.stopPropagation();          // play, do not toggle or inspect the LoRA
      togglePreviewVideo(id, holder);
    });
    return play;
  }

  /* Swap a tile's still frame for a muted looping video, and back again.
     Only one plays at a time and the bytes are fetched on demand, so a library
     full of video previews costs nothing until something is clicked. A row's
     picture plays the same way. */
  function togglePreviewVideo(id, tile) {
    var existing = tile.querySelector("video.lb-thumb");
    if (existing) { stopPreviewVideo(tile); return; }

    if (S.playingTile && S.playingTile !== tile) { stopPreviewVideo(S.playingTile); }
    S.playingTile = tile;
    tile.classList.add("lb-loading");

    requestMedia({ kind: "video", id: id }, function (response) {
      tile.classList.remove("lb-loading");
      if (!response || !response.data) {
        setStatus((response && response.error) || "Preview video unavailable.", true);
        S.playingTile = null;
        return;
      }
      var slot = tile.querySelector(".lb-thumb");
      if (!slot) { return; }
      var video = document.createElement("video");
      video.className = "lb-thumb";
      video.muted = true;               // never any audio
      video.loop = true;
      video.autoplay = true;
      video.playsInline = true;
      video.setAttribute("muted", "");
      video.setAttribute("playsinline", "");
      video.src = response.data;
      slot.replaceWith(video);
      tile.classList.add("lb-playing");
      var play = tile.querySelector(".lb-play");
      if (play) { play.textContent = "\u23F8"; play.title = "Pause preview"; }
      video.play().catch(function () { /* autoplay refusal is not fatal */ });
    });
  }

  function stopPreviewVideo(tile) {
    var video = tile.querySelector("video.lb-thumb");
    if (video) {
      video.pause();
      video.replaceWith(thumbMedia(tile.dataset.id));
    }
    tile.classList.remove("lb-playing", "lb-loading");
    var play = tile.querySelector(".lb-play");
    if (play) { play.textContent = "\u25B6"; play.title = "Play preview"; }
    if (S.playingTile === tile) { S.playingTile = null; }
  }

  /* Cap the browser at three rows: fewer rows shrink to fit, a fourth scrolls. */
  function capHeight() {
    if (!el.grid || !el.gridWrap) { return; }
    var tiles = el.grid.querySelectorAll(".lb-tile");
    if (!tiles.length) {
      el.gridWrap.style.height = "auto";
      el.gridWrap.style.maxHeight = "none";
      return;
    }

    var styles = window.getComputedStyle(el.grid);
    var columns = styles.gridTemplateColumns.split(" ").filter(function (part) { return part; }).length || 1;
    var gap = parseFloat(styles.rowGap) || 6;
    var tileHeight = tiles[0].getBoundingClientRect().height || S.zoomPx;
    var rowCount = Math.ceil(tiles.length / columns);
    var visibleRows = Math.min(rowCount, MAX_VISIBLE_ROWS);
    // The height is set on the box, so its own padding and border have to be in
    // it -- measured, not assumed, because a host stylesheet decides both.
    var wrapStyles = window.getComputedStyle(el.gridWrap);
    var chrome = (parseFloat(wrapStyles.paddingTop) || 0) + (parseFloat(wrapStyles.paddingBottom) || 0) +
      (parseFloat(wrapStyles.borderTopWidth) || 0) + (parseFloat(wrapStyles.borderBottomWidth) || 0);
    var height = visibleRows * tileHeight + (visibleRows - 1) * gap + chrome;

    el.gridWrap.style.height = Math.ceil(height) + "px";
    el.gridWrap.style.maxHeight = Math.ceil(height) + "px";
    el.gridWrap.style.overflowY = rowCount > MAX_VISIBLE_ROWS ? "auto" : "hidden";
  }

  function observeResize() {
    var onResize = function () {
      capHeight();
      // Only a change of width refits the picture column. A row's height
      // changes with what it shows -- and the column's width can decide
      // whether a chip strip needs a scrollbar -- so following heights could
      // see-saw between two sizes.
      var width = el.rows ? el.rows.clientWidth : 0;
      if (width !== S.rowsWidth) {
        S.rowsWidth = width;
        fitThumbColumn();
      }
    };
    if (typeof ResizeObserver === "function") {
      resizeObserver = new ResizeObserver(onResize);
      resizeObserver.observe(el.gridWrap);
      resizeObserver.observe(el.rows);
      return;
    }
    window.addEventListener("resize", onResize);
  }

  /* ------------------------------------------------------- thumbnails */

  function observeVisibility() {
    if (typeof IntersectionObserver !== "function") { return; }
    observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) { return; }
        var id = entry.target.dataset.id;
        if (id && !S.thumbs[id] && !S.requested[id]) {
          S.requested[id] = true;
          S.pendingThumbs.push(id);
        }
      });
      flushThumbnails();
    }, { root: el.gridWrap, rootMargin: "200px 0px" });
  }

  function scheduleThumbnails() {
    if (!observer) {
      // No IntersectionObserver: ask for a bounded first screenful instead.
      S.items.filter(matchesSearch).slice(0, 60).forEach(function (item) {
        if (!S.thumbs[item.id] && !S.requested[item.id]) {
          S.requested[item.id] = true;
          S.pendingThumbs.push(item.id);
        }
      });
      flushThumbnails();
      return;
    }
    observer.disconnect();
    el.grid.querySelectorAll(".lb-tile").forEach(function (tile) { observer.observe(tile); });
  }

  function flushThumbnails() {
    if (!S.pendingThumbs.length) { return; }
    clearTimeout(S.thumbTimer);
    S.thumbTimer = setTimeout(function () {
      // Small batches on purpose: a first-time video preview costs a frame
      // decode server-side, so a large batch would stall behind the slowest
      // file. Smaller rounds let the grid fill in progressively.
      var batch = S.pendingThumbs.splice(0, 16);
      if (batch.length) { requestMedia({ kind: "thumbs", ids: batch }, null); }
    }, 90);
  }

  /* One hidden bridge serves thumbnails, catalogue detail and media bytes, so
     requests are queued and answered strictly in order. A dropped reply must
     not wedge the queue, hence the timeout. */
  function requestMedia(request, callback, timeoutMs) {
    S.mediaQueue.push({ request: request, callback: callback, timeout: timeoutMs || 30000 });
    pumpMedia();
  }

  function pumpMedia() {
    if (S.mediaInFlight || !S.mediaQueue.length) { return; }
    var job = S.mediaQueue.shift();
    // The server echoes this back, which keeps every reply's text unique:
    // Gradio raises no change event when a reply equals the one before it, so
    // a repeated answer would otherwise sit unheard until the timeout.
    S.mediaSeq += 1;
    job.request.seq = S.mediaSeq;
    S.mediaInFlight = job;
    if (!setHidden(IDS.thumbReq, JSON.stringify(job.request))) {
      S.mediaInFlight = null;
      return;
    }
    clickHidden(IDS.thumbBtn);
    S.mediaTimer = setTimeout(function () { finishMedia(null); }, job.timeout);
  }

  function finishMedia(response) {
    clearTimeout(S.mediaTimer);
    var job = S.mediaInFlight;
    S.mediaInFlight = null;
    if (job && job.callback) {
      try { job.callback(response); } catch (error) { console.error("[LoRA Browser]", error); }
    }
    pumpMedia();
  }

  function applyMedia(json) {
    var payload = null;
    try { payload = JSON.parse(json || "{}"); } catch (error) { payload = null; }
    if (payload && payload.kind === "thumbs") { applyThumbBatch(payload); }
    // A late reply to a request that already timed out is not the answer to
    // the one now in flight; thumbnails above are keyed by id, so still useful.
    var job = S.mediaInFlight;
    if (payload && payload.seq !== undefined && (!job || payload.seq !== job.request.seq)) { return; }
    finishMedia(payload);
  }

  function applyThumbBatch(payload) {
    var map = payload.thumbs || {};
    Object.keys(map).forEach(function (id) {
      if (map[id]) { S.thumbs[id] = map[id]; }
    });

    // Swap placeholders in place -- the tile's and the row's -- because
    // rebuilding the grid here would fight scroll.
    Object.keys(map).forEach(function (id) {
      if (!map[id]) { return; }
      var key = cssEscape(id);
      var holders = [el.grid.querySelector('.lb-tile[data-id="' + key + '"]')];
      if (el.rows) {
        el.rows.querySelectorAll('[data-thumb="' + key + '"]').forEach(function (node) { holders.push(node); });
      }
      if (S.dialog) {
        S.dialog.querySelectorAll('[data-thumb="' + key + '"]').forEach(function (node) { holders.push(node); });
      }
      holders.forEach(function (holder) {
        var placeholder = holder && holder.querySelector(".lb-thumb");
        if (!placeholder || placeholder.tagName !== "DIV") { return; }
        placeholder.replaceWith(thumbMedia(id));
      });
    });
    if (S.pendingThumbs.length) { flushThumbnails(); }
  }

  /* A row shows its LoRA's picture too, and a row's tile can be scrolled out
     of view or filtered away -- so the rows ask for theirs themselves. */
  function requestRowThumbs() {
    var ids = [];
    S.rows.forEach(function (row) {
      ids.push(sourceId(row));
      (row.members || []).forEach(function (member) { ids.push(sourceId(member)); });
    });
    ids.forEach(function (id) {
      if (!S.byId[id] || S.thumbs[id] || S.requested[id]) { return; }
      S.requested[id] = true;
      S.pendingThumbs.push(id);
    });
    flushThumbnails();
  }

  function cssEscape(value) {
    if (window.CSS && window.CSS.escape) { return window.CSS.escape(value); }
    return String(value).replace(/["\\]/g, "\\$&");
  }

  /* ------------------------------------------------------ active rows */

  /* Every row has the same frame, whatever it is doing:
   *
   *   [on] LoRA name                       [Schedule v]  [i]  [x]
   *        [Phase 1 0.71] [Phase 2 1.23] [Independent]          (more than one phase)
   *        [-]  --------- slider ---------  [0.71]  [+]
   *
   * and scheduling only changes what sits above the strength control:
   *
   *   [on] LoRA name                     [~ Schedule ^]  [i]  [x]
   *        [Phase 1 ~2] [Phase 2 1.23]
   *        [Full screen] [Clear schedule]             [Delete]
   *        timeline -- tap or drag on empty space to draw
   *        [-]  --------- slider ---------  [0.9]  [+]    the selected region
   *
   * The strength control is the same control in both: it edits the plain
   * strength, or the selected region -- never a number WanGP is not applying.
   * A scheduled phase is its regions and every slot they do not cover is zero,
   * so there is no plain strength left for it to show. Before anything is drawn
   * it still edits the strength the phase is worth, and new regions start at.
   *
   * A switched-off row keeps everything and shows what switching it back on
   * brings back; nothing in it is editable until it is on.
   *
   * Selected phase and expansion are view state held here; regions are
   * canonical state that Python validates. Collapsing is a view change only --
   * it never deletes a schedule, and it never brings the plain control back for
   * a phase that has one. */

  function isOn(row) { return row.enabled !== false; }

  /* What a row can edit: nothing while it is switched off, preserved syntax,
     or a schedule this guidance mode cannot show truthfully. */
  function rowEditable(row) {
    if (!isOn(row) || row.multiplier_kind === "advanced") { return false; }
    return !(row.multiplier_kind === "scheduled" && !S.schedulingEnabled);
  }

  function phaseOf(row) {
    var count = Math.max(1, (row.phase_values || []).length);
    var phase = S.selectedPhaseById[row.id];
    if (phase === undefined || phase < 0 || phase >= count) { phase = 0; }
    return phase;
  }

  function scheduleOf(row, phase) {
    var list = row.phase_schedules || [];
    var index = row.schedule_shared ? 0 : phase;
    return list[index] || null;
  }

  function hasActiveSchedule(row) {
    return (row.phase_schedules || []).some(function (schedule) { return schedule && schedule.active; });
  }

  function regionKey(row, phase) {
    return row.id + ":" + (row.schedule_shared ? "shared" : phase);
  }

  /* The payload's selection wins whenever the local one has gone stale, which
     is how a freshly drawn region arrives already selected. */
  function selectedRegion(row, phase, schedule) {
    if (!schedule) { return null; }
    var regions = schedule.regions || [];
    var wanted = S.selectedRegionByKey[regionKey(row, phase)];
    var found = null;
    regions.forEach(function (region) {
      if (region.id === wanted) { found = region; }
    });
    if (found) { return found; }
    regions.forEach(function (region) {
      if (region.id === schedule.selected_region_id) { found = region; }
    });
    return found || regions[0] || null;
  }

  function rowsKey() {
    return S.rows.map(function (row) {
      var phase = phaseOf(row);
      var schedule = scheduleOf(row, phase);
      var shape = schedule
        ? schedule.slots + "/" + schedule.editable + "/" + (schedule.drawn_past_end || 0) + "/" +
          (schedule.regions || []).map(function (region) {
            return region.id + "@" + region.start + "-" + region.end;
          }).join(",")
        : "-";
      // Shared is structure too: the chip strip differs, and a list with no
      // ';' turning into one per stage keeps the same stage-1 shape.
      // So is what "Fill from stage 1" offers: its tooltip names the values.
      var fill = row.stage_2_fill;
      // Which phases are scheduled: each one's pill wears a badge for it.
      var scheduled = (row.phase_schedules || []).map(function (each) {
        return each && each.active ? "1" : "0";
      }).join("");
      return [
        row.id, rowName(row), row.multiplier_kind, (row.phase_values || []).length,
        row.missing ? "m" : "", row.system_managed ? "s" : "",
        isOn(row) ? "on" : "off:" + (row.summary || ""),
        phase, S.scheduleOpenById[row.id] ? "open" : "shut", shape,
        row.schedule_shared ? "shared" : "own", scheduled,
        fill ? (fill.empty ? "fill-none" : "fill-" + fill.values.join(",")) : "",
        // What a list cell is, though not the values it shows.
        (row.phase_summary || []).map(function (cell) {
          return cell.kind === "steps" ? "s" + cell.steps : cell.kind;
        }).join(","),
        row.kind || "", row.mode || "", buildState(row.build), row.skipped ? "sk" : "", row.empty ? "e" : "",
        (row.members || []).map(function (member) {
          return [member.id, member.mode, buildState(member.build), member.enabled === false ? 0 : 1, rowName(member)].join("/");
        }).join(","),
        row.bake ? [buildState(row.bake), row.bake.rank, row.bake.kept, row.bake.default_rank].join("/") : ""
      ].join(":");
    }).join("|") + "#" + S.phases.effective + "#" + S.nameMode + "#" + S.steps
      + "#" + S.phaseSteps.join(",") + "#" + (S.schedulingEnabled ? "sched" : "nosched")
      + "#" + S.view + "#" + S.openRow + "#" + (S.avSupported ? "av" : "")
      + "#" + (S.selecting ? "sel:" + Object.keys(S.selected).sort().join(",") : "");
  }

  function renderRows(force) {
    if (!el.rows) { return; }
    // Never rebuild the DOM under a pointer that is dragging something.
    if (S.dragging || S.drag) { return; }

    var key = rowsKey();
    if (!force && key === S.rowsKey) {
      updateRowValues();
      return;
    }
    S.rowsKey = key;

    var fragment = document.createDocumentFragment();
    if (S.view === "list" && S.rows.length) {
      fragment.appendChild(buildList());
    } else {
      S.rows.forEach(function (row) { fragment.appendChild(buildRow(row)); });
    }
    el.rows.classList.toggle("lb-rows-list", S.view === "list");
    el.rows.innerHTML = "";
    if (!S.rows.length) {
      var empty = document.createElement("div");
      empty.className = "lb-empty";
      empty.textContent = "No LoRAs are active. Click a thumbnail to include one.";
      el.rows.appendChild(empty);
    } else {
      el.rows.appendChild(fragment);
    }
    fitThumbColumn();
    requestRowThumbs();
    paintCollapseAll();
  }

  /* Rows whose Schedule menu is open right now. */
  function openScheduleCount() {
    if (!S.schedulingEnabled) { return 0; }
    return S.rows.filter(function (row) {
      return !!S.scheduleOpenById[row.id] && rowEditable(row);
    }).length;
  }

  /* Every open Schedule menu closed in one go. A view change, as each row's own
     Schedule button is: every schedule stays exactly as it is. */
  function collapseAll() {
    S.scheduleOpenById = {};
    S.openRow = "";
    closeScheduleModal();
    renderRows(true);
  }

  function paintCollapseAll() {
    if (!el.collapseAll) { return; }
    var count = openScheduleCount();
    var open = S.view === "list" && !!S.openRow;
    el.collapseAll.disabled = !count && !open;
    el.collapseAll.title = count
      ? "Close the " + (count === 1 ? "open schedule" : count + " open schedules") + " - nothing in them changes"
      : open ? "Close the open LoRA - nothing in it changes" : "No schedule is open";
  }

  /* The picture column is as wide as a row at rest is tall -- a row showing
     its strengths, not an open timeline -- so such a row shows a square; and
     it is that wide on every row, so switches, names and controls stay in one
     column. A taller row (an open timeline) or a shorter one (switched off)
     crops its picture to its own height rather than pushing its controls
     sideways. A narrow panel gives the picture less of its width. */
  function fitThumbColumn() {
    if (!el.rows) { return; }
    var width = el.rows.clientWidth;
    if (!width) { return; }                  // a hidden tab measures nothing
    var rest = 0;
    el.rows.querySelectorAll(".lb-row").forEach(function (node) {
      if (node.querySelector(".lb-weight-main, .lb-sched-summary")) {
        var height = node.clientHeight;
        node.querySelectorAll(".lb-row-extra, .lb-members").forEach(function (part) { height -= part.offsetHeight; });
        rest = Math.max(rest, height);
      }
    });
    if (!rest) { return; }                   // nothing at rest: keep the last size
    var side = Math.round(Math.max(56, Math.min(rest, 168, width * 0.24)));
    var current = parseFloat(el.rows.style.getPropertyValue("--lb-thumb")) || 0;
    if (Math.abs(side - current) >= 2) { el.rows.style.setProperty("--lb-thumb", side + "px"); }
  }

  /* Repaint values without touching structure, for payloads that only moved a
     number (and so must not steal focus or interrupt a gesture). */
  function updateRowValues() {
    S.rows.forEach(function (row) {
      var node = el.rows.querySelector('.lb-row[data-id="' + cssEscape(row.id) + '"]');
      if (node && node.__lbPaint) { node.__lbPaint(row); }
      var item = el.rows.querySelector('.lb-li[data-id="' + cssEscape(row.id) + '"]');
      if (item && item.__lbPaint) { item.__lbPaint(row); }
    });
  }

  function buildRow(row) {
    var node = document.createElement("div");
    node.className = "lb-row" + (row.missing ? " lb-missing" : "") + (isOn(row) ? "" : " lb-row-off") +
      (row.kind === "group" ? " lb-row-group" : "");
    node.dataset.id = row.id;

    var phase = phaseOf(row);
    var schedule = scheduleOf(row, phase);
    var editable = rowEditable(row);
    var open = editable && !!S.scheduleOpenById[row.id] && S.schedulingEnabled;

    // The picture first, then everything the row edits in a column of its own.
    node.appendChild(buildRowThumb(row));
    var main = document.createElement("div");
    main.className = "lb-row-main";
    node.appendChild(main);

    main.appendChild(buildRowTop(row, open, editable));
    if (row.kind === "group") { main.appendChild(buildMembers(row)); }

    if (!isOn(row)) {
      main.appendChild(buildOffSummary(row));
      appendExtras(node, row);
      return node;
    }
    if (row.multiplier_kind === "advanced") {
      main.appendChild(buildPreserved(row));
      appendExtras(node, row);
      return node;
    }
    if (!editable) {
      // A schedule this guidance mode cannot show: kept as it is, read-only.
      main.appendChild(buildKeptSchedule(row));
      appendExtras(node, row);
      return node;
    }

    var painters = [];
    if ((row.phase_values || []).length > 1) {
      var strip = buildPhaseStrip(row, phase, open);
      main.appendChild(strip.node);
      painters.push(strip.paint);
    }

    if (open) {
      // Scheduler mode: the timeline, its tools, and the same strength control.
      var built = buildSchedule(row, phase, schedule);
      main.appendChild(built.node);
      if (built.paint) { painters.push(built.paint); }
    } else if (schedule && schedule.active) {
      // Collapsed, but this phase is still scheduled: say so instead of
      // offering a strength that is not the one being applied.
      main.appendChild(buildScheduleSummary(row, schedule));
    } else {
      var weight = buildWeightLine({
        value: currentValue(row, phase, schedule),
        label: (row.name + " " + (S.phaseLabels[phase] || "strength")).trim(),
        commit: function (value) { commitBase(row, phase, value); },
        settle: flushSync
      });
      weight.node.classList.add("lb-weight-main");
      main.appendChild(weight.node);
      painters.push(function (next) {
        var nextPhase = phaseOf(next);
        weight.paint(currentValue(next, nextPhase, scheduleOf(next, nextPhase)));
      });
    }

    appendExtras(node, row);
    node.__lbPaint = function (next) {
      painters.forEach(function (paint) { paint(next); });
    };
    return node;
  }

  function currentValue(row, phase, schedule) {
    var pending = S.localValues[baseKey(row.id, phase)];
    if (pending !== undefined) { return pending; }
    if (schedule) { return schedule.base; }
    var values = row.phase_values || [];
    return values[phase] !== undefined ? values[phase] : 1;
  }

  function regionValue(row, phase, region) {
    var pending = S.localValues[regionKeyFor(row.id, phase, region.id)];
    return pending !== undefined ? pending : region.strength;
  }

  /* An edit is the user's until WanGP has answered for it. Keys are per
     control so two of them cannot overwrite each other. */
  function baseKey(id, phase) { return id + "#" + phase; }
  function regionKeyFor(id, phase, regionId) { return id + "#" + phase + "#" + regionId; }

  /* The LoRA's picture, so a row is known at a glance: the tile's preview -- a
     video's first frame, with the tile's play control -- or the placeholder a
     LoRA with none shows, so every row keeps the same margin. Pressing it
     opens Inspect, as the "i" beside the name does. */
  function buildRowThumb(row) {
    var cell = document.createElement("div");
    cell.className = "lb-row-thumb";
    cell.dataset.id = row.id;

    if (row.kind === "group") {
      // A group shows its members' pictures, one above the other.
      cell.classList.add("lb-row-thumb-group");
      (row.members || []).slice(0, 3).forEach(function (member) {
        var part = document.createElement("div");
        part.className = "lb-thumb-part";
        part.dataset.thumb = sourceId(member);
        part.appendChild(thumbMedia(sourceId(member)));
        cell.appendChild(part);
      });
      return cell;
    }

    var source = sourceId(row);
    cell.dataset.thumb = source;
    var label = (row.has_catalogue ? "Inspect " : "Inspect / fetch info for ")
      + (row.civitai_name || row.name);
    var open = document.createElement("button");
    open.type = "button";
    open.className = "lb-row-thumb-open";
    open.title = label;
    open.setAttribute("aria-label", label);
    open.appendChild(thumbMedia(source));
    open.addEventListener("click", function () { openInspect(source); });
    cell.appendChild(open);

    var item = S.byId[source];
    if (item && item.has_video) { cell.appendChild(playButton(source, cell, rowName(row))); }
    return cell;
  }

  function buildRowTop(row, open, editable) {
    var top = document.createElement("div");
    top.className = "lb-row-top";

    // On/off first: the one thing worth reaching without looking. WanGP's own
    // accelerator LoRAs are its to manage, so they have no switch.
    if (!row.system_managed) {
      top.appendChild(powerSwitch(row));
    }

    // The name truncates before the controls do; the full one is in the tooltip.
    var name = document.createElement("div");
    name.className = "lb-row-name";
    name.textContent = rowName(row);
    name.title = [row.civitai_name, row.name].filter(Boolean).join("\n");
    if (row.kind === "group") {
      name.title = "Group " + row.group + ": " + (row.members || []).map(rowName).join(", ");
      name.appendChild(note((row.members || []).length + " LoRAs move as one"));
    }
    if (row.system_managed) { name.appendChild(note("WanGP-managed")); }
    if (row.missing) { name.appendChild(note("Missing locally")); }
    top.appendChild(name);

    if (editable && S.schedulingEnabled) {
      var scheduled = hasActiveSchedule(row);
      var toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "lb-btn lb-schedule-toggle" + (open ? " lb-on" : "") +
        (scheduled ? " lb-scheduled" : "");
      // The word gives way on a phone; the mark and the arrow do not.
      [["lb-sched-mark", "∿"], ["lb-sched-word", "Schedule"], ["lb-sched-arrow", open ? "▴" : "▾"]]
        .forEach(function (part) {
          var span = document.createElement("span");
          span.className = part[0];
          span.textContent = part[1];
          toggle.appendChild(span);
        });
      toggle.setAttribute("aria-label", scheduled ? "Schedule - this LoRA is scheduled" : "Schedule");
      toggle.title = open
        ? "Hide the timeline - the schedule is kept"
        : scheduled
          ? "Show this LoRA's schedule"
          : "Schedule this phase over the steps instead of one strength";
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
      toggle.addEventListener("click", function () {
        var next = !S.scheduleOpenById[row.id];
        S.scheduleOpenById[row.id] = next;
        // Opening a phase that has no schedule yet asks Python for one; it
        // creates the timeline without writing anything to WanGP.
        var phase = phaseOf(row);
        if (next && !scheduleOf(row, phase)) {
          send({ type: "schedule_enable", id: row.id, phase: phase });
          return;
        }
        renderRows(true);
      });
      top.appendChild(toggle);
    }

    if (row.kind === "group") {
      var ungroup = document.createElement("button");
      ungroup.type = "button";
      ungroup.className = "lb-btn";
      ungroup.textContent = "Ungroup";
      ungroup.title = "Make these separate LoRAs again; each keeps the group's strength and schedule";
      ungroup.addEventListener("click", function () { send({ type: "ungroup", group: row.group }); });
      top.appendChild(ungroup);
    } else {
      // One tap from every row to the full record.
      var inspect = document.createElement("button");
      inspect.type = "button";
      inspect.className = "lb-btn lb-icon lb-inspect";
      inspect.textContent = "i";
      inspect.title = (row.has_catalogue ? "Inspect " : "Inspect / fetch info for ")
        + (row.civitai_name || row.name);
      inspect.setAttribute("aria-label", inspect.title);
      inspect.addEventListener("click", function () { openInspect(sourceId(row)); });
      top.appendChild(inspect);
    }

    // Remove stays isolated at the far edge so it is never hit by accident.
    var remove = document.createElement("button");
    remove.type = "button";
    remove.className = "lb-btn lb-icon lb-remove";
    remove.textContent = "×";
    remove.title = "Remove " + rowName(row) + (isOn(row) ? "" : " - and forget its settings");
    remove.setAttribute("aria-label", "Remove " + rowName(row));
    remove.addEventListener("click", function () {
      send(row.kind === "group" ? { type: "remove", id: row.id, group: row.group } : { type: "remove", id: row.id });
    });
    top.appendChild(remove);

    if (S.view === "list") {
      var done = document.createElement("button");
      done.type = "button";
      done.className = "lb-btn lb-done";
      done.textContent = "Done \u25B4";
      done.title = "Back to the list";
      done.addEventListener("click", function () { S.openRow = ""; renderRows(true); });
      top.appendChild(done);
    }
    return top;
  }

  /* Off is not removed: the LoRA keeps its strength, schedule and place, and
     WanGP neither loads nor applies it. */
  function powerSwitch(row) {
    var on = isOn(row);
    var power = document.createElement("button");
    power.type = "button";
    power.className = "lb-switch" + (on ? " lb-on" : "");
    power.setAttribute("role", "switch");
    power.setAttribute("aria-checked", on ? "true" : "false");
    power.setAttribute("aria-label", "Apply " + rowName(row));
    power.title = on
      ? "Applied - switch off to skip it and keep its settings"
      : row.skipped && row.empty
        ? "Switched off: " + (MODE_LABEL[row.mode] || "this mode").toLowerCase() +
          " leaves it nothing to apply. Switch on to apply it as normal"
        : "Switched off - kept, not applied. Switch on to apply it again";
    power.addEventListener("click", function (event) {
      event.stopPropagation();
      send(row.kind === "group"
        ? { type: "set_enabled", id: row.id, value: !on, group: row.group }
        : { type: "set_enabled", id: row.id, value: !on });
    });
    return power;
  }

  /* What a switched-off LoRA brings back when it is switched on again. */
  function buildOffSummary(row) {
    var line = document.createElement("div");
    line.className = "lb-off-summary";
    var label = document.createElement("span");
    label.className = "lb-off-label";
    label.textContent = "Off";
    line.appendChild(label);
    var text = document.createElement("span");
    text.className = "lb-off-text";
    text.textContent = row.summary || row.multiplier_raw || "";
    text.title = "Kept while it is off: " + (row.multiplier_raw || "");
    line.appendChild(text);
    return line;
  }

  /* A schedule in a guidance mode that cannot show one truthfully: what it
     says, and why it waits for One Phase. Nothing here rewrites it. */
  function buildKeptSchedule(row) {
    var wrap = document.createElement("div");
    wrap.className = "lb-preserved";
    var summary = document.createElement("div");
    summary.className = "lb-kept-summary";
    summary.textContent = row.summary || row.multiplier_raw;
    summary.title = row.multiplier_raw || "";
    wrap.appendChild(summary);
    var why = document.createElement("div");
    why.className = "lb-preserved-note";
    why.textContent = row.schedule_reason || S.schedulingDisabledReason;
    wrap.appendChild(why);
    return wrap;
  }

  /* A multiplier this editor will not rewrite: branch syntax, or a phase
     structure only WanGP's own parser can expand. */
  function buildPreserved(row) {
    var wrap = document.createElement("div");
    wrap.className = "lb-preserved";

    var raw = document.createElement("code");
    raw.className = "lb-advanced";
    raw.title = "Preserved exactly as imported";
    raw.textContent = row.multiplier_raw;
    wrap.appendChild(raw);

    if (row.schedule_reason) {
      var why = document.createElement("div");
      why.className = "lb-preserved-note";
      why.textContent = row.schedule_reason;
      wrap.appendChild(why);
    }

    var convert = document.createElement("button");
    convert.type = "button";
    convert.className = "lb-btn";
    convert.textContent = "Use sliders";
    convert.title = "Replace this multiplier with a simple one";
    convert.addEventListener("click", function () {
      if (!window.confirm("Replace the multiplier '" + row.multiplier_raw + "' with a simple one?")) { return; }
      send({ type: "convert_simple", id: row.id });
    });
    wrap.appendChild(convert);
    return wrap;
  }

  /* The one mark for "this phase is scheduled" -- its strength changes from
     step to step, which is what the wave stands for -- wherever it shows: a
     phase chip, the Schedule button's wave, a collapsed schedule. The theme's
     text colour as a fill with the card colour on it, so it stands out from a
     plain value in any theme, whatever its accent. */
  function schedBadge(label) {
    var badge = document.createElement("span");
    badge.className = "lb-sched-badge";
    var wave = document.createElement("span");
    wave.className = "lb-wave";
    wave.setAttribute("aria-hidden", "true");
    wave.textContent = "\u223F";
    badge.appendChild(wave);
    badge.appendChild(document.createTextNode(label));
    return badge;
  }

  /* Phase chips: every effective phase, its current value, and which one the
     strength control and timeline are editing. */
  /* What a phase chip says it is worth. A scheduled phase has no single value:
     it is its regions, so the chip says it is scheduled rather than show a
     number that is not applied anywhere. */
  function chipReadout(row, index) {
    var phaseName = S.phaseLabels[index] || ("Phase " + (index + 1));
    var schedule = scheduleOf(row, index);
    if (schedule && schedule.active) {
      var count = (schedule.regions || []).length;
      return {
        scheduled: true,
        text: "Scheduled",
        title: phaseName + " is scheduled: its strength changes over the steps (" + count +
          (count === 1 ? " region" : " regions") + "; steps no region covers are 0)"
      };
    }
    var values = row.phase_values || [];
    var value = values[index] !== undefined ? values[index] : 1;
    return {
      scheduled: false,
      text: fmt(schedule ? schedule.base : value),
      title: phaseName + " is one strength over every step"
    };
  }

  function buildPhaseStrip(row, phase, scheduling) {
    var strip = document.createElement("div");
    strip.className = "lb-phase-strip";
    var values = row.phase_values || [];
    var chipValues = [];

    values.forEach(function (value, index) {
      var shown = chipReadout(row, index);
      var chip = document.createElement("button");
      chip.type = "button";
      chip.className = "lb-chip" + (index === phase ? " lb-on" : "") +
        (shown.scheduled ? " lb-chip-sched" : "");
      chip.setAttribute("aria-pressed", index === phase ? "true" : "false");
      chip.appendChild(document.createTextNode((S.phaseLabels[index] || ("Phase " + (index + 1))) + " "));
      // A plain value can change under a drag and is repainted in place; a
      // scheduled phase's badge only changes with the structure.
      var readout = shown.scheduled ? null : document.createElement("strong");
      if (readout) {
        readout.textContent = shown.text;
        chip.appendChild(readout);
      } else {
        chip.appendChild(schedBadge(shown.text));
      }
      chip.title = shown.title;
      chip.addEventListener("click", function () {
        if (S.selectedPhaseById[row.id] === index) { return; }
        S.selectedPhaseById[row.id] = index;
        // Switching chips is a view change: it must not rewrite a token.
        if (S.scheduleOpenById[row.id] && !scheduleOf(row, index)) {
          send({ type: "schedule_enable", id: row.id, phase: index });
          return;
        }
        renderRows(true);
      });
      strip.appendChild(chip);
      chipValues.push(readout);
    });

    if (row.schedule_shared && values.length > 1) {
      var sharedNote = document.createElement("span");
      sharedNote.className = "lb-chip lb-chip-static";
      sharedNote.textContent = "≡ Shared schedule";
      sharedNote.title = S.phaseSteps.length > 1
        ? "One comma schedule with no ';' runs in both stages: stretched over each stage's steps"
        : "One comma schedule spans the whole run, so every phase uses it";
      strip.appendChild(sharedNote);
    } else if (values.length > 1 && !scheduling) {
      // Linking acts on the plain strength control, which scheduler mode does
      // not have, so it is not offered there.
      var linked = !!S.linked[row.id];
      var link = document.createElement("button");
      link.type = "button";
      link.className = "lb-chip" + (linked ? " lb-on" : "");
      link.textContent = linked ? "⚭ Linked" : "⚮ Independent";
      link.title = linked
        ? "One value for every phase - schedules stay separate"
        : "Each phase keeps its own value";
      link.setAttribute("aria-pressed", linked ? "true" : "false");
      link.addEventListener("click", function () {
        S.linked[row.id] = !S.linked[row.id];
        renderRows(true);
      });
      strip.appendChild(link);
    }

    return {
      node: strip,
      paint: function (next) {
        (next.phase_values || []).forEach(function (value, index) {
          var shown = chipReadout(next, index);
          if (!chipValues[index] || shown.scheduled) { return; }
          chipValues[index].textContent = shown.text;
        });
      }
    };
  }

  /* What a scheduled phase shows when its timeline is collapsed: what it is
     doing, and a way back in. Deliberately not a control -- the strength this
     phase applies varies over the run, so there is no one number to set. */
  function buildScheduleSummary(row, schedule) {
    var regions = schedule.regions || [];
    var summary = document.createElement("button");
    summary.type = "button";
    summary.className = "lb-sched-summary";

    summary.appendChild(schedBadge("Scheduled"));

    var label = document.createElement("span");
    label.className = "lb-sched-summary-text";
    label.textContent = regions.length
      ? regions.map(function (region) {
          return fmt(region.strength) + " at " + rangeLabel(region, schedule).toLowerCase();
        }).join(",  ")
      : "nothing drawn - every step is 0";
    summary.appendChild(label);

    var hint = document.createElement("span");
    hint.className = "lb-sched-summary-hint";
    hint.textContent = "Edit schedule";
    summary.appendChild(hint);

    summary.title = "This phase is scheduled over " + schedule.slots
      + " steps; steps no region covers are 0. Tap to edit it.";
    summary.addEventListener("click", function () {
      S.scheduleOpenById[row.id] = true;
      renderRows(true);
    });
    return summary;
  }

  /* ------------------------------------------------- strength controls */

  /* [-] slider [number] [+], with three deliberately different semantics:
   *   slider  - coarse, 0..1 on a 0.05 grid,
   *   +/-     - exact, 0.01 a tap, not confined to the slider's window,
   *   number  - exact direct entry, never quantised.
   * A value the slider cannot show is not rewritten: the thumb clamps and the
   * number keeps the truth until the slider is actually moved. */
  function buildWeightLine(options) {
    var line = document.createElement("div");
    line.className = "lb-weight";
    var accepted = Number(options.value);
    if (!isFinite(accepted)) { accepted = 1; }

    var down = stepButton("−", "Decrease by " + fmt(S.nudgeStep));
    var range = document.createElement("input");
    range.type = "range";
    range.min = String(S.sliderMin);
    range.max = String(S.sliderMax);
    range.step = String(S.sliderStep);
    range.setAttribute("aria-label", options.label || "Strength");

    var number = document.createElement("input");
    number.type = "number";
    number.step = String(S.nudgeStep);
    number.min = String(S.valueMin);
    number.max = String(S.valueMax);
    number.className = "lb-number";
    number.setAttribute("aria-label", (options.label || "Strength") + " value");

    var up = stepButton("+", "Increase by " + fmt(S.nudgeStep));

    function paint(value) {
      var next = Number(value);
      if (!isFinite(next)) { return; }
      accepted = next;
      if (document.activeElement !== number) { number.value = fmt(next); }
      syncSliderState(range, next);
    }

    function apply(value) {
      paint(value);
      if (options.commit) { options.commit(value); }
    }

    function nudgeBy(delta) {
      var next = round4(accepted + delta);
      if (next < S.valueMin || next > S.valueMax) { return; }
      apply(next);
      if (options.settle) { options.settle(); }
    }

    down.addEventListener("click", function () { nudgeBy(-S.nudgeStep); });
    up.addEventListener("click", function () { nudgeBy(S.nudgeStep); });

    range.addEventListener("pointerdown", function () { S.dragging = true; });
    range.addEventListener("input", function () { apply(Number(range.value)); });
    var settle = function () {
      S.dragging = false;
      if (options.settle) { options.settle(); }
    };
    range.addEventListener("change", settle);
    range.addEventListener("pointerup", settle);
    range.addEventListener("pointercancel", settle);

    // Typing is not committed until the field settles: partial input like "-"
    // or "1.5e" would otherwise be pushed to WanGP mid-keystroke.
    number.addEventListener("input", function () { syncSliderState(range, number.value); });
    var applyTyped = function () {
      var value = Number(number.value);
      if (number.value.trim() === "" || !isFinite(value) ||
          value < S.valueMin || value > S.valueMax) {
        number.value = fmt(accepted);
        syncSliderState(range, accepted);
        setStatus("Strength must be between " + fmt(S.valueMin) + " and " + fmt(S.valueMax) + ".", true);
        return;
      }
      apply(round4(value));
      if (options.settle) { options.settle(); }
    };
    number.addEventListener("change", applyTyped);
    number.addEventListener("blur", applyTyped);
    number.addEventListener("keydown", function (event) {
      if (event.key === "Enter") { event.preventDefault(); applyTyped(); }
    });

    if (options.disabled) {
      [down, range, number, up].forEach(function (control) { control.disabled = true; });
    }

    line.appendChild(down);
    line.appendChild(range);
    line.appendChild(number);
    line.appendChild(up);
    paint(accepted);
    return { node: line, paint: paint, number: number, range: range };
  }

  function stepButton(glyph, title) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "lb-btn lb-step";
    button.textContent = glyph;
    button.title = title;
    button.setAttribute("aria-label", title);
    return button;
  }

  function round4(value) {
    return Math.round(Number(value) * 10000) / 10000;
  }

  function clampToSlider(value) {
    var number = Number(value);
    if (!isFinite(number)) { return S.sliderMin; }
    return Math.min(S.sliderMax, Math.max(S.sliderMin, number));
  }

  /* The slider covers 0..1 on a coarse grid. A value outside it, or between
     grid stops, is still valid: the thumb goes to the nearest place it can and
     says so, and the number field keeps the exact value. */
  function syncSliderState(range, rawValue) {
    var value = Number(rawValue);
    if (!isFinite(value)) { return; }
    var outside = value < S.sliderMin || value > S.sliderMax;
    range.value = String(clampToSlider(value));
    var offGrid = Math.abs(Number(range.value) - value) > 1e-9;
    range.classList.toggle("lb-out-of-range", outside);
    range.title = outside
      ? "Exact value " + fmt(value) + " is outside the slider range; the number field holds it"
      : offGrid
        ? "Exact value " + fmt(value) + "; the slider moves in steps of " + fmt(S.sliderStep)
        : "";
  }

  /* Record an edit locally and schedule the native write. The row is never
     rebuilt from here, so a drag keeps its pointer capture. */
  function commitBase(row, phase, value) {
    var linked = !!S.linked[row.id] && !row.schedule_shared;
    var node = el.rows.querySelector('.lb-row[data-id="' + cssEscape(row.id) + '"]');
    if (linked && node) {
      node.querySelectorAll(".lb-phase-strip .lb-chip strong").forEach(function (readout) {
        readout.textContent = fmt(value);
      });
    }
    S.localValues[baseKey(row.id, phase)] = Number(value);
    if (linked) {
      (row.phase_values || []).forEach(function (ignored, index) {
        S.localValues[baseKey(row.id, index)] = Number(value);
      });
    }
    S.pendingValues[row.id + "#" + (linked ? "all" : phase)] = {
      type: "set_strength", id: row.id, phase: phase, value: Number(value), linked: linked
    };
    scheduleSync();
  }

  /* ------------------------------------------------------- timeline */

  /* Tools, then the timeline, then the strength control the plain row has --
     so a row looks the same scheduled or not, give or take a timeline. The
     timeline always has one slot per step: the step counter decides its
     length, never a setting here. */
  function buildSchedule(row, phase, schedule, options) {
    options = options || {};
    var box = document.createElement("div");
    box.className = "lb-schedule";
    if (!schedule) {
      var pending = document.createElement("div");
      pending.className = "lb-empty";
      pending.textContent = "Opening schedule...";
      box.appendChild(pending);
      return { node: box, paint: null };
    }

    if (!schedule.editable) {
      var locked = document.createElement("div");
      locked.className = "lb-preserved-note";
      locked.textContent = schedule.reason ||
        "This schedule is preserved exactly as imported and is read-only.";
      box.appendChild(locked);
      var fit = fitButton(row, phase, schedule);
      if (fit) { box.appendChild(fit); }
      return { node: box, paint: null };
    }

    var selected = selectedRegion(row, phase, schedule);
    box.appendChild(scheduleTools(row, phase, schedule, selected, options));
    var wrap = document.createElement("div");
    wrap.className = "lb-timeline-wrap";
    var timeline = buildTimeline(row, phase, schedule, selected);
    wrap.appendChild(timeline);
    box.appendChild(wrap);
    var weight = buildScheduleWeight(row, phase, schedule, selected, timeline);
    box.appendChild(weight.node);

    // Strengths can change without any region moving, so they repaint in place
    // rather than through a rebuild that would interrupt whoever is typing.
    return {
      node: box,
      paint: function (next) {
        var nextSchedule = scheduleOf(next, phaseOf(next));
        if (!nextSchedule) { return; }
        var regions = nextSchedule.regions || [];
        regions.forEach(function (region) {
          var node = timeline.querySelector('.lb-region[data-region="' + cssEscape(region.id) + '"]');
          var readout = node && node.querySelector("strong");
          if (readout) { readout.textContent = fmt(regionValue(next, phase, region)); }
        });
        if (!regions.length) {
          weight.paint(currentValue(next, phase, nextSchedule));
          return;
        }
        regions.forEach(function (region) {
          if (selected && region.id === selected.id) { weight.paint(regionValue(next, phase, region)); }
        });
      }
    };
  }

  /* One row of tools for the whole schedule, and Delete for the selected
     region at its far end -- there only while a region is selected. */
  function scheduleTools(row, phase, schedule, selected, options) {
    var tools = document.createElement("div");
    tools.className = "lb-sched-tools";

    if (!options.modal) {
      // A long run leaves little room per slot inline; this is where the
      // timeline gets the width to work in.
      var expand = document.createElement("button");
      expand.type = "button";
      expand.className = "lb-btn";
      expand.textContent = "⤢ Full screen";
      expand.title = "Edit this schedule in a full-screen view";
      expand.addEventListener("click", function () { openScheduleModal(row.id, phase); });
      tools.appendChild(expand);
    }

    var clear = document.createElement("button");
    clear.type = "button";
    clear.className = "lb-btn";
    clear.textContent = "Clear schedule";
    clear.title = (row.phase_values || []).length > 1 && !row.schedule_shared
      ? "Remove only this phase's schedule and go back to one strength"
      : "Remove the schedule and go back to one strength";
    clear.addEventListener("click", function () {
      if ((schedule.regions || []).length &&
          !window.confirm("Remove this schedule and keep " + fmt(schedule.base) + " as the strength?")) {
        return;
      }
      delete S.selectedRegionByKey[regionKey(row, phase)];
      S.scheduleOpenById[row.id] = false;
      if (options.modal) { closeScheduleModal(); }
      send({ type: "schedule_clear_phase", id: row.id, phase: phase });
    });
    tools.appendChild(clear);

    // An empty stage 2 can be drawn once from stage 1. Disabled, it says why
    // there is nothing to carry.
    var fillInfo = row.stage_2_fill;
    if (schedule.stage === 2 && fillInfo && !(schedule.regions || []).length) {
      var fill = document.createElement("button");
      fill.type = "button";
      fill.className = "lb-btn lb-fill-stage-2";
      fill.textContent = "Fill from stage 1";
      fill.title = fillInfo.note;
      fill.disabled = !!fillInfo.empty;
      fill.addEventListener("click", function () {
        delete S.selectedRegionByKey[regionKey(row, phase)];
        send({ type: "schedule_fill_stage_2", id: row.id });
      });
      tools.appendChild(fill);
    }

    if (selected) {
      var remove = document.createElement("button");
      remove.type = "button";
      remove.className = "lb-btn lb-delete-region";
      remove.textContent = "Delete";
      remove.title = "Delete the selected region (" + rangeLabel(selected, schedule).toLowerCase()
        + "); its steps go back to 0";
      remove.addEventListener("click", function () {
        delete S.selectedRegionByKey[regionKey(row, phase)];
        send({ type: "schedule_delete_region", id: row.id, phase: phase, region_id: selected.id });
      });
      tools.appendChild(remove);
    }
    return tools;
  }

  /* The plain row's strength control, pointed at what a scheduled phase has:
     the selected region -- or, before anything is drawn, the strength the
     phase is worth and new regions start at. */
  function buildScheduleWeight(row, phase, schedule, selected, timeline) {
    var drawn = (schedule.regions || []).length > 0;
    if (!drawn || !selected) {
      return buildWeightLine({
        value: currentValue(row, phase, schedule),
        label: (row.name + " " + (S.phaseLabels[phase] || "strength")).trim(),
        commit: function (value) { commitBase(row, phase, value); },
        settle: flushSync
      });
    }
    return buildWeightLine({
      value: regionValue(row, phase, selected),
      label: "Strength at " + rangeLabel(selected, schedule).toLowerCase(),
      commit: function (value) {
        var node = timeline.querySelector('.lb-region[data-region="' + cssEscape(selected.id) + '"]');
        var readout = node && node.querySelector("strong");
        if (readout) { readout.textContent = fmt(value); }
        S.localValues[regionKeyFor(row.id, phase, selected.id)] = Number(value);
        S.pendingValues[regionKeyFor(row.id, phase, selected.id)] = {
          type: "schedule_set_region_strength", id: row.id, phase: phase,
          region_id: selected.id, value: Number(value)
        };
        scheduleSync();
      },
      settle: flushSync
    });
  }

  /* A list too long to drag is offered one way back to editable: one value
     per step of the run, which is what every editable timeline is. */
  function fitButton(row, phase, schedule) {
    var steps = schedule.stage_steps || S.steps;
    if (!schedule.normalization_required || !steps) { return null; }
    var button = document.createElement("button");
    button.type = "button";
    button.className = "lb-btn";
    button.textContent = "Fit to " + steps + " steps";
    button.title = "Resample it to one value per step - this rewrites the multiplier";
    button.addEventListener("click", function () {
      setStatus("Fitting this schedule to " + steps + " steps...");
      send({ type: "schedule_normalize", id: row.id, phase: phase, slots: steps });
    });
    return button;
  }

  function buildTimeline(row, phase, schedule, selected) {
    var slots = Math.max(1, schedule.slots);
    var timeline = document.createElement("div");
    timeline.className = "lb-timeline";
    timeline.dataset.slots = String(slots);

    var ticks = tickPositions(slots);
    ticks.forEach(function (slot) {
      var line = document.createElement("div");
      line.className = "lb-gridline";
      line.style.left = ((slot - 1) / slots * 100) + "%";
      timeline.appendChild(line);

      var tick = document.createElement("div");
      tick.className = "lb-tick";
      tick.style.left = ((slot - 0.5) / slots * 100) + "%";
      tick.textContent = String(slot);
      timeline.appendChild(tick);
    });

    // Uncovered steps are zero, and saying so is the difference between "the
    // rest is off" and "the rest is some strength you cannot see". Drawing is
    // the only way in, so an empty timeline says how.
    var gapNote = document.createElement("div");
    gapNote.className = "lb-baseline";
    gapNote.textContent = (schedule.regions || []).length
      ? "empty steps = " + fmt(schedule.gap_strength || 0)
      : "Tap or drag to draw a region";
    timeline.appendChild(gapNote);

    var kept = keptPastEnd(schedule);
    if (kept) { timeline.appendChild(kept); }

    (schedule.regions || []).forEach(function (region) {
      timeline.appendChild(buildRegion(row, phase, schedule, region, selected));
    });

    // Empty space is a target too: press and drag to draw a region, or just tap
    // for one of the default width. Regions sit above this, so a gesture that
    // starts here always starts in free space.
    timeline.addEventListener("pointerdown", function (event) {
      if (event.target.closest(".lb-region")) { return; }
      beginDraw(event, { row: row, phase: phase, schedule: schedule, timeline: timeline, slots: slots });
    });
    timeline.title = "Drag across empty space to draw a region, or tap for one";
    return timeline;
  }

  /* A lower step count hides the end of a schedule rather than deleting it,
     so the timeline says when something is waiting past its last step. */
  function keptPastEnd(schedule) {
    var last = Number(schedule.drawn_past_end) || 0;
    if (last <= schedule.slots) { return null; }
    var more = last - schedule.slots;
    var label = document.createElement("div");
    label.className = "lb-kept";
    label.textContent = "+" + more + (more === 1 ? " step kept" : " steps kept");
    label.title = "Steps " + (schedule.slots + 1) + "-" + last + " are past the end of this "
      + schedule.slots + "-step run. They are kept, not deleted: raise the step count "
      + "and they come back as they were.";
    return label;
  }

  /* ----------------------------------------------- full-screen schedule */

  /* The same editor the row holds, given the whole window. Everything inside is
     the inline code path, so dragging, drawing and committing behave
     identically -- only the width differs. */
  function openScheduleModal(id, phase) {
    closeScheduleModal();
    var row = rowById(id);
    if (!row) { return; }

    var overlay = document.createElement("div");
    overlay.className = "lb-modal-overlay " + NS;
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.innerHTML =
      '<div class="lb-modal lb-sched-modal">' +
        '<div class="lb-modal-head">' +
          '<div class="lb-modal-title" data-lb="title"></div>' +
          '<button type="button" class="lb-btn lb-icon" data-lb="close" aria-label="Close">\u00D7</button>' +
        '</div>' +
        '<div class="lb-sched-modal-body" data-lb="body"></div>' +
      '</div>';
    document.body.appendChild(overlay);

    S.scheduleModal = { id: id, phase: phase, node: overlay };
    overlay.querySelector('[data-lb="close"]').addEventListener("click", closeScheduleModal);
    overlay.addEventListener("click", function (event) {
      if (event.target === overlay) { closeScheduleModal(); }
    });
    document.addEventListener("keydown", onScheduleModalKey);
    paintScheduleModal();
  }

  function onScheduleModalKey(event) {
    if (event.key === "Escape") { closeScheduleModal(); }
  }

  function closeScheduleModal() {
    if (!S.scheduleModal) { return; }
    document.removeEventListener("keydown", onScheduleModalKey);
    S.scheduleModal.node.remove();
    S.scheduleModal = null;
  }

  function rowById(id) {
    var found = null;
    S.rows.forEach(function (row) { if (row.id === id) { found = row; } });
    return found;
  }

  /* Rebuilt from each payload, the way the rows are -- never while a pointer is
     mid-gesture inside it. */
  function paintScheduleModal() {
    var modal = S.scheduleModal;
    if (!modal || S.drag || S.dragging) { return; }

    var row = rowById(modal.id);
    var schedule = row ? scheduleOf(row, modal.phase) : null;
    if (!row || !schedule) { closeScheduleModal(); return; }

    // The inline editor names the phase with its chip; here the title does.
    var phased = (row.phase_values || []).length > 1 && !row.schedule_shared;
    modal.node.querySelector('[data-lb="title"]').textContent =
      "Schedule \u2014 " + rowName(row) +
      (phased ? " \u00b7 " + (S.phaseLabels[modal.phase] || ("Phase " + (modal.phase + 1))) : "");
    var body = modal.node.querySelector('[data-lb="body"]');
    body.innerHTML = "";
    body.appendChild(buildSchedule(row, modal.phase, schedule, { modal: true }).node);
  }

  /* ---------------------------------------------------- drawing a region */

  function slotAt(timeline, clientX, slots) {
    var rect = timeline.getBoundingClientRect();
    if (!rect.width) { return 1; }
    var slot = Math.floor((clientX - rect.left) / (rect.width / slots)) + 1;
    return Math.max(1, Math.min(slots, slot));
  }

  /* The free run a slot falls in. Drawing is clipped to it, so a gesture that
     starts in a gap cannot swallow the region next to it -- that is what
     dragging a region is for. */
  function freeRunAt(regions, slots, slot) {
    var first = 1;
    var last = slots;
    (regions || []).forEach(function (region) {
      if (region.end < slot && region.end + 1 > first) { first = region.end + 1; }
      if (region.start > slot && region.start - 1 < last) { last = region.start - 1; }
      if (region.start <= slot && slot <= region.end) { first = 0; }
    });
    return first === 0 || first > last ? null : { start: first, end: last };
  }

  function beginDraw(event, context) {
    if (S.drag) { return; }
    var anchor = slotAt(context.timeline, event.clientX, context.slots);
    var run = freeRunAt(context.schedule.regions, context.slots, anchor);
    if (!run) { return; }

    event.preventDefault();
    var ghost = document.createElement("div");
    ghost.className = "lb-region lb-on lb-ghost";
    var body = document.createElement("div");
    body.className = "lb-region-body";
    body.appendChild(document.createElement("strong"));
    body.appendChild(document.createElement("span"));
    ghost.appendChild(body);
    context.timeline.appendChild(ghost);

    S.drag = {
      mode: "draw",
      context: context,
      anchor: anchor,
      run: run,
      ghost: ghost,
      current: { start: anchor, end: anchor },
      moved: false,
      pointerId: event.pointerId
    };
    paintGhost();

    try { context.timeline.setPointerCapture(event.pointerId); } catch (error) { /* mouse */ }
    context.timeline.addEventListener("pointermove", onDrawMove);
    context.timeline.addEventListener("pointerup", onDrawEnd);
    context.timeline.addEventListener("pointercancel", onDrawCancel);
  }

  function paintGhost() {
    var drag = S.drag;
    if (!drag || drag.mode !== "draw") { return; }
    var schedule = drag.context.schedule;
    var region = {
      start: drag.current.start, end: drag.current.end,
      strength: schedule.seed_strength !== undefined ? schedule.seed_strength : schedule.base
    };
    placeRegion(drag.ghost, region, drag.context.slots, drag.context.schedule);
  }

  function onDrawMove(event) {
    var drag = S.drag;
    if (!drag || drag.mode !== "draw") { return; }
    var slot = slotAt(drag.context.timeline, event.clientX, drag.context.slots);
    slot = Math.max(drag.run.start, Math.min(drag.run.end, slot));
    var next = {
      start: Math.min(drag.anchor, slot),
      end: Math.max(drag.anchor, slot)
    };
    if (next.start === drag.current.start && next.end === drag.current.end) { return; }
    drag.current = next;
    drag.moved = true;
    paintGhost();
  }

  function onDrawEnd() {
    var drag = S.drag;
    if (!drag || drag.mode !== "draw") { return; }
    var context = drag.context;
    var span = drag.current;
    var drew = drag.moved;
    releaseDraw();

    // A tap asks for a region of the default width; a drag asks for the one
    // that was drawn. Python clips either to the free run and owns the result.
    delete S.selectedRegionByKey[regionKey(context.row, context.phase)];
    send({
      type: "schedule_add_region",
      id: context.row.id,
      phase: context.phase,
      start: span.start,
      end: drew ? span.end : null
    });
  }

  function onDrawCancel() {
    if (!S.drag || S.drag.mode !== "draw") { return; }
    releaseDraw();
    renderRows(true);
  }

  function releaseDraw() {
    var drag = S.drag;
    if (!drag || drag.mode !== "draw") { return; }
    var timeline = drag.context.timeline;
    timeline.removeEventListener("pointermove", onDrawMove);
    timeline.removeEventListener("pointerup", onDrawEnd);
    timeline.removeEventListener("pointercancel", onDrawCancel);
    try { timeline.releasePointerCapture(drag.pointerId); } catch (error) { /* already gone */ }
    if (drag.ghost && drag.ghost.parentElement) { drag.ghost.remove(); }
    S.drag = null;
  }

  function tickPositions(slots) {
    var step = Math.max(1, Math.round(slots / 10));
    var ticks = [];
    for (var slot = 1; slot <= slots; slot += step) { ticks.push(slot); }
    if (ticks[ticks.length - 1] !== slots) { ticks.push(slots); }
    return ticks;
  }

  function buildRegion(row, phase, schedule, region, selected) {
    var shown = { id: region.id, start: region.start, end: region.end,
                  strength: regionValue(row, phase, region) };
    var slots = Math.max(1, schedule.slots);
    var node = document.createElement("div");
    node.className = "lb-region" + (selected && selected.id === region.id ? " lb-on" : "");
    node.dataset.region = region.id;
    node.tabIndex = 0;
    node.setAttribute("role", "button");

    var left = document.createElement("div");
    left.className = "lb-handle lb-handle-left";
    left.textContent = "⋮";
    var body = document.createElement("div");
    body.className = "lb-region-body";
    var strength = document.createElement("strong");
    var span = document.createElement("span");
    body.appendChild(strength);
    body.appendChild(span);
    var right = document.createElement("div");
    right.className = "lb-handle lb-handle-right";
    right.textContent = "⋮";
    node.appendChild(left);
    node.appendChild(body);
    node.appendChild(right);

    placeRegion(node, shown, slots, schedule);

    var start = function (event, mode) {
      beginDrag(event, {
        row: row, phase: phase, schedule: schedule, region: region,
        node: node, mode: mode, slots: slots
      });
    };
    node.addEventListener("pointerdown", function (event) {
      if (event.target === left || event.target === right) { return; }
      start(event, "move");
    });
    left.addEventListener("pointerdown", function (event) { start(event, "left"); });
    right.addEventListener("pointerdown", function (event) { start(event, "right"); });

    node.addEventListener("click", function () { selectRegion(row, phase, region.id); });
    node.addEventListener("keydown", function (event) {
      if (event.key !== "Enter" && event.key !== " ") { return; }
      event.preventDefault();
      selectRegion(row, phase, region.id);
    });
    return node;
  }

  function placeRegion(node, region, slots, schedule) {
    node.style.left = ((region.start - 1) / slots * 100) + "%";
    node.style.width = ((region.end - region.start + 1) / slots * 100) + "%";
    var strength = node.querySelector("strong");
    var span = node.querySelector(".lb-region-body span");
    if (strength) { strength.textContent = fmt(region.strength); }
    if (span) { span.textContent = rangeLabel(region, schedule); }
    node.setAttribute("aria-label", rangeLabel(region, schedule) + " at " + fmt(region.strength));
  }

  function rangeLabel(region, schedule) {
    // Within a stage of known length a slot is a step too, just of that stage.
    var step = schedule && (schedule.coordinate_mode === "global_exact"
      || schedule.coordinate_mode === "stage_exact");
    if (region.start === region.end) { return (step ? "Step " : "Slot ") + region.start; }
    return (step ? "Steps " : "Slots ") + region.start + "–" + region.end;
  }

  function selectRegion(row, phase, regionId) {
    if (S.selectedRegionByKey[regionKey(row, phase)] === regionId) { return; }
    S.selectedRegionByKey[regionKey(row, phase)] = regionId;
    renderRows(true);
  }

  /* Pointer Events with capture, so a drag survives leaving the element and
     works the same with a mouse, a pen or a thumb. Only the dragged region's
     styles change until the pointer is released. */
  function beginDrag(event, context) {
    if (S.drag) { return; }
    event.preventDefault();
    event.stopPropagation();

    var timeline = context.node.parentElement;
    var rect = timeline.getBoundingClientRect();
    var slotPx = rect.width / context.slots;
    if (!isFinite(slotPx) || slotPx <= 0) { return; }

    S.selectedRegionByKey[regionKey(context.row, context.phase)] = context.region.id;
    context.node.classList.add("lb-on", "lb-dragging");

    S.drag = {
      mode: context.mode,
      context: context,
      startX: event.clientX,
      slotPx: slotPx,
      origin: { start: context.region.start, end: context.region.end },
      current: { start: context.region.start, end: context.region.end },
      moved: false,
      pointerId: event.pointerId
    };

    try { context.node.setPointerCapture(event.pointerId); } catch (error) { /* mouse fallback */ }
    context.node.addEventListener("pointermove", onDragMove);
    context.node.addEventListener("pointerup", onDragEnd);
    context.node.addEventListener("pointercancel", onDragCancel);
  }

  function onDragMove(event) {
    var drag = S.drag;
    if (!drag || drag.mode === "draw") { return; }
    var delta = Math.round((event.clientX - drag.startX) / drag.slotPx);
    var slots = drag.context.slots;
    var origin = drag.origin;
    var next;

    if (drag.context.mode === "move") {
      var width = origin.end - origin.start + 1;
      var first = Math.max(1, Math.min(slots - width + 1, origin.start + delta));
      next = { start: first, end: first + width - 1 };
    } else if (drag.context.mode === "left") {
      next = { start: Math.max(1, Math.min(origin.end, origin.start + delta)), end: origin.end };
    } else {
      next = { start: origin.start, end: Math.max(origin.start, Math.min(slots, origin.end + delta)) };
    }

    if (next.start === drag.current.start && next.end === drag.current.end) { return; }
    drag.current = next;
    drag.moved = true;

    // Preview only: the canonical bounds are whatever Python returns. The
    // region's own label says where it is going.
    var preview = { start: next.start, end: next.end, strength: drag.context.region.strength };
    placeRegion(drag.context.node, preview, slots, drag.context.schedule);
  }

  function onDragEnd() {
    var drag = S.drag;
    if (!drag || drag.mode === "draw") { return; }
    var context = drag.context;
    releaseDrag();

    if (!drag.moved) { renderRows(true); return; }
    send({
      type: "schedule_commit_region",
      id: context.row.id,
      phase: context.phase,
      region_id: context.region.id,
      start: drag.current.start,
      end: drag.current.end,
      keep_width: context.mode === "move"
    });
  }

  /* An interrupted gesture is not an edit: go back to the committed state. */
  function onDragCancel() {
    if (!S.drag || S.drag.mode === "draw") { return; }
    releaseDrag();
    renderRows(true);
  }

  function releaseDrag() {
    var drag = S.drag;
    if (!drag || drag.mode === "draw") { return; }
    var node = drag.context.node;
    node.removeEventListener("pointermove", onDragMove);
    node.removeEventListener("pointerup", onDragEnd);
    node.removeEventListener("pointercancel", onDragCancel);
    try { node.releasePointerCapture(drag.pointerId); } catch (error) { /* already gone */ }
    node.classList.remove("lb-dragging");
    S.drag = null;
  }

  function pill(label) {
    var span = document.createElement("span");
    span.className = "lb-pill";
    span.textContent = label;
    return span;
  }

  function rowName(row) {
    if (S.nameMode === "civitai" && row.civitai_name) { return row.civitai_name; }
    return row.label || row.name;
  }

  function note(label) {
    var span = document.createElement("span");
    span.className = "lb-row-note";
    span.textContent = label;
    return span;
  }

  function scheduleSync() {
    clearTimeout(S.syncTimer);
    S.syncTimer = setTimeout(flushSync, SYNC_DEBOUNCE_MS);
  }

  /* One batch at a time. A second request would be built from the native state
     the first has not written yet, so its answer would arrive later and undo the
     newer edit -- which is exactly what made fast taps spring back. Anything
     queued meanwhile is coalesced by key and sent when the answer lands. */
  function flushSync() {
    clearTimeout(S.syncTimer);
    var actions = Object.keys(S.pendingValues).map(function (key) { return S.pendingValues[key]; });
    if (!actions.length || S.inFlight) { return; }

    S.pendingValues = {};
    S.inFlight = true;
    // If an answer never comes, do not strand the queue forever.
    clearTimeout(S.inFlightTimer);
    S.inFlightTimer = setTimeout(function () {
      S.inFlight = false;
      flushSync();
    }, 5000);
    send({ type: "batch", actions: actions });
  }

  function settleSync() {
    clearTimeout(S.inFlightTimer);
    S.inFlight = false;
    if (Object.keys(S.pendingValues).length) {
      flushSync();
      return false;
    }
    return true;
  }


  /* ------------------------------------------- list view, modes, groups */

  /* A mode change or a bake renames the file a row loads, not the row: what
     the row had open or chosen comes along to its new id. */
  function carryRowState(before, after) {
    var oldIds = {};
    before.forEach(function (row) { oldIds[rowKey(row)] = row.id; });
    after.forEach(function (row) {
      var was = oldIds[rowKey(row)];
      if (!was || was === row.id) { return; }
      [S.scheduleOpenById, S.selectedPhaseById, S.linked].forEach(function (map) {
        if (map[was] !== undefined && map[row.id] === undefined) { map[row.id] = map[was]; }
        delete map[was];
      });
      Object.keys(S.selectedRegionByKey).forEach(function (key) {
        if (key.indexOf(was + ":") !== 0) { return; }
        S.selectedRegionByKey[row.id + key.slice(was.length)] = S.selectedRegionByKey[key];
        delete S.selectedRegionByKey[key];
      });
    });
  }

  var MODE_LABEL = { normal: "Normal", visual: "Visual only", audio: "Audio only" };
  var MODE_SHORT = { normal: "Normal", visual: "Visual", audio: "Audio" };
  var MODE_HINT = {
    normal: "Apply everything this LoRA changes",
    visual: "Apply only what this LoRA changes in the picture",
    audio: "Apply only what this LoRA changes in the sound"
  };
  var GENERATE_WARNING = "If you generate before this finishes, you may get an unexpected result: " +
    "WanGP reports any file not built yet as missing.";
  var ICONS = {
    picture: '<svg viewBox="0 0 16 16" aria-hidden="true"><rect x="1.5" y="2.8" width="13" height="10.4" rx="1.6" ' +
      'fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M3.4 11.2l3-3.3 2.3 2.2 1.6-1.6 2.4 2.7" ' +
      'fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round" stroke-linecap="round"/></svg>',
    sound: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2.3 6h2.6l3.6-3v10l-3.6-3H2.3z" fill="currentColor"/>' +
      '<path d="M11 5.6a3.4 3.4 0 0 1 0 4.8M12.9 3.7a6 6 0 0 1 0 8.6" fill="none" stroke="currentColor" ' +
      'stroke-width="1.4" stroke-linecap="round"/></svg>',
    link: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M6.6 9.4l2.8-2.8M5.4 7.1L3.8 8.7a2.6 2.6 0 0 0 3.6 ' +
      '3.6L9 10.7M10.6 8.9l1.6-1.6a2.6 2.6 0 0 0-3.6-3.6L7 5.3" fill="none" stroke="currentColor" ' +
      'stroke-width="1.6" stroke-linecap="round"/></svg>'
  };

  function icon(name) {
    var holder = document.createElement("span");
    holder.className = "lb-ic";
    holder.setAttribute("aria-hidden", "true");
    holder.innerHTML = ICONS[name] || "";
    return holder;
  }

  /* The LoRA a row stands for. A visual-only copy is still its original's
     row: its name, its picture, its catalogue. */
  function sourceId(row) { return row.source_id || row.id; }

  /* What identifies a row from one payload to the next, whatever file it
     loads: a mode change or a bake renames the file, not the row. */
  function rowKey(row) { return row.kind === "group" ? "g:" + row.group : sourceId(row); }

  function buildState(build) { return build ? text(build.state) : ""; }

  function plural(count, one, many) { return count + " " + (count === 1 ? one : many); }

  function megabytes(bytes) {
    var value = Number(bytes) || 0;
    var mb = value / 1048576;
    if (mb >= 1024) { return (mb / 1024).toFixed(2) + " GB"; }
    if (mb < 1) { return Math.max(1, Math.round(value / 1024)) + " KB"; }
    return (mb >= 100 ? String(Math.round(mb)) : mb >= 10 ? mb.toFixed(1) : mb.toFixed(2)) + " MB";
  }

  function listJoin(names) {
    if (names.length < 2) { return names.join(""); }
    return names.slice(0, -1).join(", ") + " and " + names[names.length - 1];
  }

  function span(className, value) {
    var node = document.createElement("span");
    if (className) { node.className = className; }
    if (value !== undefined) { node.textContent = value; }
    return node;
  }

  function codeText(value) {
    var node = document.createElement("code");
    node.textContent = value;
    return node;
  }

  function paragraph(value, className) {
    var node = document.createElement("p");
    if (className) { node.className = className; }
    if (value) { node.textContent = value; }
    return node;
  }

  function makeButton(label, className, onClick, title) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = className || "lb-btn";
    button.textContent = label;
    if (title) { button.title = title; }
    if (onClick) { button.addEventListener("click", onClick); }
    return button;
  }

  function setView(view, persist) {
    view = view === "cards" ? "cards" : "list";
    if (view === S.view) { return; }
    S.view = view;
    S.openRow = "";
    if (view !== "list") { S.selecting = false; S.selected = {}; }
    if (persist) { send({ type: "view", value: view }); }
    renderHeader();
    renderRows(true);
    paintBulk();
  }

  /* Picking rows for one change: the list's own mode, so Cards gives way. */
  function setSelecting(on) {
    S.selecting = !!on;
    S.selected = {};
    if (S.selecting) {
      S.openRow = "";
      if (S.view !== "list") { setView("list", true); return; }
    }
    renderHeader();
    renderRows(true);
    paintBulk();
  }

  function avMenuItems() {
    var every = function (mode) { return function () { send({ type: "set_mode_all", mode: mode }); }; };
    return [
      { label: "Make every LoRA visual only", run: every("visual") },
      { label: "Make every LoRA audio only", run: every("audio") },
      { label: "Make every LoRA normal", run: every("normal") },
      { separator: true },
      { label: "Select which LoRAs…", run: function () { setSelecting(true); } }
    ];
  }

  /* --------------------------------------------------------- the list */

  /* The active LoRAs one line each: picture, name, what each phase applies,
     and the switch. Nothing here edits a strength: a click opens that LoRA's
     editor under its line, and opening one closes the one open before. */
  function buildList() {
    var phases = listPhaseCount();
    var list = document.createElement("div");
    list.className = "lb-list" + (S.selecting ? " lb-list-selecting" : "");
    list.style.setProperty("--lb-list-cols", listColumns(phases, false));
    list.style.setProperty("--lb-list-cols-narrow", listColumns(phases, true));
    list.appendChild(buildListHead(phases));
    S.rows.forEach(function (row) {
      var open = !S.selecting && S.openRow === rowKey(row);
      var holder = list;
      if (row.kind === "group") {
        holder = document.createElement("div");
        holder.className = "lb-li-group" + (open ? " lb-li-group-open" : "");
        list.appendChild(holder);
      }
      holder.appendChild(buildListItem(row, open));
      if (open) {
        var editor = document.createElement("div");
        editor.className = "lb-li-editor";
        editor.appendChild(buildRow(row));
        holder.appendChild(editor);
      } else if (row.kind === "group") {
        var members = row.members || [];
        members.forEach(function (member, index) {
          holder.appendChild(buildMemberItem(row, member, index === members.length - 1));
        });
      }
    });
    return list;
  }

  function listPhaseCount() {
    return Math.max(1, (S.phases && S.phases.effective) || 1);
  }

  function listColumns(phases, narrow) {
    var pick = S.selecting ? "30px " : "";
    if (narrow) { return pick + "44px 40px minmax(0, 1fr) 18px"; }
    var cells = [];
    for (var i = 0; i < phases; i += 1) { cells.push("minmax(0, 1fr)"); }
    return pick + "44px 40px minmax(0, 1.7fr) " + cells.join(" ") + " 18px";
  }

  function phaseName(index) {
    return S.phaseLabels[index] || ("Phase " + (index + 1));
  }

  function buildListHead(phases) {
    var head = document.createElement("div");
    head.className = "lb-li-head";
    head.setAttribute("aria-hidden", "true");
    if (S.selecting) { head.appendChild(span("")); }
    head.appendChild(span("", "On"));
    head.appendChild(span(""));
    head.appendChild(span("", "LoRA"));
    for (var i = 0; i < phases; i += 1) {
      var steps = S.phaseSteps[i];
      head.appendChild(span("", phaseName(i) + (steps ? " · " + plural(steps, "step", "steps") : "")));
    }
    head.appendChild(span(""));
    return head;
  }

  function buildListItem(row, open) {
    var key = rowKey(row);
    var group = row.kind === "group";
    var picked = !!S.selected[key];
    var item = document.createElement("div");
    item.className = "lb-li" + (group ? " lb-li-ghead" : "") + (isOn(row) ? "" : " lb-li-off") +
      (open ? " lb-li-open" : "") + (picked ? " lb-li-picked" : "") + (row.missing ? " lb-li-missing" : "");
    item.dataset.id = row.id;
    item.dataset.key = key;

    if (S.selecting) { item.appendChild(pickBox(row, key, picked)); }
    item.appendChild(powerSwitch(row));

    var thumb = document.createElement("div");
    thumb.className = "lb-li-thumb";
    if (group) {
      thumb.classList.add("lb-li-badge");
      thumb.appendChild(icon("link"));
      thumb.appendChild(span("lb-li-letter", row.group));
    } else {
      thumb.dataset.thumb = sourceId(row);
      thumb.appendChild(thumbMedia(sourceId(row)));
    }
    item.appendChild(thumb);

    var main = document.createElement("div");
    main.className = "lb-li-main";
    var name = document.createElement("button");
    name.type = "button";
    name.className = "lb-li-name";
    if (S.selecting) {
      name.title = "Select " + rowName(row);
    } else {
      name.setAttribute("aria-expanded", open ? "true" : "false");
      name.title = (open ? "Close " : "Edit ") + rowName(row);
    }
    name.appendChild(span("lb-li-label", rowName(row)));
    if (group) {
      name.appendChild(span("lb-li-sub", plural((row.members || []).length, "LoRA", "LoRAs") + ", one strength"));
      var bake = bakePill(row);
      if (bake) { name.appendChild(bake); }
    } else {
      modePills(row).forEach(function (node) { name.appendChild(node); });
      if (row.missing) { name.appendChild(note("Missing locally")); }
      if (row.system_managed) { name.appendChild(note("WanGP-managed")); }
    }
    main.appendChild(name);
    var cells = document.createElement("div");
    cells.className = "lb-li-phs";
    main.appendChild(cells);
    item.appendChild(main);

    var chevron = span("lb-li-chev", S.selecting ? "" : open ? "▴" : "▾");
    chevron.setAttribute("aria-hidden", "true");
    item.appendChild(chevron);

    item.addEventListener("click", function () {
      if (S.selecting) { togglePicked(key); return; }
      toggleOpenRow(key);
    });
    item.__lbPaint = function (next) { paintPhaseCells(cells, next); };
    item.__lbPaint(row);
    return item;
  }

  /* A member of a closed group: listed under it, with no settings of its own
     to show -- it applies the group's. */
  function buildMemberItem(group, member, last) {
    var item = document.createElement("div");
    item.className = "lb-li lb-li-member" + (isOn(group) ? "" : " lb-li-dim");
    item.dataset.member = member.id;
    if (S.selecting) { item.appendChild(span("")); }
    var tree = span("lb-li-tree" + (last ? "" : " lb-li-tree-next"));
    tree.setAttribute("aria-hidden", "true");
    item.appendChild(tree);
    var thumb = document.createElement("div");
    thumb.className = "lb-li-thumb";
    thumb.dataset.thumb = sourceId(member);
    thumb.appendChild(thumbMedia(sourceId(member)));
    item.appendChild(thumb);
    var main = document.createElement("div");
    main.className = "lb-li-main";
    var name = document.createElement("button");
    name.type = "button";
    name.className = "lb-li-name";
    name.title = "Edit Group " + group.group;
    name.setAttribute("aria-label", rowName(member) + ", in Group " + group.group + ". Open the group");
    name.appendChild(span("lb-li-label", rowName(member)));
    modePills(member).forEach(function (node) { name.appendChild(node); });
    if (member.missing) { name.appendChild(note("Missing locally")); }
    main.appendChild(name);
    var shared = span("lb-li-shared", "moves with Group " + group.group);
    shared.style.gridColumn = "span " + listPhaseCount();
    main.appendChild(shared);
    item.appendChild(main);
    item.appendChild(span(""));
    item.addEventListener("click", function () {
      if (S.selecting) { togglePicked(rowKey(group)); return; }
      toggleOpenRow(rowKey(group));
    });
    return item;
  }

  function toggleOpenRow(key) {
    var closing = S.openRow === key;
    S.openRow = closing ? "" : key;
    renderRows(true);
    var item = el.rows.querySelector('.lb-li[data-key="' + cssEscape(key) + '"]');
    if (!item) { return; }
    if (closing) {
      var name = item.querySelector(".lb-li-name");
      if (name) { name.focus(); }
      return;
    }
    var editor = item.parentNode.querySelector(".lb-li-editor");
    if (editor && editor.scrollIntoView) { editor.scrollIntoView({ block: "nearest" }); }
  }

  /* What each phase applies: its value, or for a scheduled phase how many
     steps it runs on -- never each step's strength. Those are in the editor. */
  function paintPhaseCells(cells, row) {
    cells.innerHTML = "";
    var summary = row.phase_summary || [];
    var count = listPhaseCount();
    if (summary.length === 1 && summary[0].kind === "raw") {
      var raw = document.createElement("div");
      raw.className = "lb-li-ph lb-li-raw";
      raw.style.gridColumn = "span " + count;
      raw.title = "Kept exactly as written: " + summary[0].text;
      raw.appendChild(codeText(summary[0].text));
      cells.appendChild(raw);
      return;
    }
    for (var i = 0; i < count; i += 1) {
      cells.appendChild(phaseCell(summary[i] || { kind: "value", value: 0 }, i, count));
    }
  }

  function phaseCell(cell, index, count) {
    var node = document.createElement("div");
    node.className = "lb-li-ph";
    if (count > 1) { node.dataset.short = "P" + (index + 1); }
    if (cell.kind === "steps") {
      node.classList.add("lb-li-sched");
      var wave = span("lb-li-wave", "∿");
      wave.setAttribute("aria-hidden", "true");
      node.appendChild(wave);
      node.appendChild(codeText(plural(cell.steps, "step", "steps")));
      node.title = phaseName(index) + ": scheduled on " + spansText(cell.spans) + " of " + cell.of +
        ". Open the row to see the strengths.";
      return node;
    }
    if (Number(cell.value) === 0) { node.classList.add("lb-li-zero"); }
    node.appendChild(codeText(fmt(cell.value)));
    node.title = phaseName(index) + ": x" + fmt(cell.value);
    return node;
  }

  function spansText(spans) {
    return (spans || []).map(function (pair) {
      return pair[0] === pair[1] ? "step " + pair[0] : "steps " + pair[0] + "–" + pair[1];
    }).join(", ") || "no steps";
  }

  /* Visual only / Audio only, and where its copy is. */
  function modePills(row) {
    var mode = row.mode || "normal";
    if (mode === "normal") { return []; }
    var node = span("lb-mode-pill lb-mode-" + mode);
    node.appendChild(icon(mode === "visual" ? "picture" : "sound"));
    node.appendChild(document.createTextNode(MODE_LABEL[mode]));
    var state = buildState(row.build);
    if (row.empty) {
      node.appendChild(document.createTextNode(" · nothing to apply"));
      node.title = MODE_LABEL[mode] + " leaves this LoRA nothing to apply, so it is switched off.";
    } else if (state === "building") {
      node.appendChild(document.createTextNode(" · building "));
      var pct = span("", (row.build.pct || 0) + "%");
      pct.dataset.buildPct = row.id;
      node.appendChild(pct);
    } else if (state === "queued") {
      node.appendChild(document.createTextNode(" · queued"));
    } else if (state === "failed") {
      node.appendChild(document.createTextNode(" · build failed"));
      node.classList.add("lb-pill-warn");
      node.title = row.build.error || "";
    } else if (state === "pending" || state === "missing" || state === "unavailable") {
      node.appendChild(document.createTextNode(" · not built"));
      node.classList.add("lb-pill-warn");
    }
    return [node];
  }

  function bakePill(row) {
    if (!row.baked || !row.bake) { return null; }
    var bake = row.bake;
    var state = buildState(bake);
    var node = span("lb-mode-pill lb-bake-pill");
    if (state === "built") {
      node.textContent = "Baked r" + bake.rank + (bake.bytes ? " · " + megabytes(bake.bytes) : "");
    } else if (state === "building") {
      node.textContent = "Baking ";
      var pct = span("", (bake.pct || 0) + "%");
      pct.dataset.buildPct = row.bake_id;
      node.appendChild(pct);
    } else if (state === "queued") {
      node.textContent = "Bake queued";
    } else if (state === "failed") {
      node.textContent = "Bake failed";
      node.classList.add("lb-pill-warn");
      node.title = bake.error || "";
    } else {
      node.textContent = "Bake not built";
      node.classList.add("lb-pill-warn");
    }
    return node;
  }

  function pickBox(row, key, picked) {
    var box = document.createElement("button");
    box.type = "button";
    box.className = "lb-pick";
    box.setAttribute("role", "checkbox");
    box.setAttribute("aria-checked", picked ? "true" : "false");
    box.setAttribute("aria-label", "Select " + rowName(row));
    box.textContent = "✓";
    box.addEventListener("click", function (event) {
      event.stopPropagation();
      togglePicked(key);
    });
    return box;
  }

  function togglePicked(key) {
    if (S.selected[key]) { delete S.selected[key]; } else { S.selected[key] = true; }
    renderRows(true);
    paintBulk();
  }

  function pickedRows() {
    return S.rows.filter(function (row) { return !!S.selected[rowKey(row)]; });
  }

  /* The bar under the header while rows are being picked. */
  function paintBulk() {
    var bar = el.bulk;
    if (!bar) { return; }
    if (!S.selecting) {
      bar.hidden = true;
      bar.innerHTML = "";
      return;
    }
    // A row that left the stack is no longer picked.
    var present = {};
    S.rows.forEach(function (row) { present[rowKey(row)] = true; });
    Object.keys(S.selected).forEach(function (key) { if (!present[key]) { delete S.selected[key]; } });

    var rows = pickedRows();
    bar.hidden = false;
    bar.innerHTML = "";
    bar.appendChild(span("lb-bulk-count",
      rows.length ? plural(rows.length, "row", "rows") + " selected" : "Tap rows to select them"));
    if (S.avSupported) {
      var set = span("lb-bulk-set");
      set.appendChild(span("lb-muted", "Make"));
      ["visual", "audio", "normal"].forEach(function (mode) {
        var button = makeButton(MODE_LABEL[mode], "lb-btn", function () {
          send({ type: "set_mode_all", mode: mode, ids: pickedRows().map(function (row) { return row.id; }) });
        }, MODE_HINT[mode]);
        button.disabled = !rows.length;
        set.appendChild(button);
      });
      bar.appendChild(set);
    }
    var loose = rows.filter(function (row) { return row.kind !== "group" && !row.system_managed; });
    var canGroup = loose.length >= 2 && loose.length === rows.length;
    var group = makeButton("Group", "lb-btn", function () { requestGroup(loose); },
      canGroup ? "Make these move together with one strength and schedule"
        : "Select two or more LoRAs that are not in a group yet");
    group.disabled = !canGroup;
    bar.appendChild(group);
    bar.appendChild(span("lb-grow"));
    bar.appendChild(makeButton("Select all", "lb-btn", function () {
      S.rows.forEach(function (row) { S.selected[rowKey(row)] = true; });
      renderRows(true);
      paintBulk();
    }));
    bar.appendChild(makeButton("Done", "lb-btn", function () { setSelecting(false); }));
  }

  /* -------------------------------------------------- editor sections */

  /* What an editor adds under a LoRA's strengths: which side it applies and
     what WanGP loads for it, and grouping. A group's editor has its members
     at the top and its bake at the bottom. It is a band across the whole
     card, under the picture: the picture stays beside the strengths. */
  function appendExtras(card, row) {
    if (row.system_managed) { return; }
    var box = document.createElement("div");
    box.className = "lb-row-extra";
    if (row.kind === "group") {
      box.appendChild(bakeBlock(row));
      card.appendChild(box);
      return;
    }
    var mode = row.mode || "normal";
    var line = document.createElement("div");
    line.className = "lb-extra-line";
    box.appendChild(line);
    if (S.avSupported) {
      line.appendChild(span("lb-extra-label", "Applies to"));
      line.appendChild(modeSeg(row, false));
      line.appendChild(span("lb-grow"));
    } else {
      line.appendChild(span("lb-extra-label", "Group"));
      line.appendChild(span("lb-muted", "Not in a group."));
    }
    var picker = groupPicker(row);
    if (picker) { line.appendChild(picker); }
    if (S.avSupported && S.view === "list" && row.streams) { box.appendChild(streamView(row)); }
    if (mode !== "normal") {
      var load = loadLine(row);
      if (load) { box.appendChild(load); }
    }
    card.appendChild(box);
  }

  function modeSeg(row, small) {
    var current = row.mode || "normal";
    var seg = document.createElement("div");
    seg.className = "lb-seg" + (small ? " lb-seg-sm" : "");
    seg.setAttribute("role", "radiogroup");
    seg.setAttribute("aria-label", rowName(row) + " applies to");
    ["normal", "visual", "audio"].forEach(function (mode) {
      var button = document.createElement("button");
      button.type = "button";
      button.setAttribute("role", "radio");
      button.className = "lb-seg-" + mode;
      button.setAttribute("aria-checked", current === mode ? "true" : "false");
      if (mode !== "normal") { button.appendChild(icon(mode === "visual" ? "picture" : "sound")); }
      button.appendChild(document.createTextNode(small ? MODE_SHORT[mode] : MODE_LABEL[mode]));
      button.title = MODE_HINT[mode];
      button.addEventListener("click", function (event) {
        event.stopPropagation();
        if (current !== mode) { send({ type: "set_mode", id: row.id, mode: mode }); }
      });
      seg.appendChild(button);
    });
    return seg;
  }

  /* How much of the LoRA changes the picture and how much the sound. */
  function streamView(row) {
    var streams = row.streams || {};
    var mode = row.mode || "normal";
    var parts = [
      ["picture", "Picture", streams.picture || {}, mode === "audio"],
      ["shared", "Both", streams.shared || {}, false],
      ["sound", "Sound", streams.sound || {}, mode === "visual"]
    ];
    var total = parts.reduce(function (sum, part) { return sum + (part[2].bytes || 0); }, 0);
    var wrap = document.createElement("div");
    wrap.className = "lb-streams";
    var bar = document.createElement("div");
    bar.className = "lb-sbar";
    bar.setAttribute("aria-hidden", "true");
    var legend = document.createElement("div");
    legend.className = "lb-slegend";
    parts.forEach(function (part) {
      var info = part[2];
      if (total && info.bytes) {
        var piece = document.createElement("i");
        piece.className = "lb-s-" + part[0] + (part[3] ? " lb-s-out" : "");
        piece.style.width = (info.bytes / total * 100).toFixed(1) + "%";
        bar.appendChild(piece);
      }
      if (part[0] === "shared" && !info.tensors) { return; }
      var item = span("");
      var swatch = document.createElement("i");
      swatch.className = "lb-sw lb-s-" + part[0];
      item.appendChild(swatch);
      item.appendChild(span("lb-s-name", part[1]));
      item.appendChild(document.createTextNode(info.tensors
        ? " " + plural(info.tensors, "tensor", "tensors") + " · " + megabytes(info.bytes) +
          (part[3] ? " · left out" : part[0] === "shared" ? " · kept either way" : "")
        : " none"));
      legend.appendChild(item);
    });
    wrap.appendChild(bar);
    wrap.appendChild(legend);
    return wrap;
  }

  function progressBar(key, pct) {
    var bar = document.createElement("div");
    bar.className = "lb-prog";
    var fill = document.createElement("i");
    fill.dataset.buildBar = key;
    fill.style.width = (pct || 0) + "%";
    bar.appendChild(fill);
    return bar;
  }

  /* What WanGP loads for a LoRA in Visual only or Audio only, and how far
     along its copy is. */
  function loadLine(row) {
    var mode = row.mode || "normal";
    var build = row.build || {};
    var state = build.state || "";
    var side = mode === "visual" ? "picture" : "sound";
    var other = mode === "visual" ? "sound" : "picture";
    var box = document.createElement("div");
    box.className = "lb-load";
    var line = document.createElement("div");
    box.appendChild(line);
    var add = function (value) { line.appendChild(typeof value === "string" ? document.createTextNode(value) : value); };

    if (row.empty) {
      add(rowName(row) + " only changes the " + other + ", so " + MODE_LABEL[mode].toLowerCase() +
        " leaves it nothing to apply and it is switched off. Normal or " +
        (mode === "visual" ? "Audio" : "Visual") + " only switches it back on.");
      return box;
    }
    if (state === "built") {
      add("WanGP loads ");
      add(codeText(row.id));
      add((build.bytes ? " · " + megabytes(build.bytes) : "") + ", an exact copy of its " + side + " tensors.");
    } else if (state === "building") {
      add("Building ");
      add(codeText(row.id));
      add(" · ");
      var pct = span("", (build.pct || 0) + "%");
      pct.dataset.buildPct = row.id;
      add(pct);
      add(". If you generate before it is ready, WanGP reports it as missing.");
      box.appendChild(progressBar(row.id, build.pct));
    } else if (state === "queued") {
      add("Waiting to build ");
      add(codeText(row.id));
      add(", after the files ahead of it. If you generate before it is ready, WanGP reports it as missing.");
    } else if (state === "failed") {
      box.classList.add("lb-load-warn");
      add("Could not build ");
      add(codeText(row.id));
      add(": " + (build.error || "unknown error") + ".");
      var retry = document.createElement("div");
      retry.appendChild(makeButton("Retry", "lb-btn", function () { send({ type: "build_retry", id: row.id }); }));
      box.appendChild(retry);
    } else if (state === "missing") {
      box.classList.add("lb-load-warn");
      add("The original file is not here, so its " + side + " copy cannot be built.");
    } else if (state === "unavailable") {
      box.classList.add("lb-load-warn");
      add("This model has no LoRA folder to build copies in.");
    } else if (state === "pending") {
      add("Not built yet: ");
      add(codeText(row.id));
      add(".");
    } else {
      return null;
    }
    return box;
  }

  /* "Group with..." -- another LoRA, or a group to join. */
  function groupPicker(row) {
    var others = S.rows.filter(function (other) { return other !== row && !other.system_managed; });
    if (!others.length) { return null; }
    var select = document.createElement("select");
    select.className = "lb-select";
    select.setAttribute("aria-label", "Group " + rowName(row) + " with another LoRA");
    select.appendChild(new Option("Group with…", ""));
    others.forEach(function (other) {
      if (other.kind === "group") {
        select.appendChild(new Option("Add to Group " + other.group, "g:" + other.group));
        return;
      }
      var same = (other.multiplier_raw || "") === (row.multiplier_raw || "");
      select.appendChild(new Option(rowName(other) + (same ? " · same settings" : ""), "r:" + rowKey(other)));
    });
    select.addEventListener("click", function (event) { event.stopPropagation(); });
    select.addEventListener("change", function () {
      var value = select.value;
      select.value = "";
      if (!value) { return; }
      if (value.slice(0, 2) === "g:") {
        send({ type: "group_add", group: value.slice(2), id: row.id });
        return;
      }
      var other = S.rows.filter(function (candidate) { return rowKey(candidate) === value.slice(2); })[0];
      if (other) { requestGroup([row, other]); }
    });
    return select;
  }

  /* Grouping LoRAs that disagree asks whose strength and schedule the group
     keeps; LoRAs that already agree are grouped at once. */
  function requestGroup(rows) {
    if (rows.length < 2) { return; }
    var first = rows[0].multiplier_raw || "";
    var same = rows.every(function (row) { return (row.multiplier_raw || "") === first; });
    if (same) {
      createGroup(rows, rows[0].id);
      return;
    }
    var pick = rows[0].id;
    var body = document.createElement("div");
    body.appendChild(paragraph("These LoRAs have different strengths or schedules. A group shares one, " +
      "so pick whose the group keeps. You can change it afterwards."));
    var options = document.createElement("div");
    options.className = "lb-opts";
    options.setAttribute("role", "radiogroup");
    options.setAttribute("aria-label", "Settings the group keeps");
    rows.forEach(function (row, index) {
      options.appendChild(optionRow("lb-group-pick", row.id, index === 0, rowName(row),
        MODE_LABEL[row.mode || "normal"], ["x" + (row.multiplier_raw || "1")], function () { pick = row.id; }));
    });
    body.appendChild(options);
    if (S.avSupported) {
      body.appendChild(paragraph("Each member keeps its own Normal, Visual only or Audio only setting.", "lb-fine"));
    }
    openDialog("Group " + plural(rows.length, "LoRA", "LoRAs"), body, [
      { label: "Cancel" },
      { label: "Group them", primary: true, run: function () { createGroup(rows, pick); } }
    ]);
  }

  function createGroup(rows, lead) {
    send({ type: "group_create", ids: rows.map(function (row) { return row.id; }), lead: lead });
    if (S.selecting) { setSelecting(false); }
  }

  function optionRow(name, value, checked, title, sub, side, onPick) {
    var label = document.createElement("label");
    label.className = "lb-opt";
    var input = document.createElement("input");
    input.type = "radio";
    input.name = name;
    input.value = value;
    input.checked = !!checked;
    input.addEventListener("change", function () { if (input.checked) { onPick(); } });
    label.appendChild(input);
    var main = span("lb-opt-main");
    main.appendChild(span("lb-opt-title", title));
    if (sub) { main.appendChild(span("lb-opt-sub", sub)); }
    label.appendChild(main);
    var aside = span("lb-opt-side");
    (side || []).forEach(function (value, index) { aside.appendChild(span(index ? "" : "lb-opt-big", value)); });
    label.appendChild(aside);
    return label;
  }

  /* A group's members, each with its own side and a way out of the group. */
  function buildMembers(row) {
    var list = document.createElement("div");
    list.className = "lb-members";
    (row.members || []).forEach(function (member) {
      var item = document.createElement("div");
      item.className = "lb-member" + (S.avSupported ? "" : " lb-member-noseg");
      var thumb = document.createElement("div");
      thumb.className = "lb-member-thumb";
      thumb.dataset.thumb = sourceId(member);
      thumb.appendChild(thumbMedia(sourceId(member)));
      item.appendChild(thumb);
      var name = document.createElement("div");
      name.className = "lb-member-name";
      name.title = [member.civitai_name, member.name].filter(Boolean).join("\n");
      name.appendChild(span("lb-li-label", rowName(member)));
      var detail = memberNote(member, row);
      if (detail) { name.appendChild(span("lb-member-note", detail)); }
      item.appendChild(name);
      if (S.avSupported) { item.appendChild(modeSeg(member, true)); }
      var leave = makeButton("×", "lb-btn lb-icon", function () {
        send({ type: "group_leave", group: row.group, id: member.id });
      }, "Take " + rowName(member) + " out of Group " + row.group);
      leave.setAttribute("aria-label", leave.title);
      item.appendChild(leave);
      list.appendChild(item);
    });
    return list;
  }

  function memberNote(member, group) {
    var parts = [];
    var streams = member.streams;
    if (streams && streams.rank) { parts.push("Rank " + streams.rank); }
    var mode = member.mode || "normal";
    if (member.empty) {
      parts.push("skipped: nothing to apply in " + MODE_LABEL[mode].toLowerCase());
    } else if (group.baked) {
      parts.push(buildState(group.bake) === "built" ? "inside the group's baked file"
        : "goes into the group's bake once it is built");
    } else if (mode !== "normal") {
      var state = buildState(member.build);
      parts.push(state === "built" ? "exact " + mode + "-only copy"
        : state === "building" ? "building its copy"
        : state === "failed" ? "its copy could not be built"
        : "waiting to build its copy");
    } else if (streams) {
      var picture = streams.picture && streams.picture.tensors;
      var sound = streams.sound && streams.sound.tensors;
      parts.push(picture && sound ? "changes picture and sound"
        : picture ? "changes the picture only" : sound ? "changes the sound only" : "");
    }
    return parts.filter(Boolean).join(" · ");
  }

  function keptText(bake) {
    if (bake.exact) { return "all"; }
    if (bake.kept === undefined || bake.kept === null) { return "most"; }
    var pct = Number(bake.kept) * 100;
    return (pct >= 99.95 ? "over 99.9" : pct.toFixed(1)) + "%";
  }

  /* A group's memory: its members loaded side by side, or one baked file. */
  function bakeBlock(row) {
    var box = document.createElement("div");
    box.className = "lb-extra-block";
    var head = document.createElement("div");
    head.className = "lb-extra-line";
    head.appendChild(span("lb-extra-label", "Memory"));
    box.appendChild(head);
    var load = document.createElement("div");
    load.className = "lb-load";
    var line = document.createElement("div");
    load.appendChild(line);
    box.appendChild(load);

    var bake = row.bake || {};
    var count = (row.members || []).length;
    var unbake = function () {
      return makeButton("Unbake", "lb-btn", function () { send({ type: "unbake", group: row.group }); },
        "Load the members separately again");
    };
    if (!row.baked) {
      head.appendChild(span("lb-muted", "Not baked"));
      head.appendChild(span("lb-grow"));
      head.appendChild(makeButton("Bake to save memory…", "lb-btn", function () { openBakeDialog(row); }));
      line.textContent = "WanGP loads " + plural(count, "file", "files") + " for this group. Baking re-fits " +
        "them into one file at the rank you choose; the originals stay, and Unbake goes back to them.";
      return box;
    }
    var state = buildState(bake);
    if (state === "built") {
      head.appendChild(span("lb-mode-pill lb-bake-pill", "Baked r" + bake.rank));
      head.appendChild(span("lb-grow"));
      head.appendChild(makeButton("Change size…", "lb-btn", function () { openBakeDialog(row); }));
      head.appendChild(unbake());
      line.textContent = "WanGP loads one file" + (bake.bytes ? " of " + megabytes(bake.bytes) : "") +
        " instead of " + count + ". It keeps " + keptText(bake) + " of their combined change. " +
        "Strength and schedule still apply live.";
    } else if (state === "building" || state === "queued") {
      head.appendChild(span("", (state === "building" ? "Baking" : "Bake queued") + " at rank " + bake.rank));
      head.appendChild(span("lb-grow"));
      head.appendChild(unbake());
      if (state === "building") {
        line.appendChild(document.createTextNode("Building one file from the members' current settings · "));
        var pct = span("", (bake.pct || 0) + "%");
        pct.dataset.buildPct = row.bake_id;
        line.appendChild(pct);
        line.appendChild(document.createTextNode(". If you generate before it is ready, WanGP reports it as missing."));
        load.appendChild(progressBar(row.bake_id, bake.pct));
      } else {
        line.textContent = "Waiting for the files ahead of it. If you generate before it is ready, " +
          "WanGP reports it as missing.";
      }
    } else if (state === "failed") {
      head.appendChild(span("lb-mode-pill lb-bake-pill lb-pill-warn", "Bake failed"));
      head.appendChild(span("lb-grow"));
      head.appendChild(makeButton("Retry", "lb-btn", function () { send({ type: "build_retry", id: row.bake_id }); }));
      head.appendChild(unbake());
      load.classList.add("lb-load-warn");
      line.textContent = (bake.error || "The bake did not finish") + ". Unbake to load the members separately.";
    } else {
      head.appendChild(span("lb-mode-pill lb-bake-pill lb-pill-warn", "Not built"));
      head.appendChild(span("lb-grow"));
      head.appendChild(unbake());
      load.classList.add("lb-load-warn");
      line.textContent = state === "unavailable" ? "This model has no LoRA folder to build the baked file in."
        : "The baked file is not built yet. If you generate now, WanGP reports it as missing.";
    }
    return box;
  }

  /* ------------------------------------------------------------ dialogs */

  function openDialog(title, body, actions) {
    closeMenu();
    closeDialog();
    var overlay = document.createElement("div");
    overlay.className = "lb-modal-overlay " + NS;
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", title);
    var box = document.createElement("div");
    box.className = "lb-modal lb-dialog";
    var head = document.createElement("div");
    head.className = "lb-modal-head";
    head.appendChild(span("lb-modal-title", title));
    var close = makeButton("×", "lb-btn lb-icon", closeDialog);
    close.setAttribute("aria-label", "Close");
    head.appendChild(close);
    box.appendChild(head);
    body.classList.add("lb-dialog-body");
    box.appendChild(body);
    var foot = document.createElement("div");
    foot.className = "lb-dialog-foot";
    box.appendChild(foot);
    overlay.appendChild(box);

    var dialog = {
      overlay: overlay,
      body: body,
      setActions: function (list) {
        foot.innerHTML = "";
        list.forEach(function (action) {
          var button = makeButton(action.label,
            "lb-btn" + (action.primary ? " lb-primary" : "") + (action.left ? " lb-left" : ""));
          button.disabled = !!action.disabled;
          button.addEventListener("click", function () {
            if (action.run) { action.run(button); }
            if (!action.keepOpen) { closeDialog(); }
          });
          foot.appendChild(button);
        });
      }
    };
    dialog.setActions(actions);
    S.dialogReturn = document.activeElement;
    document.body.appendChild(overlay);
    S.dialog = overlay;
    overlay.addEventListener("click", function (event) {
      if (event.target === overlay) { closeDialog(); }
    });
    document.addEventListener("keydown", onDialogKey);
    var first = box.querySelector(".lb-dialog-body input:checked, .lb-dialog-foot .lb-primary") || close;
    first.focus();
    return dialog;
  }

  function onDialogKey(event) {
    if (event.key === "Escape") { closeDialog(); }
  }

  function closeDialog() {
    if (!S.dialog) { return; }
    document.removeEventListener("keydown", onDialogKey);
    S.dialog.remove();
    S.dialog = null;
    var back = S.dialogReturn;
    S.dialogReturn = null;
    if (back && back.focus && document.body.contains(back)) { back.focus(); }
  }

  function openBakeDialog(row) {
    var body = document.createElement("div");
    body.appendChild(paragraph("Reading the LoRAs…", "lb-muted"));
    var dialog = openDialog((row.baked ? "Change the size of " : "Bake ") + "Group " + row.group, body,
      [{ label: "Cancel" }]);
    var members = (row.members || []).map(function (member) { return member.id; });
    requestMedia({ kind: "bake_plan", members: members, group: row.group }, function (plan) {
      if (S.dialog !== dialog.overlay) { return; }
      renderBakePlan(dialog, row, plan);
    });
  }

  function renderBakePlan(dialog, row, plan) {
    var body = dialog.body;
    body.innerHTML = "";
    if (!plan || plan.error || !plan.options) {
      body.appendChild(paragraph(plan && plan.error ? plan.error : "Could not read the LoRAs in this group."));
      dialog.setActions([{ label: "Close", primary: true }]);
      return;
    }
    var names = (row.members || []).map(rowName);
    var ranks = plan.options.map(function (option) { return option.rank; });
    var current = row.baked && row.bake ? row.bake.rank : 0;
    var pick = ranks.indexOf(current) >= 0 ? current : plan.default_rank;
    body.appendChild(paragraph("Builds one LoRA from " + listJoin(names) + ". It starts at the rank of the " +
      "highest-rank LoRA in the group; pick another size if you want one. The originals are not touched, and " +
      "Unbake goes back to them."));
    body.appendChild(paragraph("Now: WanGP loads " + plural(names.length, "file", "files") + ", " +
      megabytes(plan.current_bytes) + " in all, rank " + plan.combined_rank + " combined.", "lb-muted"));
    var options = document.createElement("div");
    options.className = "lb-opts";
    options.setAttribute("role", "radiogroup");
    options.setAttribute("aria-label", "Baked size");
    plan.options.forEach(function (option) {
      var saves = plan.current_bytes - option.bytes;
      var title = (option.exact ? "Exact, rank " : "Rank ") + option.rank +
        (option.rank === plan.default_rank ? " · default" : "") +
        (option.rank === current ? " · now" : "");
      var sub = option.exact
        ? "Same memory as now: WanGP already runs the group this way. Only worth it to hand someone one file."
        : (option.rank === plan.default_rank ? "Matches the highest rank in the group. " : "") +
          (saves > 0 ? "Saves " + megabytes(saves) + "." : "");
      options.appendChild(optionRow("lb-bake-pick", String(option.rank), option.rank === pick, title, sub,
        [megabytes(option.bytes), option.exact ? "exact" : "re-fit"], function () { pick = option.rank; }));
    });
    body.appendChild(options);
    body.appendChild(paragraph("A smaller file keeps the strongest directions of the group's combined change. " +
      "How much it keeps is measured while it is built, and shown on the group afterwards.", "lb-fine"));
    dialog.setActions([
      { label: "Cancel" },
      { label: row.baked ? "Rebuild" : "Bake", primary: true, run: function () {
        send({ type: "bake", group: row.group, rank: pick });
      } }
    ]);
    var checked = body.querySelector("input:checked");
    if (checked) { checked.focus(); }
  }

  /* Every built file this stack uses, by the id the stack holds it by. */
  function stackFileIds() {
    var ids = [];
    S.rows.forEach(function (row) {
      ids.push(row.id);
      if (row.bake_id) { ids.push(row.bake_id); }
      (row.members || []).forEach(function (member) { ids.push(member.id); });
    });
    return ids;
  }

  function openBuiltDialog() {
    var body = document.createElement("div");
    body.appendChild(paragraph("Reading the built files…", "lb-muted"));
    var dialog = openDialog("Built files", body, [{ label: "Close", primary: true }]);
    loadBuilt(dialog, "");
  }

  function loadBuilt(dialog, action) {
    var request = { kind: "built", in_use: stackFileIds() };
    if (action) { request.action = action; }
    requestMedia(request, function (reply) {
      if (action) { send({ type: "ready" }); }          // the status bar's count
      if (S.dialog !== dialog.overlay) { return; }
      renderBuilt(dialog, reply);
    });
  }

  function renderBuilt(dialog, reply) {
    var body = dialog.body;
    body.innerHTML = "";
    if (!reply || reply.error) {
      body.appendChild(paragraph(reply && reply.error ? reply.error : "Could not read the built files."));
      dialog.setActions([{ label: "Close", primary: true }]);
      return;
    }
    var intro = paragraph("");
    intro.appendChild(document.createTextNode("Stored in "));
    intro.appendChild(codeText(reply.folder + "/"));
    intro.appendChild(document.createTextNode(reply.linked
      ? " and linked into this model's LoRA folder as "
      : " and kept in this model's LoRA folder as "));
    intro.appendChild(codeText(reply.link));
    intro.appendChild(document.createTextNode(", so WanGP loads them like any other LoRA."));
    body.appendChild(intro);

    var files = reply.files || [];
    if (!files.length) {
      body.appendChild(paragraph("Nothing is built for this model yet.", "lb-muted"));
    } else {
      var wrap = document.createElement("div");
      wrap.className = "lb-table-wrap";
      var table = document.createElement("table");
      table.className = "lb-table";
      var headRow = table.createTHead().insertRow();
      ["File", "Built from", "Size", "Used by"].forEach(function (label) {
        var cell = document.createElement("th");
        cell.scope = "col";
        cell.textContent = label;
        headRow.appendChild(cell);
      });
      var tbody = table.createTBody();
      files.forEach(function (file) {
        var line = tbody.insertRow();
        var name = line.insertCell();
        name.className = "lb-t-file";
        name.appendChild(codeText(text(file.id).split("/").pop()));
        name.title = file.id;
        var from = line.insertCell();
        from.textContent = file.kind === "bake"
          ? "Group bake, rank " + file.rank + ": " + listJoin(file.members || []) +
            (file.kept !== null && file.kept !== undefined ? " · keeps " + keptText(file) : "")
          : (file.label || "") + " · " + (MODE_LABEL[file.mode] || file.mode || "");
        var size = line.insertCell();
        size.className = "lb-t-num";
        size.textContent = megabytes(file.bytes);
        var used = line.insertCell();
        if ((file.used_by || []).length) {
          used.textContent = file.used_by.join(", ");
        } else {
          used.appendChild(span("lb-muted", "nothing"));
        }
      });
      wrap.appendChild(table);
      body.appendChild(wrap);
    }
    body.appendChild(paragraph("Delete unused removes only files that neither this stack nor a saved profile " +
      "uses. Anything deleted is built again from its originals if it is ever used again.", "lb-fine"));
    var unused = reply.unused || 0;
    dialog.setActions([
      { label: "Delete unused (" + unused + ")", left: true, disabled: !unused, keepOpen: true,
        run: function (button) { button.disabled = true; loadBuilt(dialog, "delete_unused"); } },
      { label: "Close", primary: true }
    ]);
  }

  function paintBuiltLink() {
    if (!el.built) { return; }
    var built = S.built || {};
    el.built.hidden = !built.count;
    el.built.textContent = "Built files: " + (built.count || 0) + " · " + megabytes(built.bytes);
  }

  /* ------------------------------------------------------------- builds */

  /* While this stack waits for files, a bar above the rows says so and
     warns -- the panel never blocks Generate. */
  function paintBuildBar(live) {
    var bar = el.buildBar;
    if (!bar) { return; }
    var builds = S.builds || {};
    if (!builds.waiting) {
      bar.hidden = true;
      bar.innerHTML = "";
      return;
    }
    var status = live || builds;
    var failed = Object.keys(builds.failed || {});
    bar.hidden = false;
    bar.innerHTML = "";
    bar.className = "lb-buildbar" + (status.running ? " lb-buildbar-running" : " lb-buildbar-stuck");
    var line = document.createElement("div");
    line.className = "lb-buildbar-row";
    var mark = span("lb-warn-ic", "!");
    mark.setAttribute("aria-hidden", "true");
    line.appendChild(mark);
    var words = span("lb-buildbar-text");
    var strong = document.createElement("b");
    words.appendChild(strong);
    line.appendChild(words);
    bar.appendChild(line);

    if (status.running) {
      var queued = Array.isArray(status.queued) ? status.queued.length : (status.queued || 0);
      var done = status.finished || 0;
      var current = status.current;
      if (current) {
        strong.textContent = "Building " + (done + 1) + " of " + (done + 1 + queued);
        words.appendChild(document.createTextNode(" · " + current.label + " · " + (current.pct || 0) + "%"));
      } else {
        strong.textContent = "Waiting to build " + plural(builds.waiting, "file", "files");
      }
      words.appendChild(document.createElement("br"));
      words.appendChild(document.createTextNode(GENERATE_WARNING));
      if (current) { bar.appendChild(progressBar("", current.pct)); }
      return;
    }
    if (failed.length) {
      strong.textContent = "Could not build " + plural(failed.length, "file", "files");
      words.appendChild(document.createTextNode(": " + builds.failed[failed[0]]));
      words.appendChild(document.createElement("br"));
      words.appendChild(document.createTextNode("If you generate now, WanGP reports " +
        (failed.length === 1 ? "it" : "them") + " as missing."));
      line.appendChild(makeButton("Retry", "lb-btn", function () {
        send({ type: "batch", actions: failed.map(function (key) { return { type: "build_retry", id: key }; }) });
      }));
      return;
    }
    strong.textContent = plural(builds.waiting, "file is", "files are") + " not built yet";
    words.appendChild(document.createElement("br"));
    words.appendChild(document.createTextNode("If you generate now, WanGP reports " +
      (builds.waiting === 1 ? "it" : "them") + " as missing."));
  }

  function buildSignature(status) {
    var failures = status.failures !== undefined ? status.failures : Object.keys(status.failed || {}).length;
    return [status.generation, failures, status.current ? status.current.key : ""].join("|");
  }

  function pollBuildsSoon(delay) {
    clearTimeout(S.buildTimer);
    S.buildTimer = null;
    var builds = S.builds;
    if (!builds || !builds.running || !builds.waiting) { return; }
    S.buildTimer = setTimeout(pollBuilds, delay || 900);
  }

  /* Progress is read on the media channel; only a change in what is built
     asks for a payload, which is what redraws the rows. */
  function pollBuilds() {
    S.buildTimer = null;
    if (!S.ready || !el.rows || !document.body.contains(el.rows)) { return; }
    requestMedia({ kind: "builds" }, function (reply) {
      if (!S.builds) { return; }
      if (!reply || reply.kind !== "builds" || reply.error) { pollBuildsSoon(3000); return; }
      if (!reply.running || buildSignature(reply) !== buildSignature(S.builds)) {
        // A sync in flight brings a fresh payload of its own.
        if (!S.inFlight) { send({ type: "ready" }); }
        pollBuildsSoon(3000);                        // in case no payload comes
        return;
      }
      paintBuildProgress(reply);
      pollBuildsSoon();
    }, 5000);
  }

  function paintBuildProgress(live) {
    paintBuildBar(live);
    var current = live.current;
    if (!current || !el.rows) { return; }
    var pct = (current.pct || 0) + "%";
    el.rows.querySelectorAll('[data-build-pct="' + cssEscape(current.key) + '"]').forEach(function (node) {
      node.textContent = pct;
    });
    el.rows.querySelectorAll('[data-build-bar="' + cssEscape(current.key) + '"]').forEach(function (node) {
      node.style.width = pct;
    });
  }

  /* ------------------------------------------------------ context menu */

  function openMenu(x, y, items) {
    closeMenu();
    menu = document.createElement("div");
    menu.className = "lb-menu-portal";
    items.forEach(function (item) {
      if (item.separator) { menu.appendChild(document.createElement("hr")); return; }
      var button = document.createElement("button");
      button.type = "button";
      button.textContent = item.label;
      button.disabled = !!item.disabled;
      button.addEventListener("click", function () { closeMenu(); item.run(); });
      menu.appendChild(button);
    });
    document.body.appendChild(menu);

    var rect = menu.getBoundingClientRect();
    menu.style.left = Math.min(x, window.innerWidth - rect.width - 8) + "px";
    menu.style.top = Math.min(y, window.innerHeight - rect.height - 8) + "px";
    var first = menu.querySelector("button:not(:disabled)");
    if (first) { first.focus(); }

    // The click that opened this menu is still bubbling toward document. Arming
    // the dismiss listener synchronously would let that same click close the
    // menu again, which is why the "..." button appeared dead while right-click
    // (a contextmenu event) worked.
    setTimeout(function () {
      document.addEventListener("click", closeMenu, { once: true });
    }, 0);
    document.addEventListener("keydown", onMenuKey);
    window.addEventListener("scroll", closeMenu, { once: true, passive: true });
  }

  function onMenuKey(event) {
    if (event.key === "Escape") { closeMenu(); }
  }

  function closeMenu() {
    if (!menu) { return; }
    document.removeEventListener("keydown", onMenuKey);
    menu.remove();
    menu = null;
  }

  /* ---------------------------------------------------- paste a stack */

  /* A whole stack as text, one "name: x<multiplier>" line per LoRA, loaded in
     place of the user's LoRAs the way recalling a profile is. Python reads the
     names and the multipliers; this box only collects the text and says what
     came of it. Enter loads, Shift+Enter starts a new line, Escape closes. */
  var PASTE_EXAMPLE = "Detail_Tweaker: x0.8;0.5\nCamera_Pan: x1,1,0.875,0.875,0.25,0.25;0";

  function togglePaste() {
    if (S.paste) { closePaste(true); return; }
    openPaste();
  }

  function openPaste() {
    closeMenu();
    var pop = document.createElement("div");
    pop.className = "lb-paste " + NS;
    pop.setAttribute("role", "dialog");
    pop.setAttribute("aria-label", "Paste a LoRA stack");
    pop.innerHTML =
      '<div class="lb-paste-head">' +
        '<span class="lb-paste-title">Paste a LoRA stack</span>' +
        '<button type="button" class="lb-btn lb-icon" data-lb="paste-close" aria-label="Close">\u00D7</button>' +
      '</div>' +
      '<textarea class="lb-paste-text" data-lb="paste-text" rows="6" wrap="off" spellcheck="false" ' +
        'autocomplete="off" aria-label="LoRA stack, one name: x multiplier per line"></textarea>' +
      '<div class="lb-paste-hint">One LoRA per line, as <code>name: x multiplier</code>. ' +
        '<kbd>Enter</kbd> loads it in place of the current stack; ' +
        '<kbd>Shift</kbd>+<kbd>Enter</kbd> starts a new line.</div>' +
      '<div class="lb-paste-report" data-lb="paste-report" aria-live="polite"></div>' +
      '<div class="lb-paste-actions">' +
        '<button type="button" class="lb-btn" data-lb="paste-copy">Copy current stack</button>' +
        '<button type="button" class="lb-btn lb-primary" data-lb="paste-load">Load</button>' +
      '</div>';
    document.body.appendChild(pop);

    var pick = function (name) { return pop.querySelector('[data-lb="' + name + '"]'); };
    var paste = {
      node: pop, text: pick("paste-text"), report: pick("paste-report"),
      copy: pick("paste-copy"), load: pick("paste-load"), busy: false, seq: 0, timer: null
    };
    S.paste = paste;

    paste.text.placeholder = PASTE_EXAMPLE;
    paste.text.value = S.pasteDraft;
    paste.text.addEventListener("input", function () { S.pasteDraft = paste.text.value; });
    paste.text.addEventListener("keydown", function (event) {
      if (event.key !== "Enter" || event.shiftKey || event.altKey) { return; }
      if (event.isComposing || event.keyCode === 229) { return; }   // an IME is choosing
      event.preventDefault();
      event.stopPropagation();
      submitPaste();
    });
    pop.addEventListener("keydown", function (event) {
      if (event.key !== "Escape") { return; }
      event.preventDefault();
      event.stopPropagation();
      closePaste(true);
    });
    pick("paste-close").addEventListener("click", function () { closePaste(true); });
    paste.load.addEventListener("click", submitPaste);
    paste.copy.addEventListener("click", function () {
      if (!S.stackText) { return; }
      var count = S.stackText.split("\n").length;
      copyText(S.stackText, "the current stack (" + count + (count === 1 ? " LoRA)" : " LoRAs)"));
    });

    document.addEventListener("pointerdown", onPastePointer, true);
    window.addEventListener("resize", positionPaste);
    window.addEventListener("scroll", positionPaste, true);
    paintPaste();
    positionPaste();
    // Anything already there is selected, so a paste replaces it.
    paste.text.focus();
    paste.text.select();
  }

  function closePaste(restoreFocus) {
    var paste = S.paste;
    if (!paste) { return; }
    clearTimeout(paste.timer);
    document.removeEventListener("pointerdown", onPastePointer, true);
    window.removeEventListener("resize", positionPaste);
    window.removeEventListener("scroll", positionPaste, true);
    paste.node.remove();
    S.paste = null;
    paintPaste();
    if (restoreFocus && el.paste && document.body.contains(el.paste)) { el.paste.focus(); }
  }

  /* A press anywhere else closes the box; what was typed waits for it. The
     button that opened it toggles it itself. */
  function onPastePointer(event) {
    var paste = S.paste;
    if (!paste || paste.node.contains(event.target)) { return; }
    if (el.paste && el.paste.contains(event.target)) { return; }
    closePaste(false);
  }

  /* Under the button that opened it, kept on screen: above the button when
     there is no room below, and inside the window either way. */
  function positionPaste() {
    var paste = S.paste;
    if (!paste) { return; }
    var pop = paste.node;
    var width = Math.min(560, window.innerWidth - 16);
    pop.style.width = width + "px";
    var height = pop.getBoundingClientRect().height;
    var anchor = el.paste && document.body.contains(el.paste) ? el.paste.getBoundingClientRect() : null;
    var left = (window.innerWidth - width) / 2;
    var top = 80;
    if (anchor && anchor.width) {
      left = anchor.right - width;
      top = anchor.bottom + 6;
      if (top + height > window.innerHeight - 8 && anchor.top - height - 6 >= 8) {
        top = anchor.top - height - 6;
      }
    }
    left = Math.max(8, Math.min(left, window.innerWidth - width - 8));
    top = Math.max(8, Math.min(top, window.innerHeight - height - 8));
    pop.style.left = Math.round(left) + "px";
    pop.style.top = Math.round(top) + "px";
  }

  /* What the button and the box show, from the latest payload. */
  function paintPaste() {
    if (el.paste) {
      el.paste.setAttribute("aria-expanded", S.paste ? "true" : "false");
      el.paste.classList.toggle("lb-on", !!S.paste);
    }
    var paste = S.paste;
    if (!paste) { return; }
    paste.copy.disabled = !S.stackText;
    paste.copy.title = S.stackText
      ? "Copy the LoRAs that are on, in this format"
      : "No LoRAs are on";
    paste.load.disabled = paste.busy;
    paste.load.textContent = paste.busy ? "Loading..." : "Load";
  }

  function submitPaste() {
    var paste = S.paste;
    if (!paste || paste.busy) { return; }
    var text = paste.text.value;
    if (!text.trim()) {
      showPasteReport(null, "Paste a stack first: one name: x multiplier per line.");
      return;
    }
    // Edits not yet sent were made to the stack this replaces; landing after
    // it, they would change the LoRAs just pasted.
    clearTimeout(S.syncTimer);
    S.pendingValues = {};
    S.pasteSeq += 1;
    paste.seq = S.pasteSeq;
    paste.busy = true;
    showPasteReport(null, "");
    paintPaste();
    send({ type: "stack_paste", text: text, seq: paste.seq });
    clearTimeout(paste.timer);
    paste.timer = setTimeout(function () {
      if (S.paste !== paste || !paste.busy) { return; }
      paste.busy = false;
      paintPaste();
      showPasteReport(null, "WanGP did not answer. Try again.", true);
    }, 15000);
  }

  /* The answer to a paste: a clean load closes the box, anything to say keeps
     it open with the text, so a line can be fixed and loaded again. */
  function onPasteReport(report) {
    var paste = S.paste;
    if (!paste || !report || report.seq !== paste.seq) { return; }
    clearTimeout(paste.timer);
    paste.busy = false;
    paintPaste();
    if (report.ok && !(report.notes || []).length) {
      S.pasteDraft = "";
      closePaste(true);
      return;
    }
    if (report.ok) { S.pasteDraft = ""; }
    showPasteReport(report);
  }

  function showPasteReport(report, message, warn) {
    var paste = S.paste;
    if (!paste) { return; }
    var box = paste.report;
    box.innerHTML = "";
    box.classList.toggle("lb-warn", !!warn || !!(report && !report.ok));
    if (!report) {
      box.textContent = message || "";
      return;
    }
    var summary = document.createElement("div");
    summary.className = "lb-paste-summary";
    summary.textContent = report.loaded
      ? "Loaded " + report.loaded + " of " + report.lines + (report.lines === 1 ? " LoRA." : " LoRAs.")
      : "Nothing was loaded.";
    box.appendChild(summary);

    // textContent throughout: every name here came from the clipboard.
    var list = document.createElement("ul");
    (report.problems || []).forEach(function (problem) {
      var item = document.createElement("li");
      var where = document.createElement("strong");
      where.textContent = "Line " + problem.line;
      item.appendChild(where);
      item.appendChild(document.createTextNode(" \u2014 " + problem.reason));
      item.title = problem.text;
      list.appendChild(item);
    });
    (report.notes || []).forEach(function (note) {
      var item = document.createElement("li");
      item.className = "lb-paste-note";
      item.textContent = note;
      list.appendChild(item);
    });
    if (list.childNodes.length) { box.appendChild(list); }
    positionPaste();
  }

  /* --------------------------------------------------------- inspect */

  /* A near-full-screen view of the Civitai catalogue the enrichment script
     wrote beside the LoRA: description, trigger words, and the media gallery
     with the prompt that produced each item. Media is fetched one item at a
     time so opening the view is instant even for a large gallery. */
  function openInspect(id) {
    closeMenu();
    closeInspect();

    var overlay = document.createElement("div");
    overlay.className = "lb-modal-overlay " + NS;
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.innerHTML =
      '<div class="lb-modal">' +
        '<div class="lb-modal-head">' +
          '<div class="lb-modal-title" data-lb="title">Loading...</div>' +
          '<button type="button" class="lb-btn lb-icon" data-lb="close" aria-label="Close">\u00D7</button>' +
        '</div>' +
        '<div class="lb-modal-body">' +
          '<div class="lb-modal-side" data-lb="side"></div>' +
          '<div class="lb-modal-media" data-lb="media"></div>' +
        '</div>' +
      '</div>';
    document.body.appendChild(overlay);
    S.modal = overlay;

    overlay.querySelector('[data-lb="close"]').addEventListener("click", closeInspect);
    overlay.addEventListener("click", function (event) {
      if (event.target === overlay) { closeInspect(); }
    });
    document.addEventListener("keydown", onInspectKey);

    requestMedia({ kind: "inspect", id: id }, function (detail) {
      if (S.modal !== overlay) { return; }
      renderInspect(overlay, id, detail);
    });
  }

  function onInspectKey(event) {
    if (event.key === "Escape") { closeInspect(); }
  }

  function closeInspect() {
    if (!S.modal) { return; }
    document.removeEventListener("keydown", onInspectKey);
    S.modal.remove();
    S.modal = null;
  }

  function renderInspect(overlay, id, detail) {
    var title = overlay.querySelector('[data-lb="title"]');
    var side = overlay.querySelector('[data-lb="side"]');
    var media = overlay.querySelector('[data-lb="media"]');

    side.innerHTML = "";
    media.innerHTML = "";

    // A hard error means the LoRA itself could not be resolved, so there is
    // nothing to fetch for either.
    if (!detail || detail.error) {
      title.textContent = (S.byId[id] || {}).name || id;
      side.textContent = (detail && detail.error) || "This LoRA could not be read.";
      return;
    }

    title.textContent = detail.civitai_name || detail.name || id;

    side.appendChild(inspectFacts(detail));
    side.appendChild(fetchSection(overlay, id, detail));
    if ((detail.trained_words || []).length) {
      side.appendChild(inspectWords(detail.trained_words));
    }
    if (detail.description) {
      side.appendChild(inspectBlock("Description", detail.description));
    }
    if (detail.version_description) {
      side.appendChild(inspectBlock("Version notes", detail.version_description));
    }

    if (!(detail.media || []).length) {
      var empty = document.createElement("div");
      empty.className = "lb-empty";
      empty.textContent = detail.has_catalogue
        ? "No media has been downloaded for this LoRA."
        : "No catalogue folder yet. Fetch from Civitai to build one.";
      media.appendChild(empty);
      return;
    }
    detail.media.forEach(function (item) {
      media.appendChild(inspectMediaCard(id, item));
    });
  }

  /* Fetching identifies the LoRA by the SHA-256 of the file itself, so it works
     for any model family Civitai knows -- there is nothing model-specific to
     configure. It can take a while on a LoRA with ten videos, so the request
     gets its own generous timeout and the button reports progress in place. */
  function fetchNote(detail) {
    if (!detail.has_catalogue) { return "Looks this file up on Civitai by its checksum."; }
    var missing = detail.missing || [];
    if (missing.length) {
      // A catalogue with pictures but no prompts still reads as finished
      // otherwise, so name what is actually absent.
      return "Missing: " + missing.join(", ") + ". Fetching adds it.";
    }
    return "This catalogue is complete. Fetching refreshes it.";
  }

  function fetchSection(overlay, id, detail) {
    var section = document.createElement("div");
    section.className = "lb-modal-section lb-fetch";

    var button = document.createElement("button");
    button.type = "button";
    button.className = "lb-btn lb-fetch-btn";
    button.textContent = detail.has_catalogue
      ? "Update from Civitai"
      : "Fetch info from Civitai";
    section.appendChild(button);

    var note = document.createElement("div");
    note.className = "lb-fetch-note";
    note.textContent = detail.note || fetchNote(detail);
    section.appendChild(note);

    button.addEventListener("click", function () {
      if (button.disabled) { return; }
      button.disabled = true;
      button.textContent = "Fetching from Civitai...";
      note.textContent = "This can take a while for a LoRA with several videos.";

      requestMedia({ kind: "fetch", id: id }, function (response) {
        if (S.modal !== overlay) { return; }
        if (!response) {
          button.disabled = false;
          button.textContent = "Fetch info from Civitai";
          note.textContent = "The fetch did not finish in time. It may still be running.";
          return;
        }
        if (!response.ok) {
          button.disabled = false;
          button.textContent = "Try again";
          note.textContent = response.message || response.error || "The fetch failed.";
          return;
        }
        // A new preview and a new Civitai name only reach the grid on the next
        // payload, and the thumbnail for this tile was cached as "none".
        delete S.thumbs[id];
        delete S.requested[id];
        send({ type: "ready" });
        // Carry the outcome into the rebuilt panel; the note line is the only
        // place it would otherwise be visible.
        response.note = response.message || "";
        renderInspect(overlay, id, response);
        // Ten minutes: a LoRA with ten videos is a real download. The bridge is
        // serialised, so nothing else is answered until this returns.
      }, 600000);
    });

    return section;
  }

  function inspectFacts(detail) {
    var wrap = document.createElement("div");
    wrap.className = "lb-modal-facts";
    [["File", detail.name], ["Version", detail.version_name],
     ["Creator", detail.creator], ["Base model", detail.base_model]]
      .forEach(function (pair) {
        if (!pair[1]) { return; }
        var row = document.createElement("div");
        var key = document.createElement("span");
        key.className = "lb-fact-key";
        key.textContent = pair[0];
        var value = document.createElement("span");
        value.textContent = pair[1];
        row.appendChild(key);
        row.appendChild(value);
        wrap.appendChild(row);
      });

    if (detail.civitai_url) {
      var link = document.createElement("a");
      link.className = "lb-modal-link";
      link.href = detail.civitai_url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = "Open on Civitai";
      wrap.appendChild(link);
    }
    return wrap;
  }

  function inspectWords(words) {
    var section = document.createElement("div");
    section.className = "lb-modal-section";

    var head = document.createElement("div");
    head.className = "lb-modal-section-head";
    head.appendChild(sectionTitle("Trigger words"));
    head.appendChild(copyButton(words.join(", "), "Copy all"));
    section.appendChild(head);

    var list = document.createElement("div");
    list.className = "lb-words";
    words.forEach(function (word) {
      var chip = document.createElement("button");
      chip.type = "button";
      chip.className = "lb-word";
      chip.textContent = word;
      chip.title = "Copy \"" + word + "\"";
      chip.addEventListener("click", function () { copyText(word); });
      list.appendChild(chip);
    });
    section.appendChild(list);
    return section;
  }

  function inspectBlock(label, text) {
    var section = document.createElement("div");
    section.className = "lb-modal-section";
    var head = document.createElement("div");
    head.className = "lb-modal-section-head";
    head.appendChild(sectionTitle(label));
    head.appendChild(copyButton(text, "Copy"));
    section.appendChild(head);

    var body = document.createElement("div");
    body.className = "lb-modal-text";
    body.textContent = text;          // never innerHTML: this is remote content
    section.appendChild(body);
    return section;
  }

  function sectionTitle(label) {
    var node = document.createElement("div");
    node.className = "lb-modal-section-title";
    node.textContent = label;
    return node;
  }

  function copyButton(text, label) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "lb-btn lb-copy";
    button.textContent = label;
    button.addEventListener("click", function () {
      copyText(text);
      var previous = button.textContent;
      button.textContent = "Copied";
      setTimeout(function () { button.textContent = previous; }, 1200);
    });
    return button;
  }

  function inspectMediaCard(id, item) {
    var card = document.createElement("figure");
    card.className = "lb-media-card";

    var frame = document.createElement("div");
    frame.className = "lb-media-frame";
    var spinner = document.createElement("div");
    spinner.className = "lb-media-loading";
    spinner.textContent = item.kind === "video" ? "Loading video..." : "Loading...";
    frame.appendChild(spinner);
    card.appendChild(frame);

    if (item.prompt || item.negative_prompt) {
      card.appendChild(mediaPrompt("Prompt", item.prompt));
      if (item.negative_prompt) {
        card.appendChild(mediaPrompt("Negative", item.negative_prompt));
      }
    }

    requestMedia({ kind: "media", id: id, index: item.index }, function (response) {
      if (!document.body.contains(card)) { return; }
      frame.innerHTML = "";
      if (!response || !response.data) {
        var failed = document.createElement("div");
        failed.className = "lb-media-loading";
        failed.textContent = (response && response.error) || "Unavailable";
        frame.appendChild(failed);
        return;
      }
      if (response.media_kind === "video") {
        var video = document.createElement("video");
        video.controls = true;
        video.loop = true;
        video.muted = true;
        video.playsInline = true;
        video.src = response.data;
        frame.appendChild(video);
      } else {
        var image = document.createElement("img");
        image.loading = "lazy";
        image.alt = item.prompt || "";
        image.src = response.data;
        frame.appendChild(image);
      }
    });
    return card;
  }

  function mediaPrompt(label, text) {
    if (!text) { return document.createElement("span"); }
    var block = document.createElement("figcaption");
    block.className = "lb-media-prompt";

    var head = document.createElement("div");
    head.className = "lb-modal-section-head";
    head.appendChild(sectionTitle(label));
    head.appendChild(copyButton(text, "Copy"));
    block.appendChild(head);

    var body = document.createElement("div");
    body.className = "lb-modal-text lb-prompt-text";
    body.textContent = text;
    block.appendChild(body);
    return block;
  }

  /* ---------------------------------------------------------- payload */

  function apply(json) {
    try {
      applyPayload(json);
    } catch (error) {
      // Any render failure must hand the native controls back rather than
      // stranding the user with neither UI.
      console.error("[LoRA Browser] render failed", error);
      S.ready = false;
      hideNative(false);
      setStatus("LoRA Browser could not render; native controls restored.", true);
    }
  }

  function applyPayload(json) {
    if (!json) { return; }
    var payload;
    try { payload = JSON.parse(json); } catch (error) {
      console.error("[LoRA Browser] bad payload", error);
      return;
    }
    if (payload.error) {
      setStatus(payload.error, true);
      hideNative(false);
      S.ready = false;
      return;
    }
    if (!mount()) { return; }
    // The answer to a paste is about the paste, even when a newer payload
    // overtook the one carrying it.
    if (payload.paste_report) { onPasteReport(payload.paste_report); }
    // Ignore a stale async response that lost the race with a newer sync.
    if (payload.revision !== undefined && payload.revision < S.revision) { return; }

    var previousModel = S.modelKey;
    S.revision = payload.revision || 0;
    S.modelKey = payload.model_key || "";
    S.phases = payload.phases || { capacity: 1, effective: 1 };
    S.phaseLabels = payload.phase_labels || ["Strength"];
    S.items = payload.items || [];
    var previousRows = S.rows || [];
    S.rows = payload.active || [];
    carryRowState(previousRows, S.rows);
    S.profiles = payload.profiles || [];
    S.profilesIncomplete = payload.profiles_incomplete || [];
    S.activeProfile = payload.active_profile || "";
    S.canDisableAll = !!payload.can_disable_all;
    S.canEnableAll = !!payload.can_enable_all;
    S.stackText = payload.stack_text || "";
    S.signature = payload.signature || "";
    S.sliderMin = payload.slider_min !== undefined ? payload.slider_min : 0;
    S.sliderMax = payload.slider_max !== undefined ? payload.slider_max : 1;
    S.sliderStep = payload.slider_step || 0.05;
    S.nudgeStep = payload.nudge_step || 0.01;
    S.valueMin = payload.value_min !== undefined ? payload.value_min : -10;
    S.valueMax = payload.value_max !== undefined ? payload.value_max : 10;
    S.valueDecimals = payload.value_decimals || 4;
    S.steps = payload.steps || 0;
    S.phaseSteps = payload.phase_steps || [];
    S.schedulingEnabled = payload.scheduling_enabled !== false;
    S.schedulingDisabledReason = payload.scheduling_disabled_reason || "";
    // Leaving One Phase closes every timeline: there is nothing left to show.
    if (!S.schedulingEnabled) { S.scheduleOpenById = {}; closeScheduleModal(); }
    S.defaultProfile = payload.default_profile || "";
    S.civitaiKeySet = !!payload.civitai_key_set;
    if (payload.sort_mode) { S.sortMode = payload.sort_mode; }
    if (payload.name_mode) { S.nameMode = payload.name_mode; }
    // The view is the panel's own from here on: a payload overtaken by a click
    // on List or Cards must not flip it back.
    if (!S.ready && payload.stack_view) { S.view = payload.stack_view === "cards" ? "cards" : "list"; }
    S.avSupported = !!payload.av_supported;
    S.builds = payload.builds || null;
    S.built = payload.built || { count: 0, bytes: 0 };

    if (previousModel && previousModel !== S.modelKey) {
      // The model changed: previews and per-LoRA UI state no longer apply.
      S.thumbs = {};
      S.requested = {};
      S.pendingThumbs = [];
      S.linked = {};
      S.selectedPhaseById = {};
      S.scheduleOpenById = {};
      S.selectedRegionByKey = {};
      S.playingTile = null;
      S.openRow = "";
      S.selected = {};
      closeInspect();
      closeScheduleModal();
      closeDialog();
    }

    S.byId = {};
    S.items.forEach(function (item) { S.byId[item.id] = item; });
    S.rows.forEach(function (row) {
      if (S.linked[row.id] === undefined && row.linked) { S.linked[row.id] = true; }
    });

    if (payload.zoom_px && !S.ready) { S.applyZoom(payload.zoom_px, false); }
    else { S.applyZoom(S.zoomPx, false); }

    // A payload is the answer to whatever was in flight. Once nothing is left
    // queued, WanGP's value is the truth again and local intent is dropped;
    // while edits are still outstanding they keep winning, so a value the user
    // has just set does not flicker back to the one before it.
    if (settleSync()) { S.localValues = {}; }

    // The step counter moved: bring every timeline to the new length before
    // anything is drawn from it. Python decides what that means per schedule
    // and answers with a payload where nothing is out of sync any more, so this
    // settles after one round rather than looping.
    if (payload.schedules_out_of_sync && !S.inFlight && !S.drag && !S.dragging) {
      send({ type: "schedule_resync" });
    }

    renderHeader();
    renderGrid();
    // Not forced: the row key knows when structure changed, and rebuilding on a
    // value-only payload would take focus out of a field being typed into.
    renderRows();
    paintScheduleModal();
    paintBulk();
    paintBuildBar();
    paintBuiltLink();
    pollBuildsSoon();
    setStatus(payload.status || "Synced to WanGP state", !!payload.status_warn);
    maybeApplyDefaultProfile();

    if (!S.ready) {
      S.ready = true;
      el.root.dataset.ready = "true";
      console.log("[LoRA Browser] ready: " + S.items.length + " LoRAs, " + S.rows.length + " active");
      // Only now is it safe to take the native controls out of the layout.
      hideNative(true);
      send({ type: "ready" });
    }
  }

  function renderHeader() {
    // A switched-off LoRA is listed but not applied, so it is not "active";
    // a group's row counts each LoRA in it.
    var loras = function (row) { return row.kind === "group" ? (row.members || []).length || 1 : 1; };
    var activeCount = 0;
    var offCount = 0;
    S.rows.forEach(function (row) {
      if (isOn(row)) { activeCount += loras(row); } else { offCount += loras(row); }
    });
    el.count.textContent = activeCount === 1 ? "1 active" : activeCount + " active";
    el.phase.textContent = S.phases.effective > 1 ? S.phases.effective + " phases" : "One phase";
    el.activeTitle.textContent = "Active LoRAs (" + activeCount + ")" + (offCount ? " \u00B7 " + offCount + " off" : "");
    el.disableAll.disabled = !S.canDisableAll;
    el.enableAll.disabled = !S.canEnableAll;
    el.viewSwitch.querySelectorAll("button").forEach(function (button) {
      button.setAttribute("aria-checked", button.dataset.view === S.view ? "true" : "false");
    });
    el.select.setAttribute("aria-pressed", S.selecting ? "true" : "false");
    el.select.classList.toggle("lb-on", S.selecting);
    el.select.disabled = !S.rows.length && !S.selecting;
    el.av.hidden = !S.avSupported;
    paintCollapseAll();
    paintPaste();

    el.sort.value = S.sortMode;
    el.naming.setAttribute("aria-pressed", S.nameMode === "civitai" ? "true" : "false");
    el.naming.title = S.nameMode === "civitai"
      ? "Showing Civitai names - click for filenames"
      : "Showing filenames - click for Civitai names";

    var current = S.activeProfile || "";
    el.profiles.innerHTML = "";
    var blank = document.createElement("option");
    blank.value = "";
    blank.textContent = S.profiles.length ? "Stack profiles..." : "No stack profiles";
    el.profiles.appendChild(blank);
    S.profiles.forEach(function (name) {
      var option = document.createElement("option");
      option.value = name;
      // The default for this model is marked so the dropdown alone tells you,
      // and so is a profile this model cannot supply every LoRA for -- it can
      // still be recalled, it just arrives short.
      var label = name;
      if (name === S.defaultProfile) { label += "  \u2605"; }
      if (S.profilesIncomplete.indexOf(name) !== -1) {
        label += "  \u26A0";
        option.title = "Some LoRAs in this profile are not available here";
      }
      option.textContent = label;
      el.profiles.appendChild(option);
    });
    el.profiles.value = current;
  }

  /* A default profile is applied only when arriving at a model with an empty
     stack, so it can never overwrite LoRAs that a preset or an import just
     put in place. */
  function maybeApplyDefaultProfile() {
    if (!S.defaultProfile || S.rows.length) { return; }
    if (S.autoDefaultApplied[S.modelKey]) { return; }
    S.autoDefaultApplied[S.modelKey] = true;
    setStatus("Applying default profile '" + S.defaultProfile + "'...");
    send({ type: "profile_recall", name: S.defaultProfile });
  }

  /* --------------------------------------------------------- diagnose */

  /* Everything needed to tell "panel never built" from "panel built but is
     not on screen" apart, without the user having to paste a script. */
  function diagnose() {
    var root = byId(IDS.root);
    var report = {
      instance: INSTANCE,
      ready: S.ready,
      mounted: S.mounted,
      items: S.items.length,
      activeRows: S.rows.length,
      forceNative: S.forceNative,
      layoutBroken: S.layoutBroken,
      root: null,
      chain: [],
      firstHidden: null,
      nativeControls: []
    };
    if (root) {
      var rect = root.getBoundingClientRect();
      var wrap = root.querySelector(".lb-grid-wrap");
      report.root = {
        id: root.id,
        width: Math.round(rect.width),
        height: Math.round(rect.height),
        onScreen: !!root.offsetParent,
        tiles: root.querySelectorAll(".lb-tile").length,
        gridHeight: wrap ? Math.round(wrap.getBoundingClientRect().height) : -1,
        cssApplied: wrap ? getComputedStyle(wrap).overflowY !== "visible" : false
      };
      var node = root, depth = 0;
      while (node && depth < 14) {
        var styles = getComputedStyle(node);
        var height = Math.round(node.getBoundingClientRect().height);
        var label = node.tagName.toLowerCase() + (node.id ? "#" + node.id : "") +
          "." + String(node.className || "").split(" ").filter(Boolean).slice(0, 2).join(".") +
          "[display=" + styles.display + ",h=" + height + "]";
        report.chain.push(label);
        if (!report.firstHidden && (styles.display === "none" || styles.visibility === "hidden" ||
            styles.opacity === "0" || (node !== root && height === 0))) {
          report.firstHidden = label;
        }
        node = node.parentElement;
        depth += 1;
      }
    }
    [IDS.nativeChoices, IDS.nativeMultipliers].forEach(function (id) {
      var node = byId(id);
      report.nativeControls.push(id + "=" + (node ? (node.offsetParent ? "visible" : "hidden") : "NOT FOUND"));
    });
    return report;
  }

  /* A panel that exists but collapsed to nothing is the worst failure mode:
     the user sees an empty tab. Detect it, say so loudly, and give the native
     controls back rather than leaving them with no LoRA UI. */
  function auditLayout() {
    if (!S.ready) { return; }
    var root = byId(IDS.root);
    if (!root) { return; }
    var block = root.closest(".block") || root.parentElement;
    var blockHeight = block ? block.getBoundingClientRect().height : 0;
    // The LoRAs tab is simply not open: zero height is expected, not a fault.
    if (blockHeight === 0) { return; }

    // The panel has padding and a border, so a collapsed one still measures
    // ~24px -- the content area is the honest signal. The grid always has
    // height when healthy, even with no results (it shows an empty state).
    var wrap = root.querySelector(".lb-grid-wrap");
    var wrapHeight = wrap ? wrap.getBoundingClientRect().height : 0;
    var collapsed = wrapHeight <= 4 || root.getBoundingClientRect().height < 48;
    if (collapsed === S.layoutBroken) { return; }

    S.layoutBroken = collapsed;
    if (collapsed) {
      console.warn("[LoRA Browser] panel rendered but has no height; restoring native controls.",
                   JSON.stringify(diagnose()));
    } else {
      console.log("[LoRA Browser] panel layout recovered.");
    }
    hideNative(S.ready);
  }

  /* ------------------------------------------------------------- boot */

  function boot() {
    if (!mount()) { return; }
    hideNative(S.ready);
    // MutationObserver catches most wipes instantly; this is the backstop for
    // anything that replaces the anchor without us seeing the mutation.
    if (!integrityTimer) {
      integrityTimer = setInterval(function () { checkIntegrity(); auditLayout(); }, 1000);
    }
  }

  window.wgpLoraBrowser[INSTANCE] = {
    apply: apply, applyMedia: applyMedia, boot: boot, diagnose: diagnose, state: S
  };

  /* One word in the browser console reports every instance. */
  window.wgpLoraBrowserDiagnose = function () {
    var all = Object.keys(window.wgpLoraBrowser).map(function (key) {
      return window.wgpLoraBrowser[key].diagnose();
    });
    var text = JSON.stringify({ instances: all }, null, 1);
    console.log(text);
    return text;
  };

  console.log("[LoRA Browser] frontend installed for instance " + INSTANCE);

  // Show the shell straight away rather than waiting for the first payload.
  boot();

  // Gradio does not guarantee that this script is installed before the first
  // payload lands in the bridge textbox, so pick up whatever is already there
  // instead of sitting idle until the next native change event.
  (function bootstrap(attempt) {
    var host = byId(IDS.payload);
    var field = host && host.querySelector("textarea, input");
    if (field && field.value) { apply(field.value); return; }
    if (attempt < 20) { setTimeout(function () { bootstrap(attempt + 1); }, 150); }
  })(0);
})();
