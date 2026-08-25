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
    sliderStep: 0.01,
    valueMin: -10,
    valueMax: 10,
    profiles: [],
    activeProfile: "",
    status: "",
    statusWarn: false,
    canRestore: false,
    thumbs: {},
    requested: {},
    pendingThumbs: [],
    linked: {},
    ready: false,
    mounted: false,
    forceNative: false,
    layoutBroken: false,
    remountQueued: false,
    dragging: false,
    lastSentSignature: null,
    signature: "",
    rowsKey: "",
    syncTimer: null,
    thumbTimer: null,
    mediaQueue: [],
    mediaInFlight: null,
    mediaTimer: null,
    playingTile: null,
    searchTerms: [],
    modal: null,
    pendingValues: {}
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
          '<button type="button" class="lb-btn" data-lb="disable-all">Disable all</button>' +
          '<button type="button" class="lb-btn" data-lb="restore" disabled>Restore</button>' +
        '</span>' +
      '</div>' +
      '<div class="lb-rows" data-lb="rows"></div>' +
      '<div class="lb-statusbar">' +
        '<span class="lb-status" data-lb="status"></span>' +
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
      activeTitle: pick("active-title"), disableAll: pick("disable-all"),
      restore: pick("restore"), rows: pick("rows"), status: pick("status"),
      nativeToggle: pick("native")
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

    el.disableAll.addEventListener("click", function () { send({ type: "disable_all" }); });
    el.restore.addEventListener("click", function () { send({ type: "restore" }); });

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
        } }
    ];
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
      { label: "Inspect", disabled: !item.has_catalogue,
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
      { label: item.active ? "Exclude LoRA" : "Include LoRA", run: function () { send({ type: "toggle", id: id, enabled: !item.active }); } }
    ];
  }

  function copyText(value) {
    // Only the basename is ever copied; absolute paths stay on the Python side.
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(value).then(function () { setStatus("Copied " + value); });
      return;
    }
    var field = document.createElement("textarea");
    field.value = value;
    document.body.appendChild(field);
    field.select();
    try { document.execCommand("copy"); setStatus("Copied " + value); } catch (error) { /* ignore */ }
    document.body.removeChild(field);
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
    tile.className = "lb-tile";
    tile.dataset.id = item.id;
    tile.tabIndex = 0;
    tile.setAttribute("role", "option");
    tile.setAttribute("aria-selected", item.active ? "true" : "false");
    var tooltip = [displayName(item), item.name];
    if (item.parent) { tooltip.push(item.parent); }
    if ((item.words || []).length) { tooltip.push("Triggers: " + item.words.join(", ")); }
    tile.title = tooltip.filter(Boolean).join("\n");

    var thumb = S.thumbs[item.id];
    var media;
    if (thumb) {
      media = document.createElement("img");
      media.className = "lb-thumb";
      media.loading = "lazy";
      media.alt = "";
      media.src = thumb;
    } else {
      media = document.createElement("div");
      media.className = "lb-thumb lb-placeholder";
      media.setAttribute("aria-hidden", "true");
      media.textContent = "\u25A6";
    }
    tile.appendChild(media);

    var badges = document.createElement("div");
    badges.className = "lb-badges";
    var check = document.createElement("span");
    if (item.active) {
      check.className = "lb-check";
      check.textContent = "\u2713";
    }
    badges.appendChild(check);
    if (item.favorite) {
      var star = document.createElement("span");
      star.className = "lb-star";
      star.textContent = "\u2605";
      badges.appendChild(star);
    }
    tile.appendChild(badges);

    // A tile whose preview is a video gets a play control. The still frame
    // stays the thumbnail; the video itself is only fetched on click.
    if (item.has_video) {
      var play = document.createElement("button");
      play.type = "button";
      play.className = "lb-play";
      play.title = "Play preview";
      play.setAttribute("aria-label", "Play preview for " + displayName(item));
      play.textContent = "\u25B6";
      play.addEventListener("click", function (event) {
        event.stopPropagation();          // play, do not toggle the LoRA
        togglePreviewVideo(item.id, tile);
      });
      tile.appendChild(play);
    }

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

  /* Swap a tile's still frame for a muted looping video, and back again.
     Only one plays at a time and the bytes are fetched on demand, so a library
     full of video previews costs nothing until something is clicked. */
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
      var still = document.createElement("div");
      still.className = "lb-thumb lb-placeholder";
      still.setAttribute("aria-hidden", "true");
      still.textContent = "\u25A6";
      var id = tile.dataset.id;
      if (S.thumbs[id]) {
        still = document.createElement("img");
        still.className = "lb-thumb";
        still.loading = "lazy";
        still.alt = "";
        still.src = S.thumbs[id];
      }
      video.replaceWith(still);
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
    var padding = 12;
    var height = visibleRows * tileHeight + (visibleRows - 1) * gap + padding;

    el.gridWrap.style.height = Math.ceil(height) + "px";
    el.gridWrap.style.maxHeight = Math.ceil(height) + "px";
    el.gridWrap.style.overflowY = rowCount > MAX_VISIBLE_ROWS ? "auto" : "hidden";
  }

  function observeResize() {
    if (typeof ResizeObserver === "function") {
      resizeObserver = new ResizeObserver(function () { capHeight(); });
      resizeObserver.observe(el.gridWrap);
      return;
    }
    window.addEventListener("resize", capHeight);
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
  function requestMedia(request, callback) {
    S.mediaQueue.push({ request: request, callback: callback });
    pumpMedia();
  }

  function pumpMedia() {
    if (S.mediaInFlight || !S.mediaQueue.length) { return; }
    var job = S.mediaQueue.shift();
    S.mediaInFlight = job;
    if (!setHidden(IDS.thumbReq, JSON.stringify(job.request))) {
      S.mediaInFlight = null;
      return;
    }
    clickHidden(IDS.thumbBtn);
    S.mediaTimer = setTimeout(function () { finishMedia(null); }, 30000);
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
    finishMedia(payload);
  }

  function applyThumbBatch(payload) {
    var map = payload.thumbs || {};
    Object.keys(map).forEach(function (id) {
      if (map[id]) { S.thumbs[id] = map[id]; }
    });

    // Swap placeholders in place; rebuilding the grid here would fight scroll.
    Object.keys(map).forEach(function (id) {
      if (!map[id]) { return; }
      var tile = el.grid.querySelector('.lb-tile[data-id="' + cssEscape(id) + '"]');
      if (!tile) { return; }
      var placeholder = tile.querySelector(".lb-thumb");
      if (!placeholder || placeholder.tagName !== "DIV") { return; }
      var image = document.createElement("img");
      image.className = "lb-thumb";
      image.loading = "lazy";
      image.alt = "";
      image.src = map[id];
      placeholder.replaceWith(image);
    });
    if (S.pendingThumbs.length) { flushThumbnails(); }
  }

  function cssEscape(value) {
    if (window.CSS && window.CSS.escape) { return window.CSS.escape(value); }
    return String(value).replace(/["\\]/g, "\\$&");
  }

  /* ------------------------------------------------------ active rows */

  function rowsKey() {
    return S.rows.map(function (row) {
      return row.id + ":" + row.multiplier_kind + ":" + (row.phase_values || []).length + ":" + (row.missing ? "m" : "") + (row.system_managed ? "s" : "");
    }).join("|") + "#" + S.phases.effective;
  }

  function renderRows(force) {
    if (!el.rows) { return; }
    // Never rebuild the DOM under a pointer that is dragging a slider.
    if (S.dragging) { return; }

    var key = rowsKey();
    if (!force && key === S.rowsKey) {
      updateRowValues();
      return;
    }
    S.rowsKey = key;

    var fragment = document.createDocumentFragment();
    S.rows.forEach(function (row) { fragment.appendChild(buildRow(row)); });
    el.rows.innerHTML = "";
    if (!S.rows.length) {
      var empty = document.createElement("div");
      empty.className = "lb-empty";
      empty.textContent = "No LoRAs are active. Click a thumbnail to include one.";
      el.rows.appendChild(empty);
    } else {
      el.rows.appendChild(fragment);
    }
  }

  function updateRowValues() {
    S.rows.forEach(function (row) {
      var node = el.rows.querySelector('.lb-row[data-id="' + cssEscape(row.id) + '"]');
      if (!node) { return; }
      (row.phase_values || []).forEach(function (value, index) {
        var range = node.querySelector('input[type="range"][data-phase="' + index + '"]');
        var number = node.querySelector('input[type="number"][data-phase="' + index + '"]');
        if (range && document.activeElement !== range) { syncSliderState(range, value); }
        if (number && document.activeElement !== number) { number.value = fmt(value); }
      });
    });
  }

  function buildRow(row) {
    var node = document.createElement("div");
    node.className = "lb-row" + (row.missing ? " lb-missing" : "");
    node.dataset.id = row.id;

    // The name is truncated so the strength controls get the width; the full
    // name lives in the tooltip and the Inspect view.
    var inspect = document.createElement("button");
    inspect.type = "button";
    inspect.className = "lb-btn lb-icon lb-inspect";
    inspect.textContent = "i";
    inspect.title = "Inspect " + (row.civitai_name || row.name);
    inspect.setAttribute("aria-label", inspect.title);
    inspect.disabled = !row.has_catalogue;
    inspect.addEventListener("click", function () { openInspect(row.id); });
    node.appendChild(inspect);

    var name = document.createElement("div");
    name.className = "lb-row-name";
    name.textContent = rowName(row);
    name.title = [row.civitai_name, row.name].filter(Boolean).join("\n");
    if (row.system_managed) { name.appendChild(note("WanGP-managed")); }
    if (row.missing) { name.appendChild(note("Missing locally")); }
    node.appendChild(name);

    if (row.multiplier_kind === "advanced") {
      var advanced = document.createElement("code");
      advanced.className = "lb-advanced";
      advanced.title = "Advanced step schedule, preserved as-is";
      advanced.textContent = row.multiplier_raw;
      node.appendChild(advanced);

      var convert = document.createElement("button");
      convert.type = "button";
      convert.className = "lb-btn";
      convert.textContent = "Use sliders";
      convert.title = "Replace this schedule with a simple multiplier";
      convert.addEventListener("click", function () {
        if (!window.confirm("Replace the advanced schedule '" + row.multiplier_raw + "' with a simple multiplier?")) { return; }
        send({ type: "convert_simple", id: row.id });
      });
      node.appendChild(convert);
    } else {
      node.appendChild(buildPhases(row));
    }

    var remove = document.createElement("button");
    remove.type = "button";
    remove.className = "lb-btn lb-icon";
    remove.textContent = "\u00D7";
    remove.title = "Remove " + row.name;
    remove.setAttribute("aria-label", "Remove " + row.name);
    remove.addEventListener("click", function () { send({ type: "remove", id: row.id }); });
    node.appendChild(remove);
    return node;
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

  function buildPhases(row) {
    var wrap = document.createElement("div");
    wrap.className = "lb-phases";
    var values = row.phase_values || [];

    if (values.length > 1) {
      var link = document.createElement("button");
      link.type = "button";
      link.className = "lb-btn lb-icon";
      var linked = !!S.linked[row.id];
      link.textContent = linked ? "\u26AD" : "\u26AE";
      link.title = linked ? "Phases linked" : "Phases independent";
      link.setAttribute("aria-pressed", linked ? "true" : "false");
      link.addEventListener("click", function () {
        S.linked[row.id] = !S.linked[row.id];
        renderRows(true);
      });
      wrap.appendChild(link);
    }

    values.forEach(function (value, index) {
      var phase = document.createElement("div");
      phase.className = "lb-phase";

      var label = document.createElement("label");
      var fieldId = "lb-" + INSTANCE + "-" + index + "-" + Math.random().toString(36).slice(2, 8);
      label.textContent = S.phaseLabels[index] || ("Phase " + (index + 1));
      label.htmlFor = fieldId;
      phase.appendChild(label);

      var range = document.createElement("input");
      range.type = "range";
      range.dataset.phase = String(index);
      range.min = String(S.sliderMin);
      range.max = String(S.sliderMax);
      range.step = String(S.sliderStep);
      range.value = String(clampToSlider(value));
      range.setAttribute("aria-label", (row.name + " " + (S.phaseLabels[index] || "")).trim());

      var number = document.createElement("input");
      number.type = "number";
      number.id = fieldId;
      number.dataset.phase = String(index);
      number.step = "0.01";
      number.min = String(S.valueMin);
      number.max = String(S.valueMax);
      number.value = fmt(value);

      // Full-height stepper buttons: easier to hit than the native spinners,
      // and they must not change the row height.
      var down = stepButton("\u2212", "Decrease by 0.01");
      var up = stepButton("+", "Increase by 0.01");
      down.addEventListener("click", function () { nudge(row, index, range, number, -0.01); });
      up.addEventListener("click", function () { nudge(row, index, range, number, 0.01); });

      wirePhaseInputs(row, index, range, number);
      phase.appendChild(range);
      phase.appendChild(down);
      phase.appendChild(number);
      phase.appendChild(up);
      syncSliderState(range, number.value);
      wrap.appendChild(phase);
    });
    return wrap;
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

  function clampToSlider(value) {
    var number = Number(value);
    if (!isFinite(number)) { return S.sliderMin; }
    return Math.min(S.sliderMax, Math.max(S.sliderMin, number));
  }

  /* The slider only covers 0..1. Outside that the value is still valid, so the
     numeric field keeps it and the slider disables itself rather than
     pretending to represent a number it cannot reach. */
  function syncSliderState(range, rawValue) {
    var number = Number(rawValue);
    var outside = !isFinite(number) || number < S.sliderMin || number > S.sliderMax;
    range.disabled = outside;
    range.classList.toggle("lb-out-of-range", outside);
    range.title = outside
      ? "Value is outside the slider range; edit it in the number field"
      : "";
    if (!outside) { range.value = String(number); }
  }

  function nudge(row, index, range, number, delta) {
    var current = Number(number.value);
    if (!isFinite(current)) { current = 1; }
    var next = Math.round((current + delta) * 100) / 100;
    if (next < S.valueMin || next > S.valueMax) { return; }
    number.value = fmt(next);
    syncSliderState(range, next);
    commitValue(row, index, next, range, number);
    flushSync();
  }

  /* Record a value locally and schedule the native sync. The row DOM is never
     rebuilt here, so a drag keeps its pointer capture. */
  function commitValue(row, index, value, range, number) {
    var linked = !!S.linked[row.id];
    if (linked) {
      var parent = range.closest(".lb-phases");
      parent.querySelectorAll('input[type="range"]').forEach(function (other) {
        other.value = String(clampToSlider(value));
        syncSliderState(other, value);
      });
      parent.querySelectorAll('input[type="number"]').forEach(function (other) {
        other.value = fmt(value);
      });
    }
    S.pendingValues[row.id + "#" + (linked ? "all" : index)] = {
      type: "set_strength", id: row.id, phase: index, value: Number(value), linked: linked
    };
    scheduleSync();
  }

  function wirePhaseInputs(row, index, range, number) {
    // Remember the last accepted value so an out-of-range entry can be undone.
    var accepted = Number(number.value);

    range.addEventListener("pointerdown", function () { S.dragging = true; });
    range.addEventListener("input", function () {
      var value = Number(range.value);
      number.value = fmt(value);
      accepted = value;
      commitValue(row, index, value, range, number);
    });
    var settle = function () {
      S.dragging = false;
      flushSync();
    };
    range.addEventListener("change", settle);
    range.addEventListener("pointerup", settle);
    range.addEventListener("pointercancel", settle);

    // Typing is not committed until the field settles: partial input like "-"
    // or "1.5e" would otherwise be pushed to WanGP mid-keystroke.
    number.addEventListener("input", function () {
      syncSliderState(range, number.value);
    });

    var applyTyped = function () {
      var value = Number(number.value);
      if (number.value.trim() === "" || !isFinite(value) ||
          value < S.valueMin || value > S.valueMax) {
        // Out of bounds or unparseable: restore the previous value.
        number.value = fmt(accepted);
        syncSliderState(range, accepted);
        setStatus("Strength must be between " + fmt(S.valueMin) + " and " + fmt(S.valueMax) + ".", true);
        return;
      }
      value = Math.round(value * 100) / 100;
      number.value = fmt(value);
      accepted = value;
      syncSliderState(range, value);
      commitValue(row, index, value, range, number);
      flushSync();
    };

    number.addEventListener("change", applyTyped);
    number.addEventListener("blur", applyTyped);
    number.addEventListener("keydown", function (event) {
      if (event.key === "Enter") { event.preventDefault(); applyTyped(); }
    });
  }

  function scheduleSync() {
    clearTimeout(S.syncTimer);
    S.syncTimer = setTimeout(flushSync, SYNC_DEBOUNCE_MS);
  }

  function flushSync() {
    clearTimeout(S.syncTimer);
    var actions = Object.keys(S.pendingValues).map(function (key) { return S.pendingValues[key]; });
    if (!actions.length) { return; }
    S.pendingValues = {};
    send({ type: "batch", actions: actions });
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

    if (!detail || detail.error) {
      title.textContent = (S.byId[id] || {}).name || id;
      side.textContent = (detail && detail.error) ||
        "No catalogue data was found for this LoRA.";
      return;
    }

    title.textContent = detail.civitai_name || detail.name || id;

    side.innerHTML = "";
    side.appendChild(inspectFacts(detail));
    if ((detail.trained_words || []).length) {
      side.appendChild(inspectWords(detail.trained_words));
    }
    if (detail.description) {
      side.appendChild(inspectBlock("Description", detail.description));
    }
    if (detail.version_description) {
      side.appendChild(inspectBlock("Version notes", detail.version_description));
    }

    media.innerHTML = "";
    if (!(detail.media || []).length) {
      var empty = document.createElement("div");
      empty.className = "lb-empty";
      empty.textContent = "No media was downloaded for this LoRA.";
      media.appendChild(empty);
      return;
    }
    detail.media.forEach(function (item) {
      media.appendChild(inspectMediaCard(id, item));
    });
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
    // Ignore a stale async response that lost the race with a newer sync.
    if (payload.revision !== undefined && payload.revision < S.revision) { return; }

    var previousModel = S.modelKey;
    S.revision = payload.revision || 0;
    S.modelKey = payload.model_key || "";
    S.phases = payload.phases || { capacity: 1, effective: 1 };
    S.phaseLabels = payload.phase_labels || ["Strength"];
    S.items = payload.items || [];
    S.rows = payload.active || [];
    S.profiles = payload.profiles || [];
    S.activeProfile = payload.active_profile || "";
    S.canRestore = !!payload.can_restore;
    S.signature = payload.signature || "";
    S.sliderMin = payload.slider_min !== undefined ? payload.slider_min : 0;
    S.sliderMax = payload.slider_max !== undefined ? payload.slider_max : 1;
    S.sliderStep = payload.slider_step || 0.01;
    S.valueMin = payload.value_min !== undefined ? payload.value_min : -10;
    S.valueMax = payload.value_max !== undefined ? payload.value_max : 10;
    S.defaultProfile = payload.default_profile || "";
    if (payload.sort_mode) { S.sortMode = payload.sort_mode; }
    if (payload.name_mode) { S.nameMode = payload.name_mode; }

    if (previousModel && previousModel !== S.modelKey) {
      // The model changed: previews and per-LoRA UI state no longer apply.
      S.thumbs = {};
      S.requested = {};
      S.pendingThumbs = [];
      S.linked = {};
      S.playingTile = null;
      closeInspect();
    }

    S.byId = {};
    S.items.forEach(function (item) { S.byId[item.id] = item; });
    S.rows.forEach(function (row) {
      if (S.linked[row.id] === undefined && row.linked) { S.linked[row.id] = true; }
    });

    if (payload.zoom_px && !S.ready) { S.applyZoom(payload.zoom_px, false); }
    else { S.applyZoom(S.zoomPx, false); }

    renderHeader();
    renderGrid();
    renderRows(true);
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
    var activeCount = S.rows.length;
    el.count.textContent = activeCount === 1 ? "1 active" : activeCount + " active";
    el.phase.textContent = S.phases.effective > 1 ? S.phases.effective + " phases" : "One phase";
    el.activeTitle.textContent = "Active LoRAs (" + activeCount + ")";
    el.restore.disabled = !S.canRestore;

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
      // The default for this model is marked so the dropdown alone tells you.
      option.textContent = name === S.defaultProfile ? name + "  \u2605" : name;
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
