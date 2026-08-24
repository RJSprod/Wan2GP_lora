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
    zoomPx: 104,
    sliderMin: 0,
    sliderMax: 2,
    sliderStep: 0.001,
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
    remountQueued: false,
    dragging: false,
    lastSentSignature: null,
    signature: "",
    rowsKey: "",
    syncTimer: null,
    thumbTimer: null,
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
  function nativeBlocks() {
    var blocks = [];
    [IDS.nativeChoices, IDS.nativeMultipliers].forEach(function (id) {
      var node = byId(id);
      if (node) { blocks.push(node.closest(".block, .form") || node); }
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
    var conceal = !!hide && panelHasContent() && !S.forceNative;
    nativeBlocks().forEach(function (block) {
      block.style.display = conceal ? "none" : "";
    });
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
          '<select class="lb-select" data-lb="profiles" aria-label="Stack profiles"></select>' +
          '<button type="button" class="lb-btn" data-lb="save">Save as...</button>' +
          '<button type="button" class="lb-btn lb-icon" data-lb="manage" title="Manage profiles" aria-label="Manage profiles">...</button>' +
          '<button type="button" class="lb-btn" data-lb="refresh">Refresh</button>' +
        '</div>' +
      '</div>' +
      '<div class="lb-toolbar">' +
        '<input class="lb-input lb-search" type="search" data-lb="search" placeholder="Filter files" aria-label="Filter LoRAs">' +
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
      save: pick("save"), manage: pick("manage"), refresh: pick("refresh"),
      search: pick("search"), zoom: pick("zoom"), zoomValue: pick("zoom-value"),
      zoomIn: pick("zoom-in"), zoomOut: pick("zoom-out"),
      gridWrap: pick("grid-wrap"), grid: pick("grid"),
      activeTitle: pick("active-title"), disableAll: pick("disable-all"),
      restore: pick("restore"), rows: pick("rows"), status: pick("status"),
      nativeToggle: pick("native")
    };

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
      S.search = el.search.value.trim().toLowerCase();
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
    return [
      { label: "Update selected", disabled: !selected, run: function () { send({ type: "profile_update", name: selected }); } },
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

  function matchesSearch(item) {
    if (!S.search) { return true; }
    var haystack = (item.name + " " + (item.tag || "") + " " + (item.parent || "")).toLowerCase();
    return haystack.indexOf(S.search) >= 0;
  }

  function renderGrid() {
    if (!el.grid) { return; }
    var visible = S.items.filter(matchesSearch);

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
    tile.title = item.parent ? item.name + "\n" + item.parent : item.name;

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

    // textContent, never innerHTML: filenames and tags are user-controlled.
    var name = document.createElement("div");
    name.className = "lb-tile-name";
    name.textContent = item.label || item.name;
    tile.appendChild(name);

    if (item.tag) {
      var tag = document.createElement("span");
      tag.className = "lb-tag";
      tag.textContent = item.tag;
      tile.appendChild(tag);
    }
    return tile;
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
      var batch = S.pendingThumbs.splice(0, 48);
      if (!batch.length) { return; }
      if (setHidden(IDS.thumbReq, JSON.stringify({ ids: batch }))) { clickHidden(IDS.thumbBtn); }
    }, 90);
  }

  function applyThumbs(json) {
    var payload;
    try { payload = JSON.parse(json || "{}"); } catch (error) { return; }
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
      if (!placeholder || placeholder.tagName === "IMG") { return; }
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
        if (range && document.activeElement !== range) { range.value = String(value); }
        if (number && document.activeElement !== number) { number.value = fmt(value); }
      });
    });
  }

  function buildRow(row) {
    var node = document.createElement("div");
    node.className = "lb-row" + (row.missing ? " lb-missing" : "");
    node.dataset.id = row.id;

    var name = document.createElement("div");
    name.className = "lb-row-name";
    name.textContent = row.label || row.name;
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
      // An imported value beyond the slider range widens the slider rather
      // than clamping the stored value.
      range.max = String(Math.max(S.sliderMax, value));
      range.step = String(S.sliderStep);
      range.value = String(value);
      range.setAttribute("aria-label", (row.name + " " + (S.phaseLabels[index] || "")).trim());

      var number = document.createElement("input");
      number.type = "number";
      number.id = fieldId;
      number.dataset.phase = String(index);
      number.step = "0.01";
      number.value = fmt(value);

      wirePhaseInputs(row, index, range, number);
      phase.appendChild(range);
      phase.appendChild(number);
      wrap.appendChild(phase);
    });
    return wrap;
  }

  function wirePhaseInputs(row, index, range, number) {
    var live = function (value, source) {
      var linked = !!S.linked[row.id];
      // Update the paired field in place; the row DOM is never rebuilt here,
      // so a drag keeps its pointer capture.
      if (source !== number) { number.value = fmt(value); }
      if (source !== range) { range.value = String(value); }
      if (linked) {
        var parent = range.closest(".lb-phases");
        parent.querySelectorAll('input[type="range"]').forEach(function (other) { other.value = String(value); });
        parent.querySelectorAll('input[type="number"]').forEach(function (other) { other.value = fmt(value); });
      }
      S.pendingValues[row.id + "#" + (linked ? "all" : index)] = {
        type: "set_strength", id: row.id, phase: index, value: Number(value), linked: linked
      };
      scheduleSync();
    };

    range.addEventListener("pointerdown", function () { S.dragging = true; });
    range.addEventListener("input", function () { live(Number(range.value), range); });
    var settle = function () {
      S.dragging = false;
      flushSync();
    };
    range.addEventListener("change", settle);
    range.addEventListener("pointerup", settle);
    range.addEventListener("pointercancel", settle);

    number.addEventListener("input", function () {
      var value = Number(number.value);
      if (!isFinite(value)) { return; }
      if (value > Number(range.max)) { range.max = String(value); }
      live(value, number);
    });
    number.addEventListener("change", flushSync);
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

    document.addEventListener("click", closeMenu, { once: true });
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
    S.sliderMax = payload.slider_max !== undefined ? payload.slider_max : 2;
    S.sliderStep = payload.slider_step || 0.001;

    if (previousModel && previousModel !== S.modelKey) {
      // The model changed: previews and per-LoRA UI state no longer apply.
      S.thumbs = {};
      S.requested = {};
      S.pendingThumbs = [];
      S.linked = {};
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

    var current = S.activeProfile || "";
    el.profiles.innerHTML = "";
    var blank = document.createElement("option");
    blank.value = "";
    blank.textContent = S.profiles.length ? "Stack profiles..." : "No stack profiles";
    el.profiles.appendChild(blank);
    S.profiles.forEach(function (name) {
      var option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      el.profiles.appendChild(option);
    });
    el.profiles.value = current;
  }

  /* ------------------------------------------------------------- boot */

  function boot() {
    if (!mount()) { return; }
    hideNative(S.ready);
    // MutationObserver catches most wipes instantly; this is the backstop for
    // anything that replaces the anchor without us seeing the mutation.
    if (!integrityTimer) {
      integrityTimer = setInterval(checkIntegrity, 1000);
    }
  }

  window.wgpLoraBrowser[INSTANCE] = { apply: apply, applyThumbs: applyThumbs, boot: boot, state: S };

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
