"""On-demand Civitai enrichment for a single LoRA.

This is the network half of the standalone ``process_*_lora.py`` enrichment
scripts, moved inside the plugin so a LoRA with no catalogue can be filled in
from the Inspect view instead of by re-running a script across a whole folder.

What deliberately did *not* come across is those scripts' model-family
detection.  A script that walks a directory has to decide which files it owns;
here the user has already pointed at one file, so the only question left is
whether Civitai recognises its SHA-256.  Dropping the detection is what makes
this work for every family WanGP can load -- MiniMax H3, the LTX 2 line, Wan --
rather than for one.

Every run is additive.  Media already on disk is never downloaded again, while
the JSON and summary documents are rebuilt each time, so a sidecar written by
an older script (or by hand) ends up in the shape :mod:`.catalogue` reads.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from . import catalogue as cat
from .utils import is_within

BASE_URL = "https://civitai.com/api/v1"
USER_AGENT = "WanGP-LoRA-Browser/1.0"

DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 2

#: Only Civitai's own hosts are ever contacted.  Media URLs arrive inside a
#: remote JSON document; treating them as arbitrary fetch targets would let a
#: tampered response aim this at a host the user never asked for.
_ALLOWED_HOST = "civitai.com"
_ALLOWED_HOST_SUFFIX = ".civitai.com"

#: Downloads are named ``NNN`` plus an extension chosen from this table -- never
#: from the URL path verbatim -- so nothing a remote document supplies decides
#: what lands on disk.
_EXTENSION_BY_MIME = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/avif": ".avif",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
}
_MEDIA_EXTENSIONS = frozenset(cat.IMAGE_EXTENSIONS + cat.VIDEO_EXTENSIONS)

HASH_CHUNK = 1024 * 1024
DOWNLOAD_CHUNK = 1024 * 1024

#: No single Civitai preview is legitimately this large.  The cap bounds what
#: one click can write to disk.
MEDIA_SIZE_LIMIT = 128 * 1024 * 1024


class _TooLarge(Exception):
    """A download that blew the size cap. Never worth retrying."""


@dataclass
class FetchReport:
    """Outcome of one fetch, in the terms the status line needs."""

    ok: bool = False
    message: str = ""
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    media_total: int = 0
    name: str = ""
    #: Civitai has no record of this exact file. Distinguished from a failure
    #: because a bulk run must be able to say "not on Civitai" rather than
    #: "something went wrong".
    not_found: bool = False


# ------------------------------------------------------------------ http


#: Matched by the bulk runner, so keep it a constant rather than a literal.
_NOT_FOUND = "Civitai does not have this exact file"


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def is_civitai_url(url: str) -> bool:
    """True only for ``https://`` URLs on Civitai's own hosts."""
    try:
        parsed = urlparse(str(url or ""))
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and (
        host == _ALLOWED_HOST or host.endswith(_ALLOWED_HOST_SUFFIX)
    )


def _retry_delay(error: HTTPError, attempt: int) -> float:
    raw = error.headers.get("Retry-After") if error.headers else None
    try:
        delay = float(raw) if raw else 1.5 * (attempt + 1)
    except (TypeError, ValueError):
        delay = 1.5 * (attempt + 1)
    return min(delay, 10.0)


def _is_retryable(status: int) -> bool:
    return status == 429 or 500 <= status <= 599


def get_json(
    url: str,
    *,
    api_key: str = "",
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> tuple[dict[str, Any] | None, str]:
    """``(payload, error)``. 404 and auth failures are final; 429/5xx retry."""
    if not is_civitai_url(url):
        return None, "refused a non-Civitai URL"

    last = "unknown Civitai error"
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers=_headers(api_key), method="GET")
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                return None, "Civitai returned an unexpected document"
            return payload, ""

        except HTTPError as error:
            if error.code == 404:
                return None, _NOT_FOUND
            if error.code in (401, 403):
                return None, (
                    f"Civitai refused the request (HTTP {error.code}); "
                    "this model may need an API key"
                )
            last = f"Civitai HTTP {error.code}"
            if not _is_retryable(error.code) or attempt >= retries:
                return None, last
            time.sleep(_retry_delay(error, attempt))

        except (URLError, TimeoutError, OSError) as error:
            last = f"could not reach Civitai ({error})"
            if attempt >= retries:
                return None, last
            time.sleep(1.5 * (attempt + 1))

        except (UnicodeDecodeError, ValueError) as error:
            return None, f"Civitai sent invalid JSON ({error})"

    return None, last


# ----------------------------------------------------------------- files


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: str, text: str) -> None:
    """Write via a temporary sibling so an interrupted run leaves no half file."""
    directory = os.path.dirname(path) or "."
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=directory, prefix=".civitai-", suffix=".tmp", delete=False
    )
    try:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)


def _atomic_write_json(path: str, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _extension_for(url: str, content_type: str, declared_kind: str) -> str:
    """Pick a safe extension from the URL, the Content-Type, then the record."""
    suffix = os.path.splitext(urlparse(url).path)[1].lower()
    if suffix in _MEDIA_EXTENSIONS:
        return suffix

    mime = str(content_type or "").split(";", 1)[0].strip().lower()
    if mime in _EXTENSION_BY_MIME:
        return _EXTENSION_BY_MIME[mime]
    guessed = (mimetypes.guess_extension(mime) or "").lower() if mime else ""
    if guessed == ".jpe":
        guessed = ".jpg"
    if guessed in _MEDIA_EXTENSIONS:
        return guessed

    return {"image": ".jpg", "video": ".mp4"}.get(declared_kind, "")


def existing_media(media_dir: str, index: int) -> str:
    """Path of media already downloaded for ``index``, or ``""``.

    This is what makes a re-fetch additive: an entry that is already on disk is
    never requested again, whatever extension a previous run chose for it.
    """
    prefix = f"{index:03d}"
    try:
        names = sorted(os.listdir(media_dir))
    except OSError:
        return ""
    for name in names:
        stem, extension = os.path.splitext(name)
        if stem == prefix and extension.lower() in _MEDIA_EXTENSIONS:
            return os.path.join(media_dir, name)
    return ""


def download_media(
    url: str,
    *,
    media_dir: str,
    index: int,
    declared_kind: str = "",
    api_key: str = "",
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> tuple[str, str]:
    """``(path, error)`` for one media URL, written as ``NNN.<ext>``."""
    if not is_civitai_url(url):
        return "", "not a Civitai URL"

    prefix = f"{index:03d}"
    last = "download failed"

    for attempt in range(retries + 1):
        partial = ""
        try:
            headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            request = Request(url, headers=headers, method="GET")

            with urlopen(request, timeout=timeout) as response:
                extension = _extension_for(
                    url, response.headers.get("Content-Type", ""), declared_kind
                )
                if not extension:
                    return "", "unrecognised media type"

                destination = os.path.join(media_dir, f"{prefix}{extension}")
                partial = destination + ".part"
                written = 0
                with open(partial, "wb") as out:
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > MEDIA_SIZE_LIMIT:
                            raise _TooLarge
                        out.write(chunk)

            os.replace(partial, destination)
            partial = ""
            return destination, ""

        except _TooLarge:
            return "", "media is larger than the size limit"

        except HTTPError as error:
            last = f"HTTP {error.code}"
            if not _is_retryable(error.code) or attempt >= retries:
                break
            time.sleep(_retry_delay(error, attempt))

        except (URLError, TimeoutError, OSError, ValueError) as error:
            last = str(error)
            if attempt >= retries:
                break
            time.sleep(1.5 * (attempt + 1))

        finally:
            if partial and os.path.exists(partial):
                try:
                    os.remove(partial)
                except OSError:
                    pass

    return "", last


# ------------------------------------------------------------- documents


def media_entries(version: dict[str, Any]) -> list[dict[str, Any]]:
    """The version's media records, de-duplicated by URL and order-preserving."""
    raw = version.get("images")
    if not isinstance(raw, list):
        return []

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        entries.append(item)
    return entries


def trained_words(version: dict[str, Any]) -> list[str]:
    raw = version.get("trainedWords")
    if not isinstance(raw, list):
        return []
    return [str(word) for word in raw if str(word).strip()]


def _creator(model: dict[str, Any]) -> str:
    creator = model.get("creator")
    return str(creator.get("username", "") or "") if isinstance(creator, dict) else ""


def build_summary(
    *,
    source_name: str,
    file_hash: str,
    version: dict[str, Any],
    model: dict[str, Any],
    media_total: int,
    media_local: int,
) -> str:
    """The cheap index document, in the exact shape ``parse_summary`` reads."""
    model_id = version.get("modelId") or model.get("id") or ""
    version_id = version.get("id") or ""

    lines = [
        f"Local file: {source_name}",
        f"SHA256: {file_hash}",
        "",
        f"Civitai model name: {model.get('name') or _version_model_name(version)}",
        f"Civitai model ID: {model_id}",
        f"Civitai version name: {version.get('name') or ''}",
        f"Civitai version ID: {version_id}",
        f"Base model: {version.get('baseModel') or ''}",
        f"Creator: {_creator(model)}",
        "",
        "Trained words:",
    ]

    words = trained_words(version)
    if words:
        lines.extend(f"- {word}" for word in words)
    else:
        lines.append("(none supplied by Civitai)")

    lines += [
        "",
        f"Version media entries: {media_total}",
        f"Media available locally after this run: {media_local}",
    ]

    if model_id and version_id:
        lines += ["", f"Civitai page: https://civitai.com/models/{model_id}?modelVersionId={version_id}"]
    elif model_id:
        lines += ["", f"Civitai page: https://civitai.com/models/{model_id}"]

    return "\n".join(lines) + "\n"


def _version_model_name(version: dict[str, Any]) -> str:
    """Version records carry a small ``model`` stub; use it when the full record
    could not be fetched."""
    stub = version.get("model")
    return str(stub.get("name", "") or "") if isinstance(stub, dict) else ""


def _write_documents(
    directory: str,
    stem: str,
    *,
    source_name: str,
    file_hash: str,
    version: dict[str, Any],
    model: dict[str, Any],
    media_total: int,
    media_local: int,
) -> None:
    _atomic_write_json(os.path.join(directory, "modelVersion.json"), version)
    if model:
        _atomic_write_json(os.path.join(directory, "model.json"), model)

    # Rebuilt rather than merged: the combined document is the one the Inspect
    # view reads, and an older or hand-made file may be missing half of it.
    _atomic_write_json(
        os.path.join(directory, f"{stem}.json"),
        {
            "sourceFile": source_name,
            "sha256": file_hash,
            "fetchedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "fetchedBy": USER_AGENT,
            "modelVersion": version,
            "model": model or None,
        },
    )

    words = trained_words(version)
    _atomic_write_text(
        os.path.join(directory, f"{stem}.txt"), "\n".join(words) + ("\n" if words else "")
    )
    _atomic_write_text(
        os.path.join(directory, cat.SUMMARY_FILENAME),
        build_summary(
            source_name=source_name,
            file_hash=file_hash,
            version=version,
            model=model,
            media_total=media_total,
            media_local=media_local,
        ),
    )


def _has_preview(directory: str, stem: str, extensions) -> bool:
    try:
        listing = {name.lower() for name in os.listdir(directory)}
    except OSError:
        # Cannot tell what is there, so do not risk writing over anything.
        return True
    lowered = stem.lower()
    return any(f"{lowered}{extension}" in listing for extension in extensions)


def promote_previews(*, host_dir: str, stem: str, media_dir: str, count: int) -> list[str]:
    """Copy the first still and the first video next to the ``.safetensors``.

    That is where :mod:`.thumbnails` looks for a tile preview, so this is what
    turns a freshly fetched catalogue into a visible thumbnail.  Nothing is
    overwritten: a preview the user already has wins.
    """
    first_image = first_video = ""
    for index in range(1, count + 1):
        path = existing_media(media_dir, index)
        if not path:
            continue
        extension = os.path.splitext(path)[1].lower()
        if not first_image and extension in cat.IMAGE_EXTENSIONS:
            first_image = path
        elif not first_video and extension in cat.VIDEO_EXTENSIONS:
            first_video = path
        if first_image and first_video:
            break

    promoted: list[str] = []
    for source, extensions in ((first_image, cat.IMAGE_EXTENSIONS), (first_video, cat.VIDEO_EXTENSIONS)):
        if not source or _has_preview(host_dir, stem, extensions):
            continue
        destination = os.path.join(host_dir, f"{stem}{os.path.splitext(source)[1].lower()}")
        try:
            shutil.copyfile(source, destination)
        except OSError:
            continue
        promoted.append(os.path.basename(destination))
    return promoted


# ----------------------------------------------------------------- fetch


def fetch_sidecar(
    lora_path: str,
    *,
    api_key: str = "",
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    lora_root: str = "",
) -> FetchReport:
    """Build or top up the ``<stem>/`` catalogue folder beside one LoRA."""
    if not lora_path or not os.path.isfile(lora_path):
        return FetchReport(message="That LoRA file is not on disk.")
    if lora_root and not is_within(lora_root, lora_path):
        return FetchReport(message="That LoRA sits outside the model's LoRA directory.")

    host_dir = os.path.dirname(lora_path)
    source_name = os.path.basename(lora_path)
    stem = os.path.splitext(source_name)[0]
    directory = os.path.join(host_dir, stem)
    media_dir = os.path.join(directory, cat.MEDIA_DIRNAME)

    try:
        file_hash = sha256_file(lora_path)
    except OSError as error:
        return FetchReport(message=f"Could not read the LoRA file: {error}")

    version, error = get_json(
        f"{BASE_URL}/model-versions/by-hash/{file_hash}",
        api_key=api_key,
        timeout=timeout,
        retries=retries,
    )
    if version is None:
        return FetchReport(
            message=f"Nothing fetched - {error}.",
            not_found=error == _NOT_FOUND,
        )

    model: dict[str, Any] = {}
    model_id = version.get("modelId")
    if isinstance(model_id, int) or (isinstance(model_id, str) and str(model_id).isdigit()):
        # Only the long description and the creator live on the model record, so
        # losing it costs detail, not the fetch.
        fetched, _ = get_json(
            f"{BASE_URL}/models/{model_id}", api_key=api_key, timeout=timeout, retries=retries
        )
        model = fetched or {}

    try:
        os.makedirs(media_dir, exist_ok=True)
    except OSError as error:
        return FetchReport(message=f"Could not create the catalogue folder: {error}")

    entries = media_entries(version)
    report = FetchReport(
        ok=True,
        media_total=len(entries),
        name=str(model.get("name") or _version_model_name(version) or stem),
    )

    for index, entry in enumerate(entries, start=1):
        url = str(entry.get("url", ""))
        # The prompt sidecar is rewritten even for media already on disk: an
        # older run may have saved a thinner record than Civitai serves now.
        try:
            _atomic_write_json(
                os.path.join(media_dir, f"{index:03d}.json"),
                {"index": index, "source_url": url, "civitai": entry},
            )
        except OSError:
            pass

        if existing_media(media_dir, index):
            report.skipped += 1
            continue

        path, download_error = download_media(
            url,
            media_dir=media_dir,
            index=index,
            declared_kind=str(entry.get("type", "") or "").lower(),
            api_key=api_key,
            timeout=timeout,
            retries=retries,
        )
        if path:
            report.downloaded += 1
        else:
            report.failed += 1
            print(f"[LoRA Browser] media {index} for {stem}: {download_error}")

    try:
        _write_documents(
            directory,
            stem,
            source_name=source_name,
            file_hash=file_hash,
            version=version,
            model=model,
            media_total=len(entries),
            media_local=report.downloaded + report.skipped,
        )
    except OSError as error:
        report.ok = False
        report.message = f"Media fetched, but the catalogue files could not be written: {error}"
        return report

    promoted = promote_previews(
        host_dir=host_dir, stem=stem, media_dir=media_dir, count=len(entries)
    )
    report.message = _describe(report, promoted)
    return report


# ------------------------------------------------------------- fetch all


def _has_media_record(media_dir: str, index: int) -> bool:
    """A prompt sidecar that actually carries its Civitai entry.

    Present-but-empty counts as absent: a folder built by something that wrote
    the files without the records is the case this exists to catch.
    """
    try:
        with open(os.path.join(media_dir, f"{index:03d}.json"), "r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return False
    return isinstance(record, dict) and isinstance(record.get("civitai"), dict)


def missing_parts(lora_path: str) -> list[str]:
    """Which pieces of a LoRA's catalogue are absent; empty when it is complete.

    Answered entirely from disk, because "Fetch all" asks it about every LoRA
    the model offers before deciding what to work on.

    The local Civitai record is the yardstick: it lists the media entries this
    LoRA is supposed to have, so a folder holding the images but no prompt
    sidecars -- the usual shape of a half-built or hand-made catalogue -- is
    reported incomplete instead of passing as finished.  It also means a LoRA
    Civitai has no media for is complete once fetched, rather than looking
    perpetually empty.

    A LoRA Civitai has never heard of has no folder at all, so it does keep
    answering "catalogue" and gets looked up again on the next run.  That costs
    one hash and one request; nothing on disk separates it from a LoRA whose
    catalogue has yet to be built, and a marker saying otherwise would go stale
    the day someone uploads it.
    """
    directory = cat.sidecar_dir(lora_path)
    if directory is None:
        return ["catalogue"]

    stem = os.path.splitext(os.path.basename(lora_path))[0]
    missing: list[str] = []
    if not os.path.isfile(os.path.join(directory, cat.SUMMARY_FILENAME)):
        missing.append("summary")

    version, _model, _combined = cat.load_records(directory, stem)
    if not version:
        # Nothing to check the rest against, and the record is itself what the
        # Inspect view reads.
        return missing + ["records"]

    if trained_words(version) and not os.path.isfile(os.path.join(directory, f"{stem}.txt")):
        missing.append("trigger words")

    media_dir = os.path.join(directory, cat.MEDIA_DIRNAME)
    expected = range(1, len(media_entries(version)) + 1)
    if any(not existing_media(media_dir, index) for index in expected):
        missing.append("media")
    if any(not _has_media_record(media_dir, index) for index in expected):
        missing.append("prompts")
    return missing


def needs_fetch(lora_path: str) -> bool:
    """True when anything in a LoRA's catalogue is missing."""
    return bool(missing_parts(lora_path))


class BulkFetch:
    """One "Fetch all" run, over many LoRAs, on a worker thread.

    It lives here rather than in ``plugin.py`` so the loop can be driven and
    inspected without Gradio: :meth:`run` is an ordinary blocking call, and
    :meth:`start` is the only part that involves a thread.  The plugin owns at
    most one of these at a time and does nothing but start it, poll
    :meth:`status` and, if asked, :meth:`cancel` it.
    """

    def __init__(
        self,
        paths,
        *,
        api_key: str = "",
        lora_root: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
    ):
        self.paths = [str(path) for path in paths]
        self._api_key = api_key
        self._lora_root = lora_root
        self._timeout = timeout
        self._retries = retries

        self._lock = threading.Lock()
        self._done = 0
        self._current = ""
        self._fetched = 0
        self._missing = 0
        self._failed = 0
        self._media = 0
        self._finished = not self.paths
        self._cancelled = False
        self._thread: threading.Thread | None = None

    def start(self) -> "BulkFetch":
        self._thread = threading.Thread(target=self.run, name="lora-browser-fetch-all", daemon=True)
        self._thread.start()
        return self

    def cancel(self) -> None:
        """Ask the loop to stop after the LoRA it is on. A fetch in flight is
        allowed to finish so it never leaves a half-written folder."""
        with self._lock:
            self._cancelled = True

    @property
    def running(self) -> bool:
        with self._lock:
            return not self._finished

    def run(self) -> None:
        try:
            for path in self.paths:
                with self._lock:
                    if self._cancelled:
                        break
                    self._current = os.path.basename(path)

                try:
                    report = fetch_sidecar(
                        path,
                        api_key=self._api_key,
                        timeout=self._timeout,
                        retries=self._retries,
                        lora_root=self._lora_root,
                    )
                except Exception as error:  # one bad LoRA must not end the run
                    print(f"[LoRA Browser] fetch all: {os.path.basename(path)}: {error}")
                    report = FetchReport(message=str(error))

                with self._lock:
                    self._done += 1
                    if report.ok:
                        self._fetched += 1
                        self._media += report.downloaded
                    elif report.not_found:
                        self._missing += 1
                    else:
                        self._failed += 1
        finally:
            with self._lock:
                self._finished = True
                self._current = ""

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total": len(self.paths),
                "done": self._done,
                "current": self._current,
                "fetched": self._fetched,
                "missing": self._missing,
                "failed": self._failed,
                "media": self._media,
                "finished": self._finished,
                "cancelled": self._cancelled,
            }

    def summary(self) -> str:
        state = self.status()
        parts = [f"Fetched {state['fetched']} of {state['total']} LoRA(s)"]
        if state["media"]:
            parts.append(f"{state['media']} media file(s)")
        if state["missing"]:
            parts.append(f"{state['missing']} not on Civitai")
        if state["failed"]:
            parts.append(f"{state['failed']} failed")
        if state["cancelled"]:
            parts.append("stopped early")
        return " - ".join(parts) + "."


def _describe(report: FetchReport, promoted: list[str]) -> str:
    parts = [f"Fetched '{report.name}'"]
    if report.downloaded:
        parts.append(f"{report.downloaded} new media")
    if report.skipped:
        parts.append(f"{report.skipped} already local")
    if report.failed:
        parts.append(f"{report.failed} failed")
    if not report.media_total:
        parts.append("no media on Civitai")
    if promoted:
        parts.append("preview added")
    return " - ".join(parts) + "."
