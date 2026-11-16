"""Controlling the PC: media and volume keys, lock screen, opening websites,
the clipboard, and finding/reading files.

Security rules, enforced here in code:
- pc_control can only do the fixed actions in PC_ACTIONS. There is no shutdown,
  restart, delete or "run this command" action, on purpose.
- Websites open in the normal browser, http(s) only, never file:, javascript: or
  links with a password in them, and never this PC or the home network.
- Files can only be found and read inside config.ALLOWED_FOLDERS. The real path is
  resolved first, so "..", shortcuts and symlinks can't escape those folders.
- Files that usually hold secrets (.env, private keys, password databases, browser
  profiles) are refused even inside an allowed folder.
- Text read from files and the clipboard is redacted for secrets and handed to the
  model as untrusted data.
"""
import ctypes
import re
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

import config
import security

# Windows virtual-key codes for the media keys on a keyboard.
_KEYS = {"volume_up": 0xAF, "volume_down": 0xAE, "mute": 0xAD,
         "play_pause": 0xB3, "next_track": 0xB0, "previous_track": 0xB1}
PC_ACTIONS = ("volume_up", "volume_down", "mute", "play_pause", "next_track",
              "previous_track", "lock_screen", "open_website", "open_settings")
# Windows Settings pages he may open (a fixed list: ms-settings: pages only change what
# sir then clicks, and nothing outside this list is reachable).
SETTINGS_PAGES = {
    "display": "display", "screen": "display", "brightness": "display", "resolution": "display",
    "sound": "sound", "audio": "sound", "bluetooth": "bluetooth", "devices": "bluetooth",
    "wifi": "network-wifi", "wi-fi": "network-wifi", "network": "network-status", "internet": "network-status",
    "notifications": "notifications", "power": "powersleep", "battery": "batterysaver", "sleep": "powersleep",
    "apps": "appsfeatures", "default apps": "defaultapps", "storage": "storagesense",
    "update": "windowsupdate", "windows update": "windowsupdate", "privacy": "privacy",
    "personalisation": "personalization", "personalization": "personalization", "background": "personalization-background",
    "themes": "themes", "colours": "colors", "colors": "colors", "mouse": "mousetouchpad", "keyboard": "typing",
    "time": "dateandtime", "date": "dateandtime", "language": "regionlanguage", "about": "about", "home": "",
}


class NotAllowed(Exception):
    pass


def _windows_only():
    if sys.platform != "win32":
        raise NotAllowed("That only works on Windows.")


def _press(vk: int, times: int = 1) -> None:
    user32 = ctypes.windll.user32
    for _ in range(times):
        user32.keybd_event(vk, 0, 0, 0)       # key down
        user32.keybd_event(vk, 0, 2, 0)       # key up (KEYEVENTF_KEYUP)
        time.sleep(0.02)


def settings_page(page: str) -> str:
    """'display settings' -> 'display'. Refuses anything not in SETTINGS_PAGES."""
    key = re.sub(r"\s+", " ", page.lower().replace("ms-settings:", "").replace("settings", "")).strip(" :")
    if key not in SETTINGS_PAGES:
        raise NotAllowed(f"I can open these settings pages: {', '.join(sorted(set(SETTINGS_PAGES) - {'home'}))}.")
    return SETTINGS_PAGES[key]


def open_settings(page: str) -> str:
    target = settings_page(page)
    _windows_only()
    import os

    os.startfile("ms-settings:" + target)
    return f"Opened the {page.strip() or 'Settings'} settings."


def check_url(url: str) -> str:
    """The address as it would be opened, or NotAllowed. Run before sir is asked to approve,
    so he is never asked to approve something that would be refused anyway."""
    url = url.strip()
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:(?!\d)", url) and "://" not in url:
        raise NotAllowed("Only normal http or https web addresses can be opened.")
    if "://" not in url:
        url = "https://" + url
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise NotAllowed("Only normal http or https web addresses can be opened.")
    if parsed.username or parsed.password:
        raise NotAllowed("Links with a username or password in them are not opened.")
    host = parsed.hostname.lower()
    if host == "localhost" or host.endswith((".local", ".lan", ".internal")) or _is_internal_ip(host):
        raise NotAllowed("I don't open addresses on this PC or the home network.")
    return url


def open_website(url: str) -> str:
    url = check_url(url)
    parsed = urlparse(url)
    webbrowser.open(url, new=2)
    return f"Opened {parsed.hostname} in the browser."


def _is_internal_ip(host: str) -> bool:
    import ipaddress

    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved


def normalise_action(action: str) -> str:
    action = action.strip().lower().replace(" ", "_").replace("-", "_")
    return {"pause": "play_pause", "play": "play_pause", "next": "next_track", "skip": "next_track",
            "previous": "previous_track", "unmute": "mute", "lock": "lock_screen", "louder": "volume_up",
            "quieter": "volume_down", "open": "open_website", "website": "open_website",
            "settings": "open_settings"}.get(action, action)


def precheck(action: str, url: str = "") -> None:
    """Refuse now what control() would refuse, before anyone is asked to approve it."""
    action = normalise_action(action)
    if action not in PC_ACTIONS:
        raise NotAllowed(f"Unknown action '{action}'. Allowed: {', '.join(PC_ACTIONS)}.")
    if action == "open_website":
        if not url:
            raise NotAllowed("Which website? Give me its address.")
        if url.strip().lower().startswith("ms-settings:"):
            raise NotAllowed("That is a Windows Settings page: use open_settings with the page name.")
        check_url(url)


def control(action: str, url: str = "", steps: int = 5) -> str:
    action = action.strip().lower().replace(" ", "_").replace("-", "_")
    action = {"pause": "play_pause", "play": "play_pause", "next": "next_track", "skip": "next_track",
              "previous": "previous_track", "unmute": "mute", "lock": "lock_screen", "louder": "volume_up",
              "quieter": "volume_down", "open": "open_website", "website": "open_website",
              "settings": "open_settings"}.get(action, action)
    if action not in PC_ACTIONS:
        raise NotAllowed(f"Unknown action '{action}'. Allowed: {', '.join(PC_ACTIONS)}.")
    if action == "open_settings":
        return open_settings(url)
    if action == "open_website":
        if not url:
            raise NotAllowed("Which website? Give me its address.")
        return open_website(url)
    _windows_only()
    if action == "lock_screen":
        ctypes.windll.user32.LockWorkStation()
        return "Screen locked."
    steps = max(1, min(int(steps or 5), 25))
    _press(_KEYS[action], steps if action.startswith("volume") else 1)
    return {"volume_up": f"Volume up {steps * 2}%.", "volume_down": f"Volume down {steps * 2}%.",
            "mute": "Toggled mute.", "play_pause": "Toggled play/pause.",
            "next_track": "Skipped to the next track.", "previous_track": "Back to the previous track."}[action]


# ---------------------------------------------------------------- clipboard

MAX_CLIPBOARD = 10_000


def _powershell(command: str, stdin: str | None = None) -> str:
    res = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                         input=stdin, capture_output=True, text=True, timeout=10, encoding="utf-8",
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if res.returncode != 0:
        raise NotAllowed("The clipboard couldn't be reached.")
    return res.stdout


def read_clipboard() -> str:
    _windows_only()
    text = _powershell("Get-Clipboard -Raw").rstrip("\r\n")
    if not text.strip():
        return "The clipboard is empty (or holds an image, which I can't read yet)."
    if len(text) > MAX_CLIPBOARD:
        text = text[:MAX_CLIPBOARD] + " [...]"
    # Password managers put passwords on the clipboard; never hand those to the model.
    found = [f for f in security.scan(text, kinds=("secret", "pii")) if f.name != "email"]
    if found:
        security.log("clipboard", found)
        text = security.redact(text)
    return security.as_untrusted("the clipboard", text)


def write_clipboard(text: str) -> str:
    _windows_only()
    if len(text) > MAX_CLIPBOARD:
        raise NotAllowed(f"That's too long for the clipboard (limit {MAX_CLIPBOARD} characters).")
    # Text goes in through stdin, never pasted into the PowerShell command itself.
    _powershell("$input | Out-String | ForEach-Object { $_.TrimEnd() } | Set-Clipboard", stdin=text)
    return "Copied to the clipboard."


# ---------------------------------------------------------------- files

READABLE = {".txt", ".md", ".csv", ".log", ".json", ".py", ".html", ".xml", ".yaml", ".yml",
            ".ini", ".cfg", ".pdf", ".docx"}
SECRET_NAMES = {".env", "id_rsa", "id_ed25519", "id_ecdsa", "credentials", "credentials.json",
                "secrets.json", "known_hosts", "authorized_keys", ".npmrc", ".pypirc", ".netrc"}
SECRET_SUFFIXES = {".pem", ".key", ".pfx", ".p12", ".kdbx", ".kdb", ".ppk", ".gpg", ".asc", ".ovpn"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "AppData", "$RECYCLE.BIN"}
MAX_FILE_BYTES = 10 * 2**20
MAX_READ_CHARS = 6_000


def allowed_roots() -> list[Path]:
    roots = []
    for folder in list(config.ALLOWED_FOLDERS) + [config.WORKSPACE]:
        p = Path(folder).expanduser()
        if p.is_dir():
            roots.append(p.resolve())
    return roots


def _inside_allowed(path: Path) -> Path:
    real = path.expanduser().resolve()
    for root in allowed_roots():
        if real == root or root in real.parents:
            return real
    names = ", ".join(str(r) for r in allowed_roots()) or "none set"
    raise NotAllowed(f"I can only look inside these folders: {names}.")


def _looks_secret(path: Path) -> bool:
    name = path.name.lower()
    if name in SECRET_NAMES or name.startswith(".env") or path.suffix.lower() in SECRET_SUFFIXES:
        return True
    return any(part.lower() in ("password", "passwords", ".ssh", ".gnupg") for part in path.parts) or \
        "password" in name


def find_files(name: str, max_results: int = 15) -> str:
    needle = name.strip().lower().strip("*")
    if len(needle) < 2:
        raise NotAllowed("Give me at least two letters of the file name.")
    roots = allowed_roots()
    if not roots:
        raise NotAllowed("No folders are allowed yet. Add some to ALLOWED_FOLDERS in config.py.")
    hits, scanned, deadline = [], 0, time.monotonic() + 8
    for root in roots:
        stack = [root]
        while stack and len(hits) < max_results and time.monotonic() < deadline:
            folder = stack.pop()
            try:
                entries = list(folder.iterdir())
            except (PermissionError, OSError):
                continue
            for entry in entries:
                scanned += 1
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    if entry.name not in SKIP_DIRS and not entry.name.startswith("."):
                        stack.append(entry)
                elif needle in entry.name.lower() and not _looks_secret(entry):
                    modified = time.strftime("%d %b %Y", time.localtime(entry.stat().st_mtime))
                    hits.append(f"{entry}  ({entry.stat().st_size // 1024} KB, {modified})")
                    if len(hits) >= max_results:
                        break
    if not hits:
        return f"No files with '{name}' in the name in: " + ", ".join(str(r) for r in roots) + "."
    return "\n".join(hits)


def _pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages[:30])


def _docx_text(path: Path) -> str:
    import docx

    return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)


def read_file(path: str) -> str:
    real = _inside_allowed(Path(path.strip().strip('"')))
    if not real.is_file():
        raise NotAllowed("That file doesn't exist. Use find_files to get the exact path.")
    if _looks_secret(real):
        security.log("read_file", [security.Finding("secret", "secret_file", "high")])
        raise NotAllowed("That looks like a file that holds passwords or keys, so I won't open it.")
    if real.suffix.lower() not in READABLE:
        raise NotAllowed(f"I can't read {real.suffix or 'that kind of'} files. I can read: "
                         + ", ".join(sorted(READABLE)) + ".")
    if real.stat().st_size > MAX_FILE_BYTES:
        raise NotAllowed("That file is over 10 MB, too big to read.")
    if real.suffix.lower() == ".pdf":
        text = _pdf_text(real)
    elif real.suffix.lower() == ".docx":
        text = _docx_text(real)
    else:
        text = real.read_text(encoding="utf-8", errors="replace")
    text = text.strip() or "(the file has no readable text)"
    total = len(text)
    if total > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS] + f"\n[... showing the first {MAX_READ_CHARS} of {total} characters]"
    found = [f for f in security.scan(text, kinds=("secret", "pii")) if f.name != "email"]
    if found:
        security.log(f"read_file:{real.name}", found)
        text = security.redact(text)
    return security.as_untrusted(f"the file {real.name}", text)
