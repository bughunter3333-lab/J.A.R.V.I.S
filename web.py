"""Internet access for Jarvis: search, reading pages, and weather.

Security rules, enforced here in code:
- Nothing goes out if config.ALLOW_INTERNET is False.
- Search queries are scanned before they leave the PC: a query containing a
  password, key or ID number is refused instead of being sent to a search engine.
- read_webpage only fetches public http(s) pages. Every address a URL resolves to
  (including after redirects) is checked, and loopback, private, link-local and
  other internal ranges are refused, so a page can't trick Jarvis into probing the
  home network or his own API (server-side request forgery).
- Page size and time are capped, and only text/HTML is read.
- Everything that comes back is wrapped as untrusted data and scanned for prompt
  injection before the model sees it (security.as_untrusted).
"""
import ipaddress
import socket
import time
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

import config
import security


class Blocked(Exception):
    """A request that breaks one of the rules above."""


TIMEOUT = 10.0
MAX_BYTES = 2_000_000
MAX_REDIRECTS = 3
MAX_PAGE_CHARS = 4_000
USER_AGENT = "JARVIS-X/1.0 (personal assistant; +https://github.com/bughunter3333-lab/J.A.R.V.I.S)"


def _require_internet() -> None:
    if not config.ALLOW_INTERNET:
        raise Blocked("Internet tools are switched off (config.ALLOW_INTERNET = False).")


def _check_outbound(text: str, what: str) -> None:
    sensitive = [f for f in security.scan(text, kinds=("secret", "pii")) if f.name != "email"]
    if sensitive:
        security.log(f"outbound:{what}", sensitive)
        raise Blocked(f"That {what} contains a password, key or ID number, so it was not sent.")


# ---------------------------------------------------------------- search

def search(query: str, news: bool = False, max_results: int = 5) -> str:
    _require_internet()
    query = query.strip()[:200]
    if not query:
        raise Blocked("Empty search.")
    _check_outbound(query, "search")
    from ddgs import DDGS

    with DDGS(timeout=TIMEOUT) as ddgs:
        if news:
            rows = ddgs.news(query, max_results=max_results, region=config.SEARCH_REGION)
            lines = [f"- {r.get('title', '')} ({r.get('source', '')}, {r.get('date', '')[:10]}): "
                     f"{r.get('body', '')[:250]} [{r.get('url', '')}]" for r in rows]
        else:
            rows = ddgs.text(query, max_results=max_results, region=config.SEARCH_REGION)
            lines = [f"- {r.get('title', '')}: {r.get('body', '')[:250]} [{r.get('href', '')}]"
                     for r in rows]
    if not lines:
        return "No results found."
    return security.as_untrusted("web search results", "\n".join(lines))


# ---------------------------------------------------------------- pages

def _is_public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    if getattr(addr, "ipv4_mapped", None):
        addr = addr.ipv4_mapped
    return not (addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_multicast
                or addr.is_reserved or addr.is_unspecified)


def check_url(url: str) -> str:
    """Return the URL if it is a public http(s) address, otherwise raise Blocked."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise Blocked("Only http and https links can be opened.")
    if not parsed.hostname:
        raise Blocked("That isn't a complete web address.")
    if parsed.username or parsed.password:
        raise Blocked("Links with a username or password in them are not opened.")
    if parsed.port not in (None, 80, 443):
        raise Blocked("Only the normal web ports (80 and 443) are allowed.")
    host = parsed.hostname
    if host.lower() in ("localhost",) or host.lower().endswith((".local", ".internal", ".lan")):
        raise Blocked("Addresses on this PC or the home network are not allowed.")
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror:
        raise Blocked(f"Couldn't find the website '{host}'.")
    for info in infos:
        if not _is_public(info[4][0]):
            raise Blocked("That address points to this PC or a private network, so it was refused.")
    return url.strip()


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form", "iframe"}

    def __init__(self):
        super().__init__()
        self.parts, self.title, self._skip, self._in_title = [], "", 0, False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False
        if tag in ("p", "div", "li", "h1", "h2", "h3", "br", "tr"):
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data.strip()
        elif not self._skip and data.strip():
            self.parts.append(data.strip() + " ")


def html_to_text(html: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(html)
    text = "".join(parser.parts)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return parser.title, "\n".join(line for line in lines if line)


def fetch(url: str) -> str:
    _require_internet()
    url = check_url(url)
    _check_outbound(url, "link")
    with httpx.Client(timeout=TIMEOUT, follow_redirects=False,
                      headers={"User-Agent": USER_AGENT}) as client:
        for _ in range(MAX_REDIRECTS + 1):
            with client.stream("GET", url) as res:
                if res.is_redirect:
                    url = check_url(urljoin(url, res.headers.get("location", "")))
                    continue
                if res.status_code >= 400:
                    raise Blocked(f"The site answered with error {res.status_code}.")
                kind = res.headers.get("content-type", "")
                if not any(t in kind for t in ("text/html", "text/plain", "application/xhtml")):
                    raise Blocked(f"That link isn't a web page ({kind.split(';')[0] or 'unknown type'}).")
                body = b""
                for chunk in res.iter_bytes():
                    body += chunk
                    if len(body) > MAX_BYTES:
                        break
                html = body.decode(res.encoding or "utf-8", errors="replace")
                break
        else:
            raise Blocked("Too many redirects.")
    title, text = html_to_text(html) if "html" in kind else ("", html)
    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + " [...]"
    return security.as_untrusted(f"web page {url}", f"Title: {title}\n{text}")


# ---------------------------------------------------------------- weather

WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain",
    65: "heavy rain", 71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
    81: "rain showers", 82: "violent rain showers", 95: "thunderstorm", 96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}


def weather(place: str = "") -> str:
    _require_internet()
    place = (place or config.HOME_CITY).strip()[:80]
    with httpx.Client(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        geo = client.get("https://geocoding-api.open-meteo.com/v1/search",
                         params={"name": place, "count": 1, "language": "en"}).json()
        if not geo.get("results"):
            raise Blocked(f"I couldn't find a place called '{place}'.")
        loc = geo["results"][0]
        data = client.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": loc["latitude"], "longitude": loc["longitude"], "timezone": "auto",
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
            "forecast_days": 2,
        }).json()
    cur, day = data["current"], data["daily"]
    name = ", ".join(x for x in (loc.get("name"), loc.get("admin1"), loc.get("country")) if x)
    return (
        f"{name}: now {cur['temperature_2m']:.0f}°C (feels like {cur['apparent_temperature']:.0f}°C), "
        f"{WEATHER_CODES.get(cur['weather_code'], 'unknown conditions')}, humidity {cur['relative_humidity_2m']}%, "
        f"wind {cur['wind_speed_10m']:.0f} km/h. Today {day['temperature_2m_min'][0]:.0f} to "
        f"{day['temperature_2m_max'][0]:.0f}°C, {day['precipitation_probability_max'][0]}% chance of rain. "
        f"Tomorrow {day['temperature_2m_min'][1]:.0f} to {day['temperature_2m_max'][1]:.0f}°C, "
        f"{WEATHER_CODES.get(day['weather_code'][1], 'unknown conditions')}, "
        f"{day['precipitation_probability_max'][1]}% chance of rain. (Source: Open-Meteo)"
    )


# ---------------------------------------------------------------- knowledge

def _get_json(client: httpx.Client, url: str, **params) -> dict | list:
    res = client.get(url, params=params or None)
    if res.status_code == 429:
        raise Blocked("That service is rate-limiting us. Try again in a minute.")
    if res.status_code == 404:
        return {}
    try:
        return res.json()
    except ValueError:
        raise Blocked("That service sent back something I couldn't read.")


def wikipedia(topic: str) -> str:
    _require_internet()
    topic = topic.strip()[:120]
    _check_outbound(topic, "lookup")
    with httpx.Client(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        found = _get_json(client, "https://en.wikipedia.org/w/api.php", action="opensearch",
                          search=topic, limit=1, namespace=0, format="json")
        if not found or len(found) < 2 or not found[1]:
            raise Blocked(f"Wikipedia has no article matching '{topic}'.")
        title = found[1][0]
        page = _get_json(client, "https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_"))
    summary = (page or {}).get("extract") or ""
    if not summary:
        raise Blocked(f"Couldn't read the Wikipedia article '{title}'.")
    link = (page.get("content_urls") or {}).get("desktop", {}).get("page", "")
    return security.as_untrusted("Wikipedia", f"{title}: {summary[:1500]} [{link}]")


def define(word: str) -> str:
    _require_internet()
    word = word.strip().lower()[:50]
    if not word.replace("-", "").replace(" ", "").isalpha():
        raise Blocked("Give me a single word to define.")
    with httpx.Client(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        data = _get_json(client, "https://api.dictionaryapi.dev/api/v2/entries/en/" + word)
    if not isinstance(data, list) or not data:
        raise Blocked(f"I couldn't find '{word}' in the dictionary.")
    lines = []
    for meaning in data[0].get("meanings", [])[:3]:
        d = meaning.get("definitions", [{}])[0]
        line = f"({meaning.get('partOfSpeech', '')}) {d.get('definition', '')}"
        if d.get("example"):
            line += f' e.g. "{d["example"]}"'
        lines.append(line)
    phonetic = data[0].get("phonetic", "")
    return f"{word} {phonetic}: " + " ".join(lines)


_rates_cache: dict[str, tuple[float, dict]] = {}


def exchange_rate(base: str, target: str) -> tuple[float, str]:
    """Rate from base to target currency (open.er-api.com, free, no key, updated daily)."""
    _require_internet()
    base, target = base.upper(), target.upper()
    cached = _rates_cache.get(base)
    if not cached or time.monotonic() - cached[0] > 3600:
        with httpx.Client(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
            data = _get_json(client, f"https://open.er-api.com/v6/latest/{base}")
        if not isinstance(data, dict) or data.get("result") != "success":
            raise Blocked(f"I couldn't get exchange rates for {base}.")
        cached = (time.monotonic(), data)
        _rates_cache[base] = cached
    data = cached[1]
    if target not in data["rates"]:
        raise Blocked(f"I don't have a rate for {target}.")
    return data["rates"][target], data.get("time_last_update_utc", "")[:16]
