#!/usr/bin/env python3
"""Build data/deadlines.json for tracker.html.

Steps:
  1. Read conference deadlines from the AI deadlines list that Hugging Face keeps
     on Papers with Code (https://paperswithcode.co/ai-deadlines), through its
     public API. Full conference names come from the huggingface/ai-deadlines
     GitHub repository, because the API does not include them.
  2. Check those conferences on OpenReview, and add the extra OpenReview venues
     listed in data/openreview_venues.yml (see scripts/openreview.py).
  3. Add the African venues kept by hand in data/african_venues.yml.
  4. Turn every deadline into a UTC time and write data/deadlines.json.

Usage:
    python scripts/update_deadlines.py                           # Papers with Code API + OpenReview
    python scripts/update_deadlines.py --source huggingface      # GitHub repo (updated less often)
    python scripts/update_deadlines.py --source paperswithcode   # old list (last updated 2024)
    python scripts/update_deadlines.py --no-openreview           # skip OpenReview

Needs Python 3.9+ and PyYAML (pip install pyyaml).
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import openreview as orv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AFRICAN = ROOT / "data" / "african_venues.yml"
DEFAULT_OUT = ROOT / "data" / "deadlines.json"
DEFAULT_OPENREVIEW = ROOT / "data" / "openreview_venues.yml"
DEFAULT_NAMES = ROOT / "data" / "full_names.yml"

SOURCES = {
    "api": {
        "label": "Papers with Code AI deadlines",
        "page": "https://paperswithcode.co/ai-deadlines",
        "url": "https://paperswithcode.co/api/v1/conferences/deadlines",
    },
    "paperswithcode": {
        "label": "paperswithcode/ai-deadlines",
        "page": "https://github.com/paperswithcode/ai-deadlines",
        "url": "https://raw.githubusercontent.com/paperswithcode/ai-deadlines/gh-pages/_data/conferences.yml",
    },
    "huggingface": {
        "label": "huggingface/ai-deadlines",
        "page": "https://github.com/huggingface/ai-deadlines",
        "url": "https://codeload.github.com/huggingface/ai-deadlines/tar.gz/main",
    },
}

# Field codes used by paperswithcode/ai-deadlines, written out in full.
FIELD_CODES = {
    "ML": "Machine Learning",
    "CV": "Computer Vision",
    "CG": "Computer Graphics",
    "NLP": "Natural Language Processing",
    "RO": "Robotics",
    "SP": "Speech and Signal Processing",
    "DM": "Data Mining",
    "AP": "Automated Planning",
    "KR": "Knowledge Representation",
    "HCI": "Human-Computer Interaction",
}

# Short time zone names that zoneinfo does not know.
TZ_ALIASES = {
    "AOE": "UTC-12",  # Anywhere on Earth
    "PST": "America/Los_Angeles",
    "PDT": "America/Los_Angeles",
    "EST": "America/New_York",
    "EDT": "America/New_York",
    "CET": "Europe/Paris",
    "CEST": "Europe/Paris",
}

# Words to keep in capitals when turning a tag like "large-language-models" into a label.
UPPER_WORDS = {"ai", "nlp", "hci"}
SMALL_WORDS = {"and", "of", "for", "on", "in", "the"}

# Conferences held every two years. The list may hold only one recent edition,
# so the gap between editions cannot always be worked out from the data.
KNOWN_CYCLE = {"iccv": 2, "eccv": 2, "accv": 2, "lrec": 2}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "conference-deadline-tracker"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def parse_tz(name) -> timezone | ZoneInfo:
    """Turn 'AoE', 'UTC-12', 'GMT+02', 'UTC' or 'Europe/London' into a tzinfo."""
    text = str(name or "AoE").strip()
    text = TZ_ALIASES.get(text.upper(), text)
    m = re.fullmatch(r"(?:UTC|GMT)\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?", text, re.IGNORECASE)
    if m:
        sign = -1 if m.group(1) == "-" else 1
        delta = timedelta(hours=int(m.group(2)), minutes=int(m.group(3) or 0))
        return timezone(sign * delta)
    if text.upper() in ("UTC", "GMT", "Z"):
        return timezone.utc
    try:
        return ZoneInfo(text)
    except (ZoneInfoNotFoundError, ValueError):
        print(f"  warning: unknown time zone {name!r}, using AoE", file=sys.stderr)
        return timezone(timedelta(hours=-12))


def to_utc(value, tz_name) -> str | None:
    """Return the deadline as an ISO 8601 UTC string, or None when it is not set (e.g. 'TBA')."""
    if value is None:
        return None
    if isinstance(value, datetime):
        local = value
    elif isinstance(value, date):
        local = datetime(value.year, value.month, value.day, 23, 59, 59)
    else:
        text = str(value).strip()
        local = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                local = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        if local is None:
            return None
        if len(text) == 10:  # date only: the deadline is the end of that day
            local = local.replace(hour=23, minute=59, second=59)
    if local.tzinfo is None:
        local = local.replace(tzinfo=parse_tz(tz_name))
    return local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tz_label(tz_name) -> str:
    text = str(tz_name or "AoE").strip()
    return "AoE" if text.upper() in ("AOE", "UTC-12", "GMT-12") else text


def as_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value).strip(), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def iso_day(d) -> str | None:
    return d.isoformat() if d else None


def event_dates(conf: dict) -> str:
    """Make a short, clear date range like 'Jun 28 - Jul 2, 2027'."""
    start, end = as_date(conf.get("start")), as_date(conf.get("end"))
    if start and end:
        if start == end:
            return f"{start:%b} {start.day}, {start.year}"
        if start.year != end.year:
            return f"{start:%b} {start.day}, {start.year} - {end:%b} {end.day}, {end.year}"
        if start.month == end.month:
            return f"{start:%b} {start.day} - {end.day}, {end.year}"
        return f"{start:%b} {start.day} - {end:%b} {end.day}, {end.year}"
    return str(conf.get("date") or "To be announced").strip()


def _plain(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


KNOWN_FIELDS = {_plain(v): v for v in FIELD_CODES.values()}


def label_from_tag(tag: str) -> str:
    """'human-computer-interaction' -> 'Human-Computer Interaction', 'web mining' -> 'Web Mining'."""
    text = str(tag).strip()
    if text.upper() in FIELD_CODES:
        return FIELD_CODES[text.upper()]
    if _plain(text) in KNOWN_FIELDS:
        return KNOWN_FIELDS[_plain(text)]
    words = [w for w in re.split(r"[-_\s]+", text) if w]
    out = []
    for i, w in enumerate(words):
        if w.lower() in UPPER_WORDS:
            out.append(w.upper())
        elif i and w.lower() in SMALL_WORDS:
            out.append(w.lower())
        else:
            out.append(w[0].upper() + w[1:].lower())
    return " ".join(out)


def fields_from(conf: dict) -> list[str]:
    raw = conf.get("sub") or conf.get("tags") or conf.get("field") or []
    if isinstance(raw, str):
        raw = [raw]
    fields = []
    for item in raw:
        name = label_from_tag(item)
        if name and name not in fields:
            fields.append(name)
    return fields or ["Other"]


def location(conf: dict) -> str:
    if conf.get("place"):
        return str(conf["place"]).strip()
    parts = [str(conf.get(k)).strip() for k in ("city", "country") if conf.get(k)]
    return ", ".join(parts) or "To be announced"


def make_record(conf: dict, *, paper, abstract, source: str, africa: bool, sub_type: str) -> dict:
    """paper and abstract are (date, timezone) pairs or None."""
    deadline_utc = to_utc(*paper) if paper else None
    record = {
        "id": str(conf.get("id") or f"{conf.get('title')}{conf.get('year')}").lower(),
        "name": f"{conf.get('title')} {conf.get('year') or ''}".strip(),
        "year": int(conf.get("year") or 0),
        "full_name": str(conf.get("full_name") or "").strip(),
        "submission_type": sub_type,
        "deadline_utc": deadline_utc,
        "deadline_timezone": tz_label(paper[1]) if paper and deadline_utc else None,
        "abstract_deadline_utc": to_utc(*abstract) if abstract else None,
        "deadline_source": source,
        "openreview_deadline_utc": None,
        "openreview_link": None,
        "upcoming": False,
        "expected_deadline_utc": None,
        "last_deadline_utc": None,
        "last_edition": None,
        "fields": fields_from(conf),
        "event_dates": event_dates(conf),
        "event_end": iso_day(as_date(conf.get("end")) or as_date(conf.get("start"))),
        "location": location(conf),
        "link": str(conf.get("link") or "").strip(),
        "africa": africa,
        "source": source,
    }
    return record


# ---------- ai-deadlines readers ----------

def read_paperswithcode(raw: bytes) -> list[dict]:
    """One flat YAML list with 'deadline', 'abstract_deadline' and 'timezone'."""
    out = []
    for conf in yaml.safe_load(raw) or []:
        tz = conf.get("timezone")
        abstract = (conf["abstract_deadline"], tz) if conf.get("abstract_deadline") else None
        out.append(make_record(conf, paper=(conf.get("deadline"), tz), abstract=abstract,
                               source="ai-deadlines", africa=False, sub_type="Full paper"))
    return out


def read_huggingface(raw: bytes) -> list[dict]:
    """A tarball with one YAML file per conference series in src/data/conferences/."""
    out = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not (member.isfile() and "/src/data/conferences/" in member.name and member.name.endswith((".yml", ".yaml"))):
                continue
            for conf in yaml.safe_load(tar.extractfile(member)) or []:
                top_tz = conf.get("timezone")
                paper = abstract = None
                for d in conf.get("deadlines") or []:
                    kind = str(d.get("type", "")).lower()
                    pair = (d.get("date"), d.get("timezone") or top_tz)
                    if kind in ("paper", "submission") and paper is None:
                        paper = pair
                    elif kind == "abstract" and abstract is None:
                        abstract = pair
                if paper is None and conf.get("deadline"):  # older entries use flat fields
                    paper = (conf["deadline"], top_tz)
                if abstract is None and conf.get("abstract_deadline"):
                    abstract = (conf["abstract_deadline"], top_tz)
                if paper is None and abstract is not None:
                    paper, abstract, sub_type = abstract, None, "Abstract"
                else:
                    sub_type = "Full paper"
                out.append(make_record(conf, paper=paper, abstract=abstract,
                                       source="ai-deadlines", africa=False, sub_type=sub_type))
    return out


def read_api(raw: bytes) -> list[dict]:
    """JSON from paperswithcode.co/api/v1/conferences/deadlines.

    Deadlines are already in UTC ("deadline_at"); "timezone" is the organisers' own zone."""
    data = json.loads(raw)
    items = data.get("results") or []
    if data.get("count") and data["count"] > len(items):
        print(f"  warning: the API says {data['count']} conferences but sent {len(items)}", file=sys.stderr)
    names = full_names_from_repo()
    out = []
    for item in items:
        title = str(item.get("short_name") or item.get("name") or "").strip()
        year = item.get("year")
        paper = abstract = None
        for d in item.get("deadlines") or []:
            kind = str(d.get("type", "")).lower()
            when = d.get("deadline_at")
            if not when:
                continue
            pair = (datetime.fromisoformat(str(when).replace("Z", "+00:00")), d.get("timezone") or "UTC")
            if kind in ("paper", "submission") and paper is None:
                paper = pair
            elif kind == "abstract" and abstract is None:
                abstract = pair
        sub_type = "Full paper"
        if paper is None and abstract is not None:
            paper, abstract, sub_type = abstract, None, "Abstract"
        conf = {
            "id": item.get("slug"),
            "title": title,
            "year": year,
            "full_name": names.get((title.lower(), year)) or names.get(title.lower()) or "",
            "link": item.get("url"),
            "start": item.get("start_date"),
            "end": item.get("end_date"),
            "place": item.get("location"),
            "tags": item.get("tags") or [],
        }
        out.append(make_record(conf, paper=paper, abstract=abstract, source="Papers with Code", africa=False, sub_type=sub_type))
    return out


def full_names_from_repo() -> dict:
    """Full conference names from the huggingface/ai-deadlines repository (best effort)."""
    try:
        raw = fetch(SOURCES["huggingface"]["url"])
    except Exception as err:  # names are nice to have, not needed
        print(f"  warning: could not read full names ({err})", file=sys.stderr)
        return {}
    names = {}
    for rec in read_huggingface(raw):
        if rec["full_name"]:
            title = rec["name"].removesuffix(str(rec["year"])).strip().lower()
            names[(title, rec["year"])] = rec["full_name"]
            names.setdefault(title, rec["full_name"])
    return names


def latest_editions(records: list[dict]) -> list[dict]:
    """Keep only the newest year of each conference, so old editions do not fill the page.

    Also note how often it happens ("cycle": 1 = every year, 2 = every two years),
    from the gap between its two newest editions."""
    years: dict[str, set[int]] = {}
    best: dict[str, dict] = {}
    for rec in records:
        key = rec["name"].removesuffix(str(rec["year"])).strip().lower()
        years.setdefault(key, set()).add(rec["year"])
        if key not in best or rec["year"] > best[key]["year"]:
            best[key] = rec
    for key, rec in best.items():
        ys = sorted(years[key], reverse=True)
        gap = ys[0] - ys[1] if len(ys) > 1 else 1
        rec["cycle"] = KNOWN_CYCLE.get(orv.plain(key), gap if gap in (1, 2) else 1)
    return list(best.values())


def read_african(path: Path) -> list[dict]:
    out = []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    for conf in data:
        out.append(make_record(
            conf,
            paper=(conf.get("deadline"), conf.get("timezone")),
            abstract=(conf["abstract_deadline"], conf.get("timezone")) if conf.get("abstract_deadline") else None,
            source="African venues list",
            africa=True,
            sub_type=str(conf.get("submission_type") or "Full paper"),
        ))
    return out


# ---------- OpenReview ----------

def or_event_dates(info: dict) -> str:
    return f"Starts {info['start']:%b} {info['start'].day}, {info['start'].year}" if info.get("start") else "To be announced"


def record_from_openreview(info: dict, *, title: str, year: int, fields: list[str], africa: bool) -> dict:
    deadline = info["deadline_utc"]
    full_name = info["title"] if plain_differs(info["title"], title) else ""
    return {
        "id": info["id"].lower(),
        "name": f"{title} {year}",
        "year": year,
        "full_name": full_name,
        "submission_type": "Paper (OpenReview)",
        "deadline_utc": deadline,
        "deadline_timezone": "AoE" if orv.looks_like_aoe(deadline) else "UTC",
        "abstract_deadline_utc": None,
        "deadline_source": "OpenReview",
        "openreview_deadline_utc": deadline,
        "openreview_link": info["link"],
        "upcoming": False,
        "expected_deadline_utc": None,
        "last_deadline_utc": None,
        "last_edition": None,
        "fields": fields or ["Other"],
        "event_dates": or_event_dates(info),
        "event_end": iso_day(info.get("start")),
        "location": info["location"] or "To be announced",
        "link": info["website"] or info["link"],
        "africa": africa,
        "source": "OpenReview",
    }


def plain_differs(a: str, b: str) -> bool:
    return bool(a) and orv.plain(a) != orv.plain(b)


def add_openreview(records: list[dict], venues_file: Path, api, active: list[str]) -> list[dict]:
    """Add OpenReview dates to matching conferences, and add the extra venues from venues_file."""
    index = orv.conference_index(active)
    by_key = {orv.plain(r["name"].removesuffix(str(r["year"]))): r for r in records}
    matched = added = 0

    # 1. Conferences already in the ai-deadlines list.
    for key, rec in list(by_key.items()):
        if rec["year"] == 0 or key not in index:
            continue
        year, vid = index[key][0]  # newest edition on OpenReview
        if year < rec["year"]:
            continue
        info = api.venue(vid)
        if not info:
            continue
        if year == rec["year"]:
            rec["openreview_deadline_utc"] = info["deadline_utc"]
            rec["openreview_link"] = info["link"]
            if rec["deadline_utc"] is None:  # ai-deadlines has no date yet
                rec["deadline_utc"] = info["deadline_utc"]
                rec["deadline_timezone"] = "AoE" if orv.looks_like_aoe(info["deadline_utc"]) else "UTC"
                rec["deadline_source"] = "OpenReview"
                rec["submission_type"] = "Paper (OpenReview)"
            if not rec["full_name"] and plain_differs(info["title"], rec["name"]):
                rec["full_name"] = info["title"]
            matched += 1
        else:  # OpenReview already has the next edition: show that one instead
            title = rec["name"].removesuffix(str(rec["year"])).strip()
            new = record_from_openreview(info, title=title, year=year, fields=rec["fields"], africa=rec["africa"])
            records[records.index(rec)] = new
            by_key[key] = new
            added += 1

    # 2. Extra venues kept by hand.
    extra = yaml.safe_load(venues_file.read_text(encoding="utf-8")) if venues_file.exists() else []
    for item in extra or []:
        vid = orv.newest_matching(active, str(item.get("openreview", "")))
        title = str(item.get("title") or "").strip()
        if not vid or not title:
            continue
        _, year, _ = orv.split_venue_id(vid)
        key = orv.plain(title)
        if key in by_key and by_key[key]["year"] >= year:
            continue  # already on the page from ai-deadlines
        info = api.venue(vid)
        if not info:
            continue
        fields = fields_from({"field": item.get("field")})
        new = record_from_openreview(info, title=title, year=year, fields=fields, africa=bool(item.get("africa")))
        if key in by_key:
            records[records.index(by_key[key])] = new
        else:
            records.append(new)
        by_key[key] = new
        added += 1

        new["cycle"] = int(item.get("cycle") or KNOWN_CYCLE.get(orv.plain(title), 1))
        new["_template"] = str(item.get("openreview", ""))

    print(f"  OpenReview: {matched} dates added to listed conferences, {added} venues added")
    return records


# ---------- Upcoming editions ----------

def add_years(iso: str, years: int) -> str:
    t = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ")
    try:
        t = t.replace(year=t.year + years)
    except ValueError:  # 29 February
        t = t.replace(year=t.year + years, day=28)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def event_is_over(rec: dict, today: date) -> bool:
    if rec.get("event_end"):
        end = date.fromisoformat(rec["event_end"])
        # Only a start date (OpenReview): allow a week for the event itself.
        return end + (timedelta(days=7) if rec["source"] == "OpenReview" else timedelta()) < today
    # No event dates: workshops usually happen within about four months of their deadline.
    deadline = datetime.strptime(rec["deadline_utc"], "%Y-%m-%dT%H:%M:%SZ").date()
    return deadline + timedelta(days=120) < today


def add_upcoming(records: list[dict], api, active: list[str]) -> list[dict]:
    """Swap finished conferences for their next edition, with the deadline marked as not announced.

    The card shows last edition's deadline and an expected month (last deadline + one cycle).
    If OpenReview already lists the next edition, its name, place and dates are used."""
    today = datetime.now(timezone.utc).date()
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    index = orv.conference_index(active) if active else {}
    out, made = [], 0
    for rec in records:
        if not rec["deadline_utc"] or rec["deadline_utc"] > now_iso or rec["year"] == 0 or not event_is_over(rec, today):
            out.append(rec)
            continue
        cycle = rec.get("cycle", 1)
        year = rec["year"] + cycle
        title = rec["name"].removesuffix(str(rec["year"])).strip()
        new = dict(rec)
        new.update({
            "id": f"{orv.plain(title)}{year}",
            "name": f"{title} {year}",
            "year": year,
            "submission_type": "Paper",
            "deadline_utc": None,
            "deadline_timezone": None,
            "abstract_deadline_utc": None,
            "deadline_source": "Expected from last edition",
            "openreview_deadline_utc": None,
            "openreview_link": None,
            "upcoming": True,
            "expected_deadline_utc": add_years(rec["deadline_utc"], cycle),
            "last_deadline_utc": rec["deadline_utc"],
            "last_edition": rec["name"],
            "event_dates": "To be announced",
            "event_end": None,
            "location": "To be announced",
        })
        # Does OpenReview already list the next edition (even without a submission form yet)?
        vid = None
        if rec.get("_template"):
            vid = orv.newest_matching(active, rec["_template"]) if active else None
            if vid and orv.split_venue_id(vid)[1] != year:
                vid = None
        else:
            for y, v in index.get(orv.plain(title), []):
                if y == year:
                    vid = v
        if vid and api:
            info = api.details(vid)
            if info.get("title") and plain_differs(info["title"], title):
                new["full_name"] = info["title"]
            new["location"] = info.get("location") or new["location"]
            new["event_dates"] = or_event_dates(info)
            new["link"] = info.get("website") or new["link"]
            new["openreview_link"] = info["link"]
        out.append(new)
        made += 1
    print(f"  {made} finished conferences replaced by their next edition (deadline not announced yet)")
    return out


AFRICAN_COUNTRIES = {
    "algeria", "angola", "benin", "botswana", "burkina faso", "burundi", "cabo verde", "cape verde", "cameroon",
    "central african republic", "chad", "comoros", "congo", "côte d'ivoire", "cote d'ivoire", "ivory coast",
    "djibouti", "egypt", "equatorial guinea", "eritrea", "eswatini", "ethiopia", "gabon", "gambia", "ghana",
    "guinea", "guinea-bissau", "kenya", "lesotho", "liberia", "libya", "madagascar", "malawi", "mali",
    "mauritania", "mauritius", "morocco", "mozambique", "namibia", "niger", "nigeria", "rwanda",
    "são tomé and príncipe", "sao tome and principe", "senegal", "seychelles", "sierra leone", "somalia",
    "south africa", "south sudan", "sudan", "tanzania", "togo", "tunisia", "uganda", "zambia", "zimbabwe",
}


# Characters you cannot see that sometimes come in with copied text,
# such as the zero-width non-joiner in OpenReview's "C\u200conference".
INVISIBLE = re.compile("[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")


def clean_text(rec: dict) -> None:
    """Remove invisible characters and extra spaces from every text field."""
    for key, value in rec.items():
        if isinstance(value, str):
            rec[key] = " ".join(INVISIBLE.sub("", value).split())


def apply_full_names(records: list[dict], path: Path) -> None:
    """Fill in full names from the hand-kept file (it wins over the sources)."""
    if not path.exists():
        return
    names = {str(k).strip().lower(): str(v).strip() for k, v in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).items()}
    for rec in records:
        title = rec["name"].removesuffix(str(rec["year"])).strip().lower()
        if title in names:
            rec["full_name"] = names[title]


def held_in_africa(location: str) -> bool:
    """True when the last part of 'City, Country' is an African country."""
    country = str(location or "").split(",")[-1].strip().lower()
    return country in AFRICAN_COUNTRIES


def sort_key(rec: dict):
    # Same order as the page before it knows the time: by deadline, unknown deadlines last.
    return (rec["deadline_utc"] is None, rec["deadline_utc"] or "", rec["name"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=sorted(SOURCES), default="api")
    ap.add_argument("--no-openreview", action="store_true", help="do not check OpenReview")
    ap.add_argument("--openreview-venues", type=Path, default=DEFAULT_OPENREVIEW,
                    help="extra OpenReview venues to show (YAML)")
    ap.add_argument("--african", type=Path, default=DEFAULT_AFRICAN, help="hand-kept African venues (YAML)")
    ap.add_argument("--names", type=Path, default=DEFAULT_NAMES, help="hand-kept full names (YAML)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    src = SOURCES[args.source]
    print(f"Downloading {src['label']} ...")
    raw = fetch(src["url"])
    reader = {"api": read_api, "paperswithcode": read_paperswithcode, "huggingface": read_huggingface}[args.source]
    records = reader(raw)
    records = latest_editions(records)
    print(f"  {len(records)} conferences (newest edition of each)")

    sources = [{"label": src["label"], "url": src["page"]}]
    api, active = None, []
    if not args.no_openreview:
        print("Checking OpenReview ...")
        api = orv.OpenReview()
        active = api.active_venues()
        if active:
            records = add_openreview(records, args.openreview_venues, api, active)
            sources.append({"label": "OpenReview", "url": "https://openreview.net/venues"})
        else:
            orv.warn("could not read the OpenReview venue list; skipping OpenReview")
    records = add_upcoming(records, api, active)
    if api:
        print(f"  OpenReview: {api.calls} calls, {api.failures} failed")
    for rec in records:  # working values, not needed on the page
        rec.pop("cycle", None)
        rec.pop("_template", None)
        if held_in_africa(rec["location"]):
            rec["africa"] = True

    african = read_african(args.african)
    print(f"  {len(african)} African venues from {args.african.name}")
    # A hand-kept entry replaces an ai-deadlines entry with the same name.
    names = {r["name"].lower() for r in african}
    records = [r for r in records if r["name"].lower() not in names] + african
    apply_full_names(records, args.names)
    for rec in records:
        clean_text(rec)
    records.sort(key=sort_key)

    payload = {
        "sources": sources,
        "conferences": records,
    }

    # Only touch the file when the data changed, so the daily job does not make empty commits.
    old = None
    if args.out.exists():
        try:
            old = json.loads(args.out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old = None
    if old and {k: v for k, v in old.items() if k != "updated_at"} == payload:
        print(f"No changes; {args.out} left as it is.")
        return 0

    payload = {"updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **payload}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} entries to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
