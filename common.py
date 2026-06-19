#!/usr/bin/env python3
"""AdaMem common utilities.

Shared infrastructure for AdaMem experiments:
- Paths (DATA_DIR / EXP_DIR rooted at this folder)
- Model clients (gen / answer / judge) over an OpenAI-compatible endpoint
- Embedding over an OpenAI-compatible embedding endpoint
- mem0 client factory + observation-date / token-accounting patches
- Progress file with cross-process lock
- Token usage accounting
- PREFERENCES / STORY_THEMES (re-exported from the original scaling spec)
- Feedback formatting (with_gold | verbose) for the multi-turn QA loop
- Judge wrapper

This module is method-agnostic. FC / M0 / ID / AdaMem all import from here.

Endpoints / credentials are read from the environment (no provider URL or key
is hard-coded). Set the following before running any method:
    OPENAI_BASE_URL / OPENAI_API_KEY          chat completions
    EMBEDDING_BASE_URL / EMBEDDING_API_KEY    embeddings
"""

import json, os, sys, time, math, shutil, requests, traceback, threading
from datetime import datetime, date, timedelta

os.environ['PYTHONUNBUFFERED'] = '1'
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Optionally load a .env that lives next to this file (never shipped with the
# submission). Keeps things working no matter where the entry script launches.
try:
    from dotenv import load_dotenv
    _ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_ENV_PATH, override=False)
except Exception:
    pass

# IMPORTANT: must be set before any `import mem0.*`. mem0 captures this at
# module-load time and otherwise spins up a singleton telemetry vector store
# under the user home dir, which breaks parallel-process runs.
os.environ.setdefault("MEM0_TELEMETRY", "False")

# ==================== PATHS ====================
ADAMEM_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ADAMEM_ROOT, "data", "scaling")
# EXP_DIR can be overridden via env (e.g. ADAMEM_EXP_TAG=scaling_v2) so we
# can keep historic runs in exp/scaling/ untouched while writing fresh
# results into a sibling directory like exp/scaling_v2/. DATA_DIR stays
# fixed -- input stories are shared.
_EXP_TAG = os.environ.get("ADAMEM_EXP_TAG", "scaling")
EXP_DIR = os.path.join(ADAMEM_ROOT, "exp", _EXP_TAG)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)

# ==================== MODELS ====================
# Paper-experiment routing: four LLM uses are fully decoupled, only MEMORY_MODEL
# is a runtime experiment dimension. See requirements.md §2 for the rationale.
#
#   DATA_GEN_MODEL   - prepare_data.py: stories, dialogues, QA,
#                       golden_feedback, golden_memories, character_profiles.
#                       Fixed across the whole 32-cell matrix so the dataset
#                       itself never becomes an experiment variable.
#   QA_ANSWER_MODEL  - The model that answers test QAs in run_id / run_fc /
#                       run_m0 / run_adamem. Fixed across the whole matrix so
#                       "answer quality" is held constant; the only variable is
#                       what memories the answerer is fed.
#   MEMORY_MODEL     - mem0 fact extraction + AdaMem policy reflection. This
#                       *is* the experiment dimension: cycled over 4 candidates.
#                       Configured via env ADAMEM_MEMORY_MODEL or set_memory_model().
#   JUDGE_MODEL      - LLM-as-judge for both "answer correct/wrong" and
#                       post-experiment extraction/recall judging. Fixed.
#
DATA_GEN_MODEL  = "gemini-3.1-pro"
QA_ANSWER_MODEL = "deepseek-v4-flash"
JUDGE_MODEL     = "deepseek-v4-flash"

# Allowed MEMORY_MODEL names. Whatever string we pass is sent to the
# OpenAI-compatible endpoint verbatim as the ``model`` field. The paper reports
# the two extraction models below; add more here to extend the matrix.
MEMORY_MODEL_CHOICES = (
    "deepseek-v4-flash",
    "gemini-3.5-flash",
)

_DEFAULT_MEMORY_MODEL_ALIAS = "deepseek-v4-flash"

def _validate_memory_model(alias):
    """Validate a vendor alias against :data:`MEMORY_MODEL_CHOICES`.

    Returns the alias unchanged on success; raises ValueError on typo so the
    matrix fails fast at startup instead of silently 4xx-ing every cell.
    """
    if alias in MEMORY_MODEL_CHOICES:
        return alias
    raise ValueError(
        f"unknown MEMORY_MODEL alias {alias!r}; expected one of "
        f"{list(MEMORY_MODEL_CHOICES)}"
    )

_MEMORY_MODEL_ALIAS = os.environ.get("ADAMEM_MEMORY_MODEL", _DEFAULT_MEMORY_MODEL_ALIAS)
MEMORY_MODEL = _validate_memory_model(_MEMORY_MODEL_ALIAS)

# Some OpenAI-compatible endpoints reject unknown request-body fields. The
# optional per-call ``client_metadata`` block (used only for request tagging)
# is therefore not sent by default. Set ADAMEM_SEND_CLIENT_METADATA=1 to
# enable it for endpoints that tolerate extra body fields.
_SEND_CLIENT_METADATA = os.environ.get("ADAMEM_SEND_CLIENT_METADATA", "0") == "1"


def _should_drop_client_metadata(model_alias):
    """True iff we should NOT attach client_metadata to the request body."""
    return not _SEND_CLIENT_METADATA

def set_memory_model(alias):
    """Override MEMORY_MODEL at runtime (used by run_*.py CLI ``--memory-model``).

    Validates ``alias`` against :data:`MEMORY_MODEL_CHOICES` and returns it.
    """
    global _MEMORY_MODEL_ALIAS, MEMORY_MODEL
    _MEMORY_MODEL_ALIAS = alias
    MEMORY_MODEL = _validate_memory_model(alias)
    return MEMORY_MODEL

def get_memory_model_alias():
    return _MEMORY_MODEL_ALIAS

# ==================== DEFAULTS ====================
NUM_STORIES_DEFAULT = 10
NUM_WEEKS_DEFAULT = 10
CONVS_PER_WEEK_DEFAULT = 10
QA_PER_WEEK_DEFAULT = 10
DEFAULT_PARALLEL = 4
MAX_RETRY = 10
RETRY_BACKOFF_CAP = 60
EMBEDDING_BATCH = 10

# ==================== EMBEDDING / MEM0 ENV ====================
# OpenAI-compatible embedding endpoint hosting a 1024-dim embedding model.
# The same URL is used by both the manual REST call (get_embeddings_batch) and
# by mem0's OpenAI-compatible embedder (build_mem0_client below).
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")
EMBEDDING_BASE = os.environ.get("EMBEDDING_BASE_URL", "")
EMBEDDING_URL = EMBEDDING_BASE.rstrip("/") + "/embeddings" if EMBEDDING_BASE else ""
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-v4")
EMBEDDING_DIMS = int(os.environ.get("EMBEDDING_DIMS", "1024"))

# OpenAI-compatible base URL for *all* AdaMem chat traffic (mem0 fact
# extraction, AdaMem reflection, QA answering, judging, data generation).
# Read from the environment; no provider URL or credential is hard-coded.
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# qdrant root for the embedded vector store.
MEM0_QDRANT_DIR = os.path.join(DATA_DIR, "_mem0_qdrant")
os.makedirs(MEM0_QDRANT_DIR, exist_ok=True)

# ==================== PROTAGONIST / CAST ====================
PROTAGONIST = "Linchen"
PREFERENCES = {
    "Boss Zhang": {"id": "P1", "desc": "only record final decisions and conclusions"},
    "Liwei":      {"id": "P2", "desc": "record serious commitments and emotional signals (Linchen's girlfriend)"},
    "Wanghao":    {"id": "P3", "desc": "only record technical conclusions and key numbers (engineering colleague)"},
    "Sister Chen":{"id": "P4", "desc": "record emotional changes and their triggers (close friend)"},
    "Mom":        {"id": "P5", "desc": "only record Linchen's promises and major family events"},
    "Jiange":     {"id": "P6", "desc": "only record schedule/time arrangements (workout buddy)"},
}

STORY_THEMES = [
    "Linchen is a product manager at an internet company, leading an enterprise AI assistant project.",
    "Linchen is a game designer at a gaming studio, developing a new mobile game.",
    "Linchen is a backend engineer at a fintech company, building a trading system.",
    "Linchen is the operations lead at an e-commerce company, preparing for the November 11 sales event.",
    "Linchen is a frontend engineer at an EdTech company, refactoring an online classroom product.",
    "Linchen is a co-founder of a medical-AI startup, preparing for the Series A round.",
    "Linchen is an algorithm engineer at a short-video platform, optimizing the recommendation system.",
    "Linchen is a test engineer at an autonomous-driving company, working on road-test acceptance.",
    "Linchen is a tech lead at a SaaS company, leading the team through a microservices split.",
    "Linchen is a data analyst at a social-product company, working on user-growth analysis.",
]

# ==================== TOPIC ANCHOR POOL ====================
# Shared across all stories. The outline LLM picks 3-5 anchors per week from
# this list and is encouraged to reuse them across weeks for stress-testing
# semantically-similar memory retrieval. Must contain >= 30 entries; each is a
# concrete recurring noun phrase (work / family / friends / personal life).
ANCHOR_POOL = [
    # work / project life
    "the morning coffee chat",
    "the weekly sync",
    "the on-call rotation",
    "the production incident",
    "the new conference room",
    "the all-hands meeting",
    "the OKR review",
    "the Shenzhen trip",
    "the Beijing business trip",
    "the customer demo",
    "the code review",
    "the design doc",
    "the year-end performance review",
    "the team offsite",
    # romance / partner
    "the weekend dinner",
    "the anniversary plan",
    "the apartment hunting",
    "the in-laws visit",
    "the Saturday brunch",
    "the movie night",
    # family
    "Mom's birthday dinner",
    "the family WeChat call",
    "the Spring Festival trip home",
    "the medical check-up reminder",
    # friends / social
    "the badminton session",
    "the swimming pool plan",
    "the gym membership",
    "the hot pot gathering",
    "the board game night",
    # consumer / tech / finance
    "the Pixel 9 phone",
    "the new MacBook",
    "the 618 promo",
    "the Double Eleven sale",
    "the credit card bill",
    "the apartment rent renewal",
    # health / lifestyle
    "the running plan",
    "the dentist appointment",
    "the vaccination shot",
    "the sleep schedule",
]
assert (
    isinstance(ANCHOR_POOL, list)
    and len(ANCHOR_POOL) >= 30
    and all(isinstance(a, str) and a.strip() for a in ANCHOR_POOL)
), "ANCHOR_POOL must be a non-empty list of >=30 trimmed strings"

# ==================== LOGGING ====================
import logging, resource

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,
)
log = logging.getLogger('adamem')
log.setLevel(logging.DEBUG)
for lib in ['httpx', 'httpcore', 'urllib3', 'openai', 'requests']:
    logging.getLogger(lib).setLevel(logging.WARNING)

def mem_mb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024

# ==================== PROGRESS (cross-process lock) ====================
import fcntl as _fcntl

PROGRESS_FILE = os.path.join(DATA_DIR, "progress.json")
_PROGRESS_LOCK_FILE = os.path.join(DATA_DIR, ".progress.lock")

class _ProgressFileLock:
    def __init__(self, path):
        self.path = path
        self.fp = None
    def __enter__(self):
        self.fp = open(self.path, "a+")
        _fcntl.flock(self.fp, _fcntl.LOCK_EX)
        return self
    def __exit__(self, *a):
        try:
            _fcntl.flock(self.fp, _fcntl.LOCK_UN)
        finally:
            self.fp.close()

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE) as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"progress.json read error ({e}), starting fresh")
    return {"started_at": datetime.now().isoformat(), "stories": {}}

def save_progress(progress):
    progress["updated_at"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def update_progress(updater):
    with _ProgressFileLock(_PROGRESS_LOCK_FILE):
        progress = load_progress()
        updater(progress)
        save_progress(progress)

# ==================== TOKEN ACCOUNTING ====================
_TOKEN_USAGE = {}
_TOKEN_USAGE_LOCK = threading.Lock()

def _record_tokens(bucket, prompt_tokens=0, completion_tokens=0, total_tokens=0):
    with _TOKEN_USAGE_LOCK:
        b = _TOKEN_USAGE.setdefault(bucket, {"prompt": 0, "completion": 0, "total": 0, "calls": 0})
        b["prompt"] += int(prompt_tokens or 0)
        b["completion"] += int(completion_tokens or 0)
        b["total"] += int(total_tokens or (int(prompt_tokens or 0) + int(completion_tokens or 0)))
        b["calls"] += 1

def _record_from_response(bucket, resp):
    try:
        usage = (resp.raw_response or {}).get("usage") or {}
    except Exception:
        usage = {}
    _record_tokens(bucket,
                   prompt_tokens=usage.get("prompt_tokens", 0),
                   completion_tokens=usage.get("completion_tokens", 0),
                   total_tokens=usage.get("total_tokens", 0))

def get_token_usage_snapshot():
    with _TOKEN_USAGE_LOCK:
        return {k: dict(v) for k, v in _TOKEN_USAGE.items()}

def reset_token_usage():
    with _TOKEN_USAGE_LOCK:
        _TOKEN_USAGE.clear()

# ==================== LLM CALLS ====================
# Single source of truth for chat traffic: all chat calls go through the
# OpenAI-compatible the chat endpoint at OPENAI_BASE_URL so every request is recorded
# under the chat endpoint/logs/. Each call may carry a business-layer
# ``client_metadata`` block (tag/story_id/week/kind) that the chat endpoint persists
# to disk and uses as the record filename prefix.

try:
    from openai import OpenAI as _OpenAIClient
    _OPENAI_SDK_AVAILABLE = True
except Exception:  # pragma: no cover - openai SDK is a hard dep but be defensive
    _OpenAIClient = None
    _OPENAI_SDK_AVAILABLE = False

_OPENAI_CLIENT_TLS = threading.local()


def _get_openai_client():
    """Return a process-local OpenAI client pointed at OPENAI_BASE_URL.

    The client is cached per-thread (OpenAI SDK is thread-safe but its httpx
    pool is not free to construct so we avoid making one per request).
    """
    if not _OPENAI_SDK_AVAILABLE:
        return None
    cli = getattr(_OPENAI_CLIENT_TLS, "client", None)
    if cli is not None:
        return cli
    cli = _OpenAIClient(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY, timeout=120.0)
    _OPENAI_CLIENT_TLS.client = cli
    return cli


def _build_client_metadata(*, tag=None, story_id=None, week=None, kind=None,
                           extra=None):
    """Build the dict consumed by the chat endpoint's ``client_metadata`` field.

    Returns ``None`` when nothing meaningful was passed, so that for callers
    that don't tag anything we keep request bodies clean.
    """
    md = {}
    if tag is not None:
        md["tag"] = str(tag)
    if story_id is not None:
        md["story_id"] = story_id
    if week is not None:
        md["week"] = week
    if kind is not None:
        md["kind"] = kind
    if extra:
        for k, v in extra.items():
            if k not in md:
                md[k] = v
    return md or None


def _retry_sleep(attempt):
    return min(2 ** (attempt + 1), RETRY_BACKOFF_CAP)


def _call_chat_endpoint_once(model, messages, *, max_tokens, temperature, timeout,
                      response_format, client_metadata):
    """One-shot non-streaming chat call against the chat endpoint via OpenAI SDK.

    Returns ``(text, usage_dict)`` or raises on hard failure. The caller is
    responsible for retrying.
    """
    cli = _get_openai_client()
    if cli is None:
        raise RuntimeError("openai SDK not available; cannot reach the chat endpoint")
    extra_body = {}
    if client_metadata is not None and not _should_drop_client_metadata(model):
        extra_body["client_metadata"] = client_metadata
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    if response_format is not None:
        kwargs["response_format"] = response_format
    if extra_body:
        kwargs["extra_body"] = extra_body
    resp = cli.chat.completions.create(**kwargs)
    # OpenAI SDK guarantees .choices[0].message.content for non-stream calls.
    text = ""
    try:
        text = resp.choices[0].message.content or ""
    except Exception:
        text = ""
    usage = {}
    try:
        u = getattr(resp, "usage", None)
        if u is not None:
            usage = {
                "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                "total_tokens": getattr(u, "total_tokens", 0) or 0,
            }
    except Exception:
        usage = {}
    return text, usage


def _call_chat_endpoint(model, prompt, *, max_tokens, temperature, timeout, bucket,
                response_format=None, client_metadata=None,
                messages=None):
    """Chat with retry through the chat endpoint only.

    Either ``prompt`` (single user turn) or ``messages`` (full chat history)
    must be provided. ``messages`` wins if both are given. No legacy endpoint
    fallback is allowed because paper experiments require complete proxy logs.
    """
    chat_messages = messages if messages is not None else (
        [{"role": "user", "content": prompt}] if prompt is not None else []
    )
    if not chat_messages:
        raise ValueError("_call_chat_endpoint requires either prompt or messages")
    last_proxy_err = None
    for attempt in range(MAX_RETRY):
        t0 = time.time()
        try:
            text, usage = _call_chat_endpoint_once(
                model, chat_messages,
                max_tokens=max_tokens, temperature=temperature, timeout=timeout,
                response_format=response_format, client_metadata=client_metadata,
            )
            elapsed = time.time() - t0
            if text:
                _record_tokens(
                    bucket,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                )
                log.debug(f"call_{bucket} OK (proxy) {elapsed:.1f}s len={len(text)} mem={mem_mb()}MB")
                return text
            log.warning(f"call_{bucket} empty (proxy) attempt={attempt+1} {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            last_proxy_err = e
            log.error(f"call_{bucket} EXCEPTION (proxy) attempt={attempt+1} {elapsed:.1f}s: {type(e).__name__}: {e}")
        time.sleep(_retry_sleep(attempt))
    raise RuntimeError(
        f"call_{bucket} FAILED after {MAX_RETRY} attempts via the chat endpoint; "
        f"OPENAI_BASE_URL={OPENAI_BASE_URL!r}; last error={last_proxy_err!r}"
    )


def make_chat_call(model, prompt=None, *, messages=None,
                   max_tokens=8192, temperature=0, timeout=120,
                   bucket="adhoc", response_format=None,
                   tag=None, story_id=None, week=None, kind=None, extra=None):
    """Public, business-friendly wrapper around ``_call_chat_endpoint``.

    Always uses the chosen ``model`` (no implicit routing). Adds an OpenAI-
    compatible ``client_metadata`` block so the chat endpoint's request log can be
    audited by tag (``memory_s1_w1_2015-07-15`` / ``qa_s1_w1_q3`` /
    ``policy_s1_w1`` / ``data_prep_s1_w1`` / ``judge_*``).
    """
    cm = _build_client_metadata(
        tag=tag, story_id=story_id, week=week, kind=kind, extra=extra,
    )
    return _call_chat_endpoint(
        model, prompt,
        max_tokens=max_tokens, temperature=temperature, timeout=timeout,
        bucket=bucket, response_format=response_format,
        client_metadata=cm, messages=messages,
    )

def call_gen(prompt, max_tokens=8192, temperature=0.7, *,
             tag=None, story_id=None, week=None, kind="data_prep"):
    return make_chat_call(
        DATA_GEN_MODEL, prompt,
        max_tokens=max_tokens, temperature=temperature, timeout=120, bucket="gen",
        tag=tag, story_id=story_id, week=week, kind=kind,
    )

def call_answer(prompt, max_tokens=8192, temperature=0, *,
                tag=None, story_id=None, week=None, kind="qa"):
    return make_chat_call(
        QA_ANSWER_MODEL, prompt,
        max_tokens=max_tokens, temperature=temperature, timeout=60, bucket="answer",
        tag=tag, story_id=story_id, week=week, kind=kind,
    ) or "ERROR"

def call_answer_messages(messages, max_tokens=8192, temperature=0, *,
                         tag=None, story_id=None, week=None, kind="qa"):
    """Same as :func:`call_answer` but takes a full ``messages`` list (FC uses this)."""
    return make_chat_call(
        QA_ANSWER_MODEL, messages=messages,
        max_tokens=max_tokens, temperature=temperature, timeout=60, bucket="answer",
        tag=tag, story_id=story_id, week=week, kind=kind,
    ) or "ERROR"

def call_judge(prompt, *, tag=None, story_id=None, week=None, kind="judge"):
    return make_chat_call(
        JUDGE_MODEL, prompt,
        max_tokens=8192, temperature=0, timeout=60, bucket="judge",
        tag=tag, story_id=story_id, week=week, kind=kind,
    ) or "WRONG"

def call_reflect(prompt, max_tokens=8192, temperature=0, force_json=False, *,
                 tag=None, story_id=None, week=None, kind="policy_update"):
    """AdaMem policy reflector. Uses the current MEMORY_MODEL.

    When force_json=True, request strict JSON object output from the underlying
    LLM (response_format={"type": "json_object"}). Falls back gracefully if the
    backend ignores it.
    """
    rf = {"type": "json_object"} if force_json else None
    return make_chat_call(
        MEMORY_MODEL, prompt,
        max_tokens=max_tokens, temperature=temperature, timeout=120, bucket="reflect",
        response_format=rf,
        tag=tag, story_id=story_id, week=week, kind=kind,
    ) or ""

def parse_json_from_llm(text):
    if not text:
        return None
    text = text.strip()
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        text = text.split('```')[1].split('```')[0].strip()
    arr_start = text.find('[')
    arr_end = text.rfind(']')
    obj_start = text.find('{')
    obj_end = text.rfind('}')
    candidates = []
    if obj_start >= 0 and obj_end > obj_start:
        candidates.append((obj_start, text[obj_start:obj_end + 1]))
    if arr_start >= 0 and arr_end > arr_start:
        candidates.append((arr_start, text[arr_start:arr_end + 1]))
    candidates.sort(key=lambda x: x[0])
    last_err = None
    for _, snippet in candidates:
        try:
            return json.loads(snippet)
        except Exception as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    return json.loads(text)

# ==================== EMBEDDING ====================
def get_embeddings_batch(texts, batch_size=EMBEDDING_BATCH):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(MAX_RETRY):
            t0 = time.time()
            try:
                resp = requests.post(
                    EMBEDDING_URL,
                    headers={'Authorization': f'Bearer {EMBEDDING_API_KEY}', 'Content-Type': 'application/json'},
                    json={'model': EMBEDDING_MODEL, 'input': batch}, timeout=30,
                )
                elapsed = time.time() - t0
                if resp.status_code == 429:
                    log.warning(f"embedding 429 attempt={attempt+1} {elapsed:.1f}s")
                    time.sleep(_retry_sleep(attempt))
                    continue
                resp.raise_for_status()
                payload = resp.json()
                all_embs.extend([d['embedding'] for d in payload['data']])
                usage = payload.get('usage') or {}
                _record_tokens("embedding",
                               prompt_tokens=usage.get("prompt_tokens", usage.get("total_tokens", 0)),
                               total_tokens=usage.get("total_tokens", 0))
                break
            except Exception as e:
                elapsed = time.time() - t0
                log.error(f"embedding EXCEPTION attempt={attempt+1} {elapsed:.1f}s: {type(e).__name__}: {e}")
                if attempt == MAX_RETRY - 1:
                    log.error(f"embedding FAILED, zeros for {len(batch)}")
                    all_embs.extend([[0] * EMBEDDING_DIMS] * len(batch))
                else:
                    time.sleep(_retry_sleep(attempt))
        time.sleep(0.1)
    return all_embs

def get_embedding(text):
    return get_embeddings_batch([text])[0]

# ==================== MEM0 CLIENT ====================
_MEM0_TLS = threading.local()

def set_mem0_observation_date(date_str):
    _MEM0_TLS.observation_date = date_str

def clear_mem0_observation_date():
    _MEM0_TLS.observation_date = None

def set_mem0_custom_instructions(instructions):
    """AdaMem hook: per-add custom_instructions injected into mem0's fact extractor."""
    _MEM0_TLS.custom_instructions = instructions

def clear_mem0_custom_instructions():
    _MEM0_TLS.custom_instructions = None

def set_mem0_client_metadata(metadata):
    """Per-add metadata injected into mem0's internal OpenAI LLM calls."""
    _MEM0_TLS.client_metadata = metadata

def clear_mem0_client_metadata():
    _MEM0_TLS.client_metadata = None

def _patch_mem0():
    """Patch mem0:
    1. Honor per-add observation_date (for back-dated conversations).
    2. Honor per-add custom_instructions (AdaMem policy injection).
    3. Force extraction prompts to ignore mem0's last_k_messages history.
    4. Token-accounting hooks for DeepSeek LLM and OpenAI embedder.
    Idempotent: only patches once per process.
    """
    import mem0.memory.main as _mm
    if getattr(_mm, "_adamem_patched", False):
        return
    _orig = _mm.generate_additive_extraction_prompt

    def _wrapper(*args, **kwargs):
        obs = getattr(_MEM0_TLS, "observation_date", None)
        if obs and not kwargs.get("timestamp") and not kwargs.get("current_date"):
            kwargs["timestamp"] = obs
            kwargs["current_date"] = obs
        ci = getattr(_MEM0_TLS, "custom_instructions", None)
        if ci and not kwargs.get("custom_instructions"):
            kwargs["custom_instructions"] = ci
        # Paper benchmark invariant: each mem0.add call should extract facts
        # only from its current New Messages batch. mem0 normally injects
        # recent stored context as last_k_messages, which can leak previous
        # days' facts into the current day's extraction prompt.
        kwargs["last_k_messages"] = []
        kwargs.setdefault("use_input_language", True)
        return _orig(*args, **kwargs)

    _mm.generate_additive_extraction_prompt = _wrapper
    _mm._adamem_patched = True
    log.info("mem0 patched (observation_date + custom_instructions + empty last_k_messages + use_input_language)")

    # Token hooks
    try:
        from mem0.llms.deepseek import DeepSeekLLM
        if not getattr(DeepSeekLLM, "_token_usage_patched", False):
            def _gen_with_usage(self, messages, response_format=None, tools=None, tool_choice="auto", **kwargs):
                params = self._get_supported_params(messages=messages, **kwargs)
                params.update({"model": self.config.model, "messages": messages})
                if response_format:
                    params["response_format"] = response_format
                if tools:
                    params["tools"] = tools
                    params["tool_choice"] = tool_choice
                response = self.client.chat.completions.create(**params)
                try:
                    u = getattr(response, "usage", None)
                    if u is not None:
                        _record_tokens("mem0_llm",
                                       prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                                       completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                                       total_tokens=getattr(u, "total_tokens", 0) or 0)
                except Exception:
                    pass
                return self._parse_response(response, tools)
            DeepSeekLLM.generate_response = _gen_with_usage
            DeepSeekLLM._token_usage_patched = True
    except Exception as e:
        log.warning(f"failed to patch mem0 DeepSeekLLM token hook: {e}")

    # OpenAI LLM hook (used after we route mem0 through the chat endpoint in
    # build_mem0_client). Same accounting as DeepSeekLLM but on the upstream
    # OpenAI-compatible client.
    try:
        from mem0.llms.openai import OpenAILLM
        if not getattr(OpenAILLM, "_token_usage_patched", False):
            def _openai_gen_with_usage(self, messages, response_format=None, tools=None,
                                       tool_choice="auto", **kwargs):
                params = self._get_supported_params(messages=messages, **kwargs)
                params.update({"model": self.config.model, "messages": messages})
                if response_format:
                    params["response_format"] = response_format
                if tools:
                    params["tools"] = tools
                    params["tool_choice"] = tool_choice
                client_metadata = getattr(_MEM0_TLS, "client_metadata", None)
                if client_metadata and not _should_drop_client_metadata(self.config.model):
                    extra_body = dict(params.get("extra_body") or {})
                    extra_body.setdefault("client_metadata", client_metadata)
                    params["extra_body"] = extra_body
                response = self.client.chat.completions.create(**params)
                try:
                    u = getattr(response, "usage", None)
                    if u is not None:
                        _record_tokens(
                            "mem0_llm",
                            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                            total_tokens=getattr(u, "total_tokens", 0) or 0,
                        )
                except Exception:
                    pass
                return self._parse_response(response, tools)
            OpenAILLM.generate_response = _openai_gen_with_usage
            OpenAILLM._token_usage_patched = True
    except Exception as e:
        log.warning(f"failed to patch mem0 OpenAILLM token hook: {e}")

    try:
        from mem0.embeddings.openai import OpenAIEmbedding
        if not getattr(OpenAIEmbedding, "_token_usage_patched", False):
            def _embed_with_usage(self, text, memory_action=None):
                t2 = text.replace("\n", " ")
                kw = {"input": [t2], "model": self.config.model, "encoding_format": "float"}
                if self._pass_dimensions_to_api:
                    kw["dimensions"] = self.config.embedding_dims
                resp = self.client.embeddings.create(**kw)
                try:
                    u = getattr(resp, "usage", None)
                    if u is not None:
                        _record_tokens("embedding",
                                       prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                                       total_tokens=getattr(u, "total_tokens", 0) or 0)
                except Exception:
                    pass
                return resp.data[0].embedding
            def _embed_batch_with_usage(self, texts, memory_action="add"):
                MAX_BATCH = 100
                texts = [t.replace("\n", " ") for t in texts]
                out = []
                for i in range(0, len(texts), MAX_BATCH):
                    chunk = texts[i:i + MAX_BATCH]
                    kw = {"input": chunk, "model": self.config.model, "encoding_format": "float"}
                    if self._pass_dimensions_to_api:
                        kw["dimensions"] = self.config.embedding_dims
                    resp = self.client.embeddings.create(**kw)
                    try:
                        u = getattr(resp, "usage", None)
                        if u is not None:
                            _record_tokens("embedding",
                                           prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                                           total_tokens=getattr(u, "total_tokens", 0) or 0)
                    except Exception:
                        pass
                    out.extend(item.embedding for item in sorted(resp.data, key=lambda x: x.index))
                return out
            OpenAIEmbedding.embed = _embed_with_usage
            OpenAIEmbedding.embed_batch = _embed_batch_with_usage
            OpenAIEmbedding._token_usage_patched = True
    except Exception as e:
        log.warning(f"failed to patch mem0 OpenAIEmbedding token hook: {e}")

def purge_mem0_collection(collection_name):
    """Hard-wipe a mem0 / qdrant collection BEFORE building the client.

    Why this exists (P0 fix):
      The previous "soft" cleanup was ``client.delete_all(user_id=...)`` wrapped
      in a swallow-all ``except``. On the embedded qdrant we use, that does NOT
      reliably remove the on-disk vectors -- residue from earlier runs (with
      the same ``method/alias/fbmode/story_id`` collection name) was leaking
      into ``client.search`` of the *next* run, which means week-1 retrieval
      could see "future" facts written by a prior week-N run.

    What this does:
      - Best-effort: open a temporary qdrant client at the collection dir and
        call ``delete_collection`` (so any in-memory handle is dropped first).
      - Hard-delete the collection's on-disk directory under
        :data:`MEM0_QDRANT_DIR`.
      - Re-create an empty directory so :func:`build_mem0_client` can lay down
        a fresh collection on the next ``client.add`` call.
      - Verify the directory is empty after the operation; raise
        ``RuntimeError`` if not (callers must NOT swallow this).

    :param collection_name: the per-cell collection name produced by
        :func:`qdrant_collection_name`. Both the qdrant collection and its
        on-disk directory share this name in our embedded setup.
    :raises RuntimeError: if the collection directory cannot be cleaned.
    """
    qdrant_path = os.path.join(MEM0_QDRANT_DIR, collection_name)
    # Step 1: best-effort drop the qdrant collection via the SDK so any
    # background thread / lock inside qdrant releases the on-disk segment
    # files cleanly before we rmtree.
    try:
        from qdrant_client import QdrantClient
        if os.path.isdir(qdrant_path):
            qc = QdrantClient(path=qdrant_path)
            try:
                qc.delete_collection(collection_name=collection_name)
            except Exception as e:
                log.info(f"[purge_mem0] delete_collection({collection_name}) "
                         f"non-fatal: {type(e).__name__}: {e}")
            try:
                qc.close()
            except Exception:
                pass
    except Exception as e:
        # qdrant_client not importable / path locked -- fall through to rmtree.
        log.info(f"[purge_mem0] qdrant SDK drop skipped for {collection_name}: "
                 f"{type(e).__name__}: {e}")
    # Step 2: hard-wipe on-disk state. This is the *authoritative* step.
    if os.path.isdir(qdrant_path):
        try:
            shutil.rmtree(qdrant_path)
        except Exception as e:
            raise RuntimeError(
                f"purge_mem0_collection: failed to rmtree {qdrant_path}: "
                f"{type(e).__name__}: {e}"
            ) from e
    # Step 3: re-create empty dir so build_mem0_client's makedirs is a no-op.
    os.makedirs(qdrant_path, exist_ok=True)
    # Step 4: verify it is empty -- if anything (e.g. another process) raced
    # us and re-populated it, fail loudly rather than silently corrupting the
    # next run.
    leftovers = os.listdir(qdrant_path)
    if leftovers:
        raise RuntimeError(
            f"purge_mem0_collection: {qdrant_path} is not empty after wipe "
            f"(leftovers={leftovers!r}). Refusing to proceed."
        )
    log.info(f"[purge_mem0] collection={collection_name} wiped clean "
             f"(path={qdrant_path})")


def build_mem0_client(collection_name, *, memory_model_alias=None):
    """Build a self-hosted mem0 Memory client with per-collection qdrant dir.

    Routing notes (paper experiments):
      - LLM provider is set to ``openai`` and pointed at ``OPENAI_BASE_URL``,
        with ``model`` resolved from :data:`MEMORY_MODEL` so each cell of the
        memory-model dimension hits a different upstream model.
      - Embedder is OpenAI-compatible (``EMBEDDING_MODEL``, ``EMBEDDING_DIMS``,
        must match the qdrant collection dimension).
      - qdrant root is :data:`MEM0_QDRANT_DIR`.

    :param memory_model_alias: optional override; falls back to the current
        process-wide :data:`MEMORY_MODEL` when ``None``.
    """
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    _patch_mem0()
    from mem0 import Memory
    qdrant_path = os.path.join(MEM0_QDRANT_DIR, collection_name)
    os.makedirs(qdrant_path, exist_ok=True)
    model_alias = (
        _validate_memory_model(memory_model_alias) if memory_model_alias else MEMORY_MODEL
    )
    config = {
        "llm": {
            # OpenAI-compatible chat endpoint. mem0's OpenAILLM accepts
            # ``openai_base_url`` to point at any OpenAI-protocol endpoint.
            "provider": "openai",
            "config": {
                "model": model_alias,
                "api_key": OPENAI_API_KEY,
                "openai_base_url": OPENAI_BASE_URL,
                "temperature": 0,
                "max_tokens": 8192,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": EMBEDDING_MODEL,
                "api_key": EMBEDDING_API_KEY,
                "openai_base_url": EMBEDDING_BASE,
                "embedding_dims": EMBEDDING_DIMS,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "path": qdrant_path,
                "embedding_model_dims": EMBEDDING_DIMS,
            },
        },
    }
    return Memory.from_config(config)

def mem0_safe_call(fn, *args, retries=5, **kwargs):
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            log.warning(f"mem0 {fn.__name__} attempt {attempt+1} failed: {type(e).__name__}: {e}")
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, RETRY_BACKOFF_CAP))

def close_mem0(client):
    try:
        if hasattr(client.vector_store, "client"):
            client.vector_store.client.close()
    except Exception:
        pass

# ==================== JUDGE ====================
def judge_answer(question, gold_answer, ai_answer, *,
                 story_id=None, week=None, qid=None):
    """Decide whether ``ai_answer`` is materially correct vs ``gold_answer``.

    When called from a method runner, the caller SHOULD pass story_id/week/qid
    so the judge LLM call is tagged like ``judge_s1_w3_w3_q5`` and is locatable
    in the chat endpoint/logs.
    """
    if ai_answer == "ERROR":
        return False
    prompt = f"""You are evaluating an AI assistant's reply in a personal-assistant
dialogue. The user's name is "Linchen". The AI is Linchen's assistant, so in the
AI's reply pronouns map as follows:
  - "you" / "your"               -> Linchen (the user)
  - "the user"                   -> Linchen (the user)
  - the bare name "Linchen"      -> Linchen (the user)
The gold answer may use the third-person name "Linchen" while the AI answer may
use the second person "you/your"; treat them as referring to the same person.

Question: {question}
Gold answer: {gold_answer}
AI answer: {ai_answer}

Rules:
- Apply the pronoun <-> name mapping above before comparing.
- For date/time answers, the gold answer may be self-anchored in the form
  "<YYYY-MM-DD> (<Weekday>) at <time> -- i.e. <relative phrase>" (or a
  date-only variant). The AI answer is CORRECT if it matches EITHER the
  absolute side (the date / weekday / clock time) OR the relative side
  (the weekday / "next ..." / "tomorrow" / etc.). It does not have to
  reproduce both halves.
- If the AI answer captures the core point of the gold answer -> CORRECT.
- Otherwise -> WRONG.
- Saying "I don't remember" / "I don't know" / refusing to answer -> WRONG.

Reply with only CORRECT or WRONG."""
    tag = (
        make_judge_tag(story_id, week, qid, suffix="answer")
        if (story_id is not None and week is not None and qid is not None)
        else None
    )
    result = call_judge(prompt, tag=tag, story_id=story_id, week=week, kind="judge")
    return "CORRECT" in (result or "").upper()

# ==================== FEEDBACK ====================
def format_feedback(qa, is_correct, fbmode, character_profiles):
    """Build a single feedback string sent in the user turn AFTER an answer.

    qa: dict with keys 'character', 'gold_answer', 'golden_feedback'
    fbmode: 'with_gold' | 'verbose'
    character_profiles: {char: {'preference_feedback': str, ...}}
    """
    if is_correct:
        return "You are right."
    golden_fb = (qa.get("golden_feedback") or "").strip()
    if not golden_fb:
        # last-resort fallback: spell out the gold answer
        golden_fb = f"the answer is: {qa.get('gold_answer','')}"
    base = f"You are wrong, {golden_fb}."
    if fbmode == "with_gold":
        return base
    if fbmode == "verbose":
        char = qa.get("character", "")
        pref_fb = ""
        if character_profiles and char in character_profiles:
            pref_fb = (character_profiles[char].get("preference_feedback") or "").strip()
        if pref_fb:
            return f"{base} {pref_fb}"
        return base
    raise ValueError(f"unknown fbmode: {fbmode}")

# ==================== STORY HELPERS ====================
def story_data_dir(story_id):
    return os.path.join(DATA_DIR, f"story_{story_id}")

def story_exp_dir(story_id):
    p = os.path.join(EXP_DIR, f"story_{story_id}")
    os.makedirs(p, exist_ok=True)
    return p

def load_story_bundle(story_id, num_weeks):
    """Load weeks / qa / oracle memories / character_profiles for one story.

    The paper schema treats session-level ``golden_memories`` as the single
    oracle source. ``ideal_data`` is derived in-memory from those goldens for
    the ID runner's existing interface; any stale ``ideal_summaries.json`` on
    disk is intentionally ignored.
    """
    sd = story_data_dir(story_id)
    weeks_data = {}
    for w in range(1, num_weeks + 1):
        with open(os.path.join(sd, f"week{w}.json")) as f:
            weeks_data[w] = json.load(f)
    with open(os.path.join(sd, "test_qa.json")) as f:
        qa_data = json.load(f)
    memories = []
    for w, wd in weeks_data.items():
        for sess in wd.get("conversations", []) or []:
            for idx, gm in enumerate(sess.get("golden_memories", []) or []):
                if not isinstance(gm, str) or not gm.strip():
                    continue
                memories.append({
                    "content": gm.strip(),
                    "week": w,
                    "character": sess.get("focal_character", ""),
                    "topic_anchor": sess.get("topic_anchor", ""),
                    "supersedes": None,
                    "source_ref": {
                        "week": w,
                        "session_id": sess.get("session_id", ""),
                        "index": idx,
                    },
                })
    ideal_data = {"story_id": story_id, "total": len(memories), "memories": memories}
    cp_path = os.path.join(sd, "character_profiles.json")
    profiles = {}
    if os.path.exists(cp_path):
        with open(cp_path) as f:
            profiles = json.load(f)
    return weeks_data, qa_data, ideal_data, profiles

def format_dialogue_as_text(messages):
    lines = []
    for m in messages:
        role = "Linchen" if m["role"] == "user" else "AI"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


# ==================== JSONL APPENDER ====================
_JSONL_LOCK = threading.Lock()

def append_jsonl(path, obj):
    """Append a JSON-serialisable object as one line to ``path``.

    Hard-fails (raises) on serialisation/IO error rather than silently dropping
    a record -- the paper experiments rely on these files being lossless.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False)
    with _JSONL_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ==================== EXPERIMENT NAMING ====================
def memory_model_slug(alias=None):
    """Directory/collection label for a memory-model alias.

    Paper experiments intentionally keep the original vendor alias unchanged so
    report/judge filters, CLI arguments and on-disk directories use one name.
    """
    return alias or _MEMORY_MODEL_ALIAS


def qdrant_collection_name(method, fbmode, story_id, *, memory_model_alias=None):
    """``<method>_<memory_model_alias>_<fbmode>_s<story>``

    All four method/model/fbmode/story dimensions are part of the name so that
    parallel experiments cannot accidentally share a vector store.
    """
    alias = memory_model_slug(memory_model_alias)
    return f"{method}_{alias}_{fbmode}_s{story_id}"


# ---- Paper-experiment output layout (exp/paper/<method>/<alias>/<fb>/story_<id>/) ----

PAPER_EXP_ROOT = os.path.join(ADAMEM_ROOT, "exp", "paper")


def paper_story_dir(method, fbmode, story_id, *, memory_model_alias=None,
                    out_root=None):
    """Resolve & create the per-cell output directory for a paper experiment.

    Layout: ``<out_root or exp/paper>/<method>/<memory_model_alias>/<fbmode>/story_<id>/``

    ``method`` is one of ``id|fc|m0|adamem`` (case-insensitive); the directory
    on disk uses lowercase. ``fbmode`` is one of ``with_gold|verbose``.
    """
    root = out_root or PAPER_EXP_ROOT
    alias = memory_model_slug(memory_model_alias)
    p = os.path.join(root, method.lower(), alias, fbmode, f"story_{story_id}")
    os.makedirs(p, exist_ok=True)
    return p


def is_paper_cell_complete(story_dir):
    """A paper cell is complete only when its explicit done marker says so."""
    done_path = os.path.join(story_dir, "done.json")
    if not os.path.exists(done_path):
        return False
    try:
        with open(done_path) as f:
            marker = json.load(f)
    except Exception:
        return False
    if marker.get("status") != "done":
        return False
    for name in marker.get("artifacts", []) or []:
        if not os.path.exists(os.path.join(story_dir, name)):
            return False
    return True


def mark_paper_cell_done(story_dir, *, method, story_id, fbmode,
                         memory_model_alias=None, extra=None):
    """Write an explicit completion marker after all artifacts are flushed."""
    marker = {
        "status": "done",
        "method": method,
        "story_id": story_id,
        "fbmode": fbmode,
        "memory_model": memory_model_alias or get_memory_model_alias(),
        "finished_at": datetime.now().isoformat(),
        "artifacts": [
            "qa_records.json",
            "extracted_memories.jsonl",
            "recall.jsonl",
        ],
    }
    if extra:
        marker.update(extra)
    os.makedirs(story_dir, exist_ok=True)
    with open(os.path.join(story_dir, "done.json"), "w", encoding="utf-8") as f:
        json.dump(marker, f, ensure_ascii=False, indent=2)


# ---- Business tags for client_metadata (must match outline.md naming) ----

def make_qa_tag(story_id, week, qid):
    return f"qa_s{story_id}_w{week}_{qid}"


def make_memory_tag(story_id, week, date_or_kind):
    """``memory_s<id>_w<n>_<date_or_kind>`` -- e.g. ``memory_s1_w1_2015-07-15``."""
    return f"memory_s{story_id}_w{week}_{date_or_kind}"


def make_policy_tag(story_id, week):
    return f"policy_s{story_id}_w{week}"


def make_judge_tag(story_id, week, qid, *, suffix=""):
    base = f"judge_s{story_id}_w{week}_{qid}"
    return f"{base}_{suffix}" if suffix else base


# ==================== STARTUP DIAGNOSTICS ====================
def print_effective_config(extra=None):
    """Pretty-print the runtime config picked up by ``run_*.py``.

    Called at the top of every entrypoint per requirement 9.1 / 8.1.
    """
    rows = [
        ("DATA_GEN_MODEL",   DATA_GEN_MODEL),
        ("QA_ANSWER_MODEL",  QA_ANSWER_MODEL),
        ("MEMORY_MODEL",     f"{MEMORY_MODEL}  (alias={_MEMORY_MODEL_ALIAS})"),
        ("JUDGE_MODEL",      JUDGE_MODEL),
        ("OPENAI_BASE_URL",  OPENAI_BASE_URL or "<MISSING>"),
        ("OPENAI_API_KEY",   "<set>" if OPENAI_API_KEY else "<MISSING>"),
        ("EMBEDDING_BASE_URL", EMBEDDING_BASE or "<MISSING>"),
        ("EMBEDDING_API_KEY", "<set>" if EMBEDDING_API_KEY else "<MISSING>"),
        ("ADAMEM_ROOT",      ADAMEM_ROOT),
        ("DATA_DIR",         DATA_DIR),
        ("EXP_DIR",          EXP_DIR),
        ("MEM0_QDRANT_DIR",  MEM0_QDRANT_DIR),
    ]
    if extra:
        rows.extend(list(extra.items()))
    width = max(len(k) for k, _ in rows)
    print("=" * 8 + " AdaMem effective config " + "=" * 8, flush=True)
    for k, v in rows:
        print(f"  {k.ljust(width)} : {v}", flush=True)
    print("=" * 41, flush=True)


def sanity_check_connectivity(*, require_proxy=True, require_embedding=True,
                              tag="sanity_check"):
    """Smoke-test the chat + embedding endpoints once at startup.

    Raises on hard failure so a misconfigured run dies in the first second
    rather than after hours of useless logs.
    Set ``require_proxy=False`` (etc.) when running unit tests offline.
    """
    if require_proxy:
        # NOTE: JUDGE_MODEL (deepseek-v4-flash) is a reasoning model; with
        # max_tokens=4 the entire budget gets consumed by reasoning_content
        # and message.content stays "" -> false-positive "empty (proxy)".
        # 512 tokens is enough headroom for the model to think briefly and
        # still produce a tiny final "ok" string for non-empty assertion.
        text = make_chat_call(
            JUDGE_MODEL,
            prompt="Reply with just the two letters: ok",
            max_tokens=512, temperature=0, timeout=30,
            bucket="sanity", tag=tag, kind="smoke",
        )
        if not text:
            raise RuntimeError(
                "sanity_check: chat call returned empty; "
                f"check OPENAI_BASE_URL={OPENAI_BASE_URL!r} / OPENAI_API_KEY"
            )
    if require_embedding:
        embs = get_embeddings_batch(["ping"])
        if not embs or len(embs[0]) < 8:
            raise RuntimeError(
                "sanity_check: embedding call returned empty; "
                f"check EMBEDDING_BASE_URL={EMBEDDING_BASE!r} / EMBEDDING_API_KEY"
            )
    log.info("sanity_check_connectivity OK")
