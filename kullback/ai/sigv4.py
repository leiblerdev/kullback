"""AWS Signature Version 4 for one HTTP request, from the standard library alone.

A provider behind AWS (Bedrock's Messages endpoint) authenticates a request by signing it: the
method, the path, the query, the headers and a hash of the exact body bytes go into a canonical
request, that is hashed into a string to sign, and a key derived from the secret, the date, the
region and the service signs it. The ai package speaks raw HTTP (it mirrors tau_ai) and takes no
SDK, so the signer is written here, following the published algorithm
(docs.aws.amazon.com/IAM/latest/UserGuide/create-signed-request.html) and botocore's SigV4Auth,
which is what the anthropic SDK's Bedrock client calls.

`now` is an argument so a test can sign byte for byte against a known answer.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
from typing import Mapping, Optional
from urllib.parse import parse_qsl, quote, urlsplit

ALGORITHM = "AWS4-HMAC-SHA256"
# Headers a proxy may add, drop or rewrite on the way, so signing them would make a valid request
# fail. botocore leaves out the same ones; `authorization` is the signature itself.
UNSIGNED_HEADERS = frozenset(("authorization", "connection", "expect", "user-agent", "x-amzn-trace-id"))


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac(key: bytes, text: str) -> bytes:
    return hmac.new(key, text.encode("utf-8"), hashlib.sha256).digest()


def _canonical_path(path: str) -> str:
    # Every service but S3 signs the path encoded once more on top of how it was sent.
    return quote(path or "/", safe="/~")


def _canonical_query(query: str) -> str:
    pairs = parse_qsl(query, keep_blank_values=True)
    return "&".join(f"{quote(k, safe='-_.~')}={quote(v, safe='-_.~')}" for k, v in sorted(pairs))


def _canonical_headers(headers: Mapping[str, str]) -> tuple[str, str]:
    """The canonical header block and the signed-header list, lowercase names, trimmed values."""
    folded: dict[str, str] = {}
    for name, value in headers.items():
        key = name.lower().strip()
        if key in UNSIGNED_HEADERS:
            continue
        folded[key] = " ".join(str(value).split())
    names = sorted(folded)
    block = "".join(f"{name}:{folded[name]}\n" for name in names)
    return block, ";".join(names)


def canonical_request(method: str, url: str, headers: Mapping[str, str], body: bytes) -> tuple[str, str]:
    """The canonical request AWS hashes, and the signed-header list it names.

    `headers` must already hold `host` and `x-amz-date`; `sign` below adds them.
    """
    parts = urlsplit(url)
    block, signed = _canonical_headers(headers)
    text = "\n".join((
        method.upper(),
        _canonical_path(parts.path),
        _canonical_query(parts.query),
        block,
        signed,
        _sha256_hex(body or b""),
    ))
    return text, signed


def signing_key(secret_key: str, date: str, region: str, service: str) -> bytes:
    """The key of the day: the secret, narrowed by date, region, service and the fixed terminator."""
    key = _hmac(f"AWS4{secret_key}".encode("utf-8"), date)
    key = _hmac(key, region)
    key = _hmac(key, service)
    return _hmac(key, "aws4_request")


def sign(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: Optional[bytes],
    *,
    access_key: str,
    secret_key: str,
    session_token: Optional[str] = None,
    region: str,
    service: str,
    now: Optional[_dt.datetime] = None,
) -> dict[str, str]:
    """The headers to send: the ones given, plus host, x-amz-date, the session token and authorization.

    The body is signed as the bytes given, so the caller sends exactly these bytes and nothing it
    re-serialises afterwards.
    """
    moment = (now or _dt.datetime.now(_dt.timezone.utc)).astimezone(_dt.timezone.utc)
    amz_date = moment.strftime("%Y%m%dT%H%M%SZ")
    date = amz_date[:8]
    out = {name.lower(): value for name, value in headers.items()}
    out.setdefault("host", urlsplit(url).netloc)
    out["x-amz-date"] = amz_date
    if session_token:
        out["x-amz-security-token"] = session_token
    request, signed = canonical_request(method, url, out, body or b"")
    scope = f"{date}/{region}/{service}/aws4_request"
    to_sign = "\n".join((ALGORITHM, amz_date, scope, _sha256_hex(request.encode("utf-8"))))
    signature = hmac.new(signing_key(secret_key, date, region, service), to_sign.encode("utf-8"),
                         hashlib.sha256).hexdigest()
    out["authorization"] = (f"{ALGORITHM} Credential={access_key}/{scope}, "
                            f"SignedHeaders={signed}, Signature={signature}")
    return out
