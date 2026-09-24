"""SigV4 against the published AWS test suite, byte for byte.

The vectors are AWS's own (github.com/awslabs/aws-c-auth, tests/aws-signing-test-suite/v4): the
example credentials, region us-east-1, service "service", 2015-08-30T12:36:00Z. Each expected
canonical request and signature is copied from the suite's header-canonical-request.txt and
header-signature.txt.
"""

from __future__ import annotations

import datetime as dt

from kullback.ai import sigv4

NOW = dt.datetime(2015, 8, 30, 12, 36, tzinfo=dt.timezone.utc)
KEYS = {"access_key": "AKIDEXAMPLE", "secret_key": "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
        "region": "us-east-1", "service": "service", "now": NOW}
EMPTY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
TOKEN = ("AQoDYXdzEPT//////////wEXAMPLEtc764bNrC9SAPBSM22wDOk4x4HIZ8j4FZTwdQWLWsKWHGBuFqwAeMicRXmxfpSP"
         "fIeoIYRqTflfKD8YUuwthAx7mSEI/qkPpKPi/kMcGdQrmGdeehM4IC1NtBmUpp2wUE8phUZampKsburEDy0KPkyQDYwT7"
         "WZ0wq5VSXDvp75YU9HFvlRd8Tx6q6fE8YQcHNVXAkiY9q6d+xo0rKwT38xVqr7ZD0u0iPPkUL64lIZbqBAz+scqKmlz"
         "m8FDrypNC9Yjc8fPOLn9FX9KSYvKTr4rvx3iSIlTJabIQwj2ICCR/oLxBA==")


def test_a_post_with_no_body_signs_host_and_date_as_the_suite_says():
    """post-vanilla."""
    headers = sigv4.sign("POST", "https://example.amazonaws.com/", {}, b"", **KEYS)
    canonical, signed = sigv4.canonical_request("POST", "https://example.amazonaws.com/",
                                                {k: v for k, v in headers.items() if k != "authorization"},
                                                b"")
    assert canonical == ("POST\n/\n\nhost:example.amazonaws.com\nx-amz-date:20150830T123600Z\n\n"
                         f"host;x-amz-date\n{EMPTY_SHA}")
    assert headers["authorization"] == (
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, "
        "SignedHeaders=host;x-amz-date, "
        "Signature=5da7c1a2acd57cee7505fc6676e4e544621c30862966e37dddb68e92efbe5d6b")


def test_a_post_with_a_body_signs_the_hash_of_the_exact_bytes():
    """post-x-www-form-urlencoded: the body hash is the last line of the canonical request."""
    given = {"Content-Type": "application/x-www-form-urlencoded", "Content-Length": "13",
             "x-amz-content-sha256": "9095672bbd1f56dfc5b65f3e153adc8731a4a654192329106275f4c7b24d0b6e"}
    headers = sigv4.sign("POST", "https://example.amazonaws.com/", given, b"Param1=value1", **KEYS)
    assert headers["authorization"].endswith(
        "SignedHeaders=content-length;content-type;host;x-amz-content-sha256;x-amz-date, "
        "Signature=d3875051da38690788ef43de4db0d8f280229d82040bfac253562e56c3f20e0b")
    other = sigv4.sign("POST", "https://example.amazonaws.com/", given, b"Param1=value2", **KEYS)
    assert other["authorization"] != headers["authorization"]


def test_a_session_token_is_sent_and_signed():
    """post-sts-header-before: temporary keys carry their token as a signed header."""
    headers = sigv4.sign("POST", "https://example.amazonaws.com/", {}, b"", session_token=TOKEN, **KEYS)
    assert headers["x-amz-security-token"] == TOKEN
    assert "SignedHeaders=host;x-amz-date;x-amz-security-token," in headers["authorization"]
    assert headers["authorization"].endswith(
        "Signature=85d96828115b5dc0cfc3bd16ad9e210dd772bbebba041836c64533a82be05ead")


def test_headers_a_proxy_may_rewrite_are_sent_but_not_signed():
    headers = sigv4.sign("POST", "https://example.amazonaws.com/", {"User-Agent": "x", "Connection": "close"},
                         b"", **KEYS)
    assert "SignedHeaders=host;x-amz-date," in headers["authorization"]
    assert headers["user-agent"] == "x"
