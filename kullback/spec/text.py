"""The word-level text rules the Builder and the Spec both read (moved from builder/cluster.py and intent.py).

Here so the Spec can tokenise and normalise without importing the Builder; cluster.py and intent.py
re-export every name they moved.
"""

from __future__ import annotations

import re

APOSTROPHE_RE = re.compile(r"[\u0027\u2019\u02bc]")  # "don't" is one word, not "don" and "t"
STOPWORDS = frozenset(
    """
    a an the and or but so then of to for in on at by from with without about into as is was were be been being
    are am it its this that these those there here i me my we our you your he she they them their his her
    do does did done have has had can could would should will shall may might must not no yes if when while
    because please thanks thank hi hello ok okay just also very really any some all
    dont doesnt didnt cant cannot couldnt wouldnt shouldnt wasnt werent isnt arent wont havent hasnt hadnt
    im ive ill id youre youve youll youd hes shes theyre theyve thats whats lets weve wed
    """.split()
)

# One token: letters and digits, with a dot, comma or colon kept only between two digits, so "$17.99"
# is "17.99", "12:30" is "12:30" and "1,000" is "1,000", while "order 2.Next" is still two tokens.
# An underscore is not part of a token, so credit_card_1234 is credit, card, 1234 and a phrase saying
# "credit card" reaches it; the phrase "credit_card_1234" splits the same way, so both forms match.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:(?<=[0-9])[.,:][0-9]+)*")


def _undouble(stem: str) -> str:
    """"cancell" is "cancel" and "shipp" is "ship"; a doubled consonant is spelling, not a word."""
    doubled = len(stem) > 4 and stem[-1] == stem[-2] and stem[-1] not in "aeiou"
    return stem[:-1] if doubled else stem


def _strip_suffix(word: str) -> str:
    """One inflection off the end of a word, and only where four letters are left standing."""
    for suffix, stem in (("ies", word[:-3] + "y"), ("es", word[:-2]), ("s", word[:-1]),
                         ("ing", _undouble(word[:-3])), ("ed", _undouble(word[:-2])), ("e", word[:-1])):
        if not word.endswith(suffix) or len(word) - len(suffix) < 4:
            continue
        if suffix == "s" and word.endswith("ss"):
            continue  # "address" is not the plural of "addres"
        return stem
    return word


def normalise(word: str) -> str:
    """One spelling for a word's simple inflections, so "earbud" and "earbuds" are the same word.

    A suffix stripper, not a lemmatiser: plurals, past tense and -ing, stripped until nothing more
    comes off, so "addresses", "address" and "changes", "changed", "change" each land on one stem. It
    never touches a word carrying a digit, so an order id and a price keep their spelling. A real
    lemmatiser (spaCy, nltk) would beat it on irregulars, and is not worth a model download and a new
    dependency in the Builder's hot path for the handful of words a customer-service line uses.
    """
    if not word.isalpha():
        return word
    while True:
        stem = _strip_suffix(word)
        if stem == word:
            return word
        word = stem
