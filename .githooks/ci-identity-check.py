"""Refuse a build when a commit misrepresents its author.

Two things fail the check, and nothing else does:

  1. AI attribution.  A commit message carries a trailer that credits an AI tool
     or agent: Co-Authored-By: Claude, Generated with Codebuff, Assisted-by:
     Copilot, the robot emoji, or the word "codebuff".  A trailer crediting a
     person is not an AI attribution and passes.

  2. Malformed identity.  A commit carries this account's name with an email
     that is not one of its emails, or carries a name that is a mangled version
     of this account's name (a truncated or mistyped login, for instance).
     Author and committer are both checked.

Commits written by other people are reported and never fail the check.  Their
history is not ours to rewrite, so a repository that already contains them stays
green while still showing them in the output.

Usage:  python3 ci-identity-check.py [ref ...]

Without refs, every ref in the repository is scanned.  The expected identity is
read at run time and never typed into this file:

  environment  IDENTITY_NAME, IDENTITY_EMAILS
  file         the "identity" file sitting next to this script
  git          git config --global user.name and user.email

Exit status is 1 when a commit fails the check, 0 otherwise.
"""

import difflib
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# A trailer line that credits someone, and the name it credits.
CREDIT_TRAILER = re.compile(
    r"^\s*(?:[-*]\s*)?(?:co-authored-by|authored-by|assisted-by|generated\s+with"
    r"|generated\s+by|written\s+by|with\s+help\s+from)\s*:\s*(?P<who>.+?)\s*$",
    re.IGNORECASE,
)
ROBOT_EMOJI = "\U0001F916"
TOOL_WORD = re.compile(r"\bcodebuff\b", re.IGNORECASE)

# Names that mark a trailer as crediting a tool rather than a person.
AI_NAME = re.compile(
    r"\b(codebuff|claude|anthropic|openai|chatgpt|gpt|copilot|cursor|windsurf"
    r"|aider|devin|gemini|codex|codewhisperer|cody|tabnine|continue\.dev"
    r"|copilot\[bot\]|\[bot\]|bot)\b",
    re.IGNORECASE,
)

# People and bots whose commits are not this account's and are reported only.
THIRD_PARTY_NAMES = {
    "eracle",
    "antonio ercole de luca",
    "elton minetto",
    "joseph",
    "josephlimtech",
    "stan",
    "xgiammyx",
    "ahmad hassan",
}
THIRD_PARTY_EMAILS = {
    "eracle@users.noreply.github.com",
    "josephlimtech@users.noreply.github.com",
    "xgiammyx@users.noreply.github.com",
    "ahmadhassan@users.noreply.github.com",
    "a.hassan@b-ted.com",
}
BOT_EMAIL = re.compile(r"^\d+\+(?P<login>[^@]+)@users\.noreply\.github\.com$")
BARE_NOREPLY = re.compile(r"^(?P<login>[^@+]+)@users\.noreply\.github\.com$")

RECORD_SEP = "\x1e"
FIELD_SEP = "\x1f"


def read_identity():
    """Return (expected_name, accepted_emails) from wherever it is available."""
    name = os.environ.get("IDENTITY_NAME", "").strip()
    emails = os.environ.get("IDENTITY_EMAILS", "").strip()
    if name and emails:
        return name, emails.split()

    path = os.path.join(HERE, "identity")
    if os.path.isfile(path):
        scope = {}
        with open(path, encoding="utf-8") as fh:
            exec(compile(fh.read(), path, "exec"), scope)  # noqa: S102
        name = scope.get("expected_name", name)
        emails = scope.get("accepted_emails", emails)

    if not name:
        name = _git("config", "--global", "user.name").strip()
    if not emails:
        emails = _git("config", "--global", "user.email").strip()

    if not name or not emails:
        sys.stderr.write(
            "identity check: no identity available (set IDENTITY_NAME and"
            " IDENTITY_EMAILS, add an identity file next to this script, or set"
            " git config --global user.name and user.email)\n"
        )
        sys.exit(2)
    return name, emails.split()


def _git(*args):
    proc = subprocess.run(  # noqa: S603
        ["git"] + list(args),
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.stdout if proc.returncode == 0 else ""


def commits(refs):
    """Yield (hash, author_name, author_email, committer_name, committer_email, body)."""
    fmt = FIELD_SEP.join(["%H", "%an", "%ae", "%cn", "%ce", "%B"]) + RECORD_SEP
    out = _git("log", "--no-renames", "--format=" + fmt, *refs)
    if not out.strip():
        return
    for record in out.split(RECORD_SEP):
        record = record.strip("\n")
        if not record.strip():
            continue
        parts = record.split(FIELD_SEP)
        while len(parts) < 6:
            parts.append("")
        yield parts[0], parts[1], parts[2], parts[3], parts[4], FIELD_SEP.join(parts[5:])


def ai_attribution(body):
    """Return the offending line when a message credits a tool instead of a person."""
    for line in body.splitlines():
        if ROBOT_EMOJI in line:
            return line.strip()[:120]
        if TOOL_WORD.search(line):
            return line.strip()[:120]
        trailer = CREDIT_TRAILER.match(line)
        if trailer and AI_NAME.search(trailer.group("who")):
            return line.strip()[:120]
    return None


def is_third_party(name, email):
    """A commit this account did not write and is not a mangled version of."""
    if name == _EXPECTED_NAME:
        return False  # this account, whichever of its emails it used
    if name.strip().lower() in THIRD_PARTY_NAMES:
        return True
    if email.lower() in THIRD_PARTY_EMAILS:
        return True
    # A GitHub no-reply address belongs to whoever the login names, and a login
    # that is not this account's is another person's address.
    if BOT_EMAIL.match(email) or BARE_NOREPLY.match(email):
        return True
    return False


def is_mangled(name):
    """A near miss on this account's name: a truncated or mistyped login."""
    got = name.strip().lower()
    want = _EXPECTED_NAME.lower()
    if got == want:
        return False
    if want.startswith(got) or got.startswith(want):
        return True
    return difflib.SequenceMatcher(None, got, want).ratio() >= 0.85


_EXPECTED_NAME = ""
_ACCEPTED_EMAILS = ()


def main(argv):
    # Set once so is_mangled and is_third_party can see them without a re-read.
    global _EXPECTED_NAME, _ACCEPTED_EMAILS
    _EXPECTED_NAME, emails = read_identity()
    _ACCEPTED_EMAILS = tuple(e.lower() for e in emails)

    refs = argv[1:] or ["--all"]
    failures = []
    notes = []
    total = 0

    for sha, an, ae, cn, ce, body in commits(refs):
        total += 1
        short = sha[:8]
        subject = body.splitlines()[0][:60] if body.strip() else ""
        trailer = ai_attribution(body)
        own_author = an == _EXPECTED_NAME
        mangled_author = own_author is False and is_mangled(an)
        third = is_third_party(an, ae)

        if third:
            if trailer:
                notes.append(
                    "third-party commit %s %s <%s> carries an AI trailer: %s"
                    " (not this account, left alone)" % (short, an, ae, trailer)
                )
            continue

        if mangled_author:
            failures.append(
                "commit %s author %r is a malformed version of %r: %s"
                % (short, an, _EXPECTED_NAME, subject)
            )
        elif own_author and ae.lower() not in _ACCEPTED_EMAILS:
            failures.append(
                "commit %s author %s <%s> uses an email that is not this"
                " account's: %s" % (short, an, ae, subject)
            )
        elif not own_author:
            notes.append(
                "commit %s author %s <%s> is not this account: %s"
                % (short, an, ae, subject)
            )

        if cn != _EXPECTED_NAME:
            if is_mangled(cn):
                failures.append(
                    "commit %s committer %r is a malformed version of %r: %s"
                    % (short, cn, _EXPECTED_NAME, subject)
                )
            elif not is_third_party(cn, ce):
                notes.append(
                    "commit %s committer %s <%s> is not this account: %s"
                    % (short, cn, ce, subject)
                )
        elif ce.lower() not in _ACCEPTED_EMAILS:
            failures.append(
                "commit %s committer %s <%s> uses an email that is not this"
                " account's: %s" % (short, cn, ce, subject)
            )

        if trailer and not third:
            failures.append(
                "commit %s credits an AI tool in its message: %s (%s)"
                % (short, trailer, subject)
            )

    print("identity check: %d commits scanned in %s" % (total, " ".join(refs)))
    for note in notes:
        print("  note    %s" % note)
    for failure in failures:
        print("  FAIL    %s" % failure)
    print(
        "identity check: %d failure%s, %d note%s"
        % (
            len(failures),
            "" if len(failures) == 1 else "s",
            len(notes),
            "" if len(notes) == 1 else "s",
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
