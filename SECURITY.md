# Security policy

`purgedcv` is a pure-Python library for cross-validation and backtest
statistics. It does not run a server, open network connections, execute
untrusted input, or handle credentials, so its attack surface is small.
The most likely security-relevant issue is a dependency advisory or a bug
that lets crafted input cause unexpected resource use.

## Supported versions

Fixes are released against the latest published version on PyPI. The
project is pre-1.0, so there is no long-term support branch: please
upgrade to the newest release before reporting an issue.

| Version | Supported |
| ------- | --------- |
| latest release on PyPI | yes |
| older releases | no |

## Reporting a vulnerability

Please report suspected vulnerabilities privately rather than opening a
public issue. Two ways:

- Use GitHub's **Report a vulnerability** button under the repository's
  **Security** tab (private advisory).
- Or email **elazarev@gmail.com** with a description and, if possible, a
  minimal reproducer.

You can expect an acknowledgement within about a week. If the report is
confirmed, a fix and a new release follow, and you will be credited in the
release notes unless you ask otherwise. If a report is declined, you will
get an explanation of why.

## Dependencies

Runtime dependencies are pinned to lower bounds in `pyproject.toml` and
kept current. If you find that `purgedcv` pulls in a dependency version
with a known advisory, that is in scope: please report it the same way.
