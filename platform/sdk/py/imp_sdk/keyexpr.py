"""Key-expression builders for the imp namespace (spec §6).

Mirrors crates/core/src/keyexpr.rs so Python HAL drivers, modules, services and
jobs address the exact same keys as the Rust core.
"""

ROOT = "imp"


def hal(station: str, device: str, signal: str) -> str:
    return f"{ROOT}/{station}/hal/{device}/{signal}"


def perc(station: str, session: str, signal: str) -> str:
    return f"{ROOT}/{station}/perc/{session}/{signal}"


def motion(station: str, plan: str, signal: str) -> str:
    return f"{ROOT}/{station}/motion/{plan}/{signal}"


def tf(station: str) -> str:
    return f"{ROOT}/{station}/tf"


def svc(station: str, service: str) -> str:
    return f"{ROOT}/{station}/svc/{service}"


def ctrl(station: str, node: str, verb: str) -> str:
    return f"{ROOT}/{station}/ctrl/{node}/{verb}"


def all_perc(station: str) -> str:
    return f"{ROOT}/{station}/perc/**"


def hal_any_device(station: str, signal: str) -> str:
    return f"{ROOT}/{station}/hal/*/{signal}"
