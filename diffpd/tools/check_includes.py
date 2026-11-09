#!/usr/bin/env python3
"""Fails if a module includes a module it is not allowed to use (the rule of plans/ARCHITECTURE_REFACTOR_PLAN.md).

Every module shares the project root as its include path, so CMake cannot stop a wrong #include "app/..." in collision/.
This reads the quoted #include lines instead. Run from anywhere: python tools/check_includes.py
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# module -> modules it may include (besides itself)
ALLOWED = {
    'core':      set(),
    'energy':    {'core'},
    'meshgen':   {'core'},
    'collision': {'core'},
    'optim':     {'core'},
    'cloth':     {'core', 'energy', 'meshgen', 'collision'},
    'solver':    {'core', 'energy', 'collision', 'cloth'},
    'app':       {'core', 'energy', 'meshgen', 'collision', 'cloth', 'solver'},
    'viewer':    {'core', 'energy', 'meshgen', 'collision', 'cloth', 'solver', 'app'},
    'driver':    {'core', 'energy', 'meshgen', 'collision', 'cloth', 'solver', 'app', 'viewer', 'optim'},
    'top':       None,  # diffpd.cpp, diffpd_checks.h: may include anything
}
FOLDERS = {'core', 'energy', 'meshgen', 'collision', 'optim', 'cloth', 'solver', 'app', 'driver'}
VIEWER_FILES = {'diffpd_viewer.h', 'diffpd_viewer.cpp', 'viewer_theme.h'}


def module_of(rel):
    first = rel.parts[0]
    if len(rel.parts) > 1 and first in FOLDERS:
        return first
    if rel.name in VIEWER_FILES:
        return 'viewer'
    return 'top'


def main():
    bad = 0
    files = [p for d in sorted(FOLDERS) for p in (ROOT / d).rglob('*') if p.suffix in ('.h', '.cpp')]
    files += [ROOT / n for n in VIEWER_FILES | {'diffpd.cpp', 'diffpd_checks.h'}]
    for path in files:
        rel = path.relative_to(ROOT)
        mod = module_of(rel)
        for n, line in enumerate(path.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
            m = re.match(r'\s*#include\s+"([^"]+)"', line)
            if not m:
                continue
            target = pathlib.PurePosixPath(m.group(1))
            if not (ROOT / target).exists():
                continue  # a vendored header (raylib, raygui, ...)
            dep = module_of(pathlib.Path(target))
            if (len(rel.parts) > 2 and rel.parts[0] == 'solver' and rel.parts[1] in ('ncp', 'impulse') and len(target.parts) > 2
                    and target.parts[0] == 'solver' and target.parts[1] in ('ncp', 'impulse') and target.parts[1] != rel.parts[1]):
                print(f'{rel}:{n}: solver/{rel.parts[1]} must not include solver/{target.parts[1]} ("{m.group(1)}")')
                bad += 1
            if mod == 'top' or dep == mod or dep in ALLOWED[mod]:
                continue
            print(f'{rel}:{n}: {mod} must not include {dep} ("{m.group(1)}")')
            bad += 1
    print('include rule: ' + ('OK' if not bad else f'{bad} violation(s)'))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
