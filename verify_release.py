#!/usr/bin/env python3
"""Check file integrity for this extracted research release; standard library only."""
from pathlib import Path
import hashlib
import sys

base=Path(__file__).resolve().parent
listing=base/'SHA256SUMS.txt'
failed=[]
seen=set()
for line in listing.read_text(encoding='utf-8').splitlines():
    if not line.strip():continue
    sha,relative=line.split('  ',1)
    path=(base/relative).resolve()
    if not path.is_relative_to(base):
        failed.append(('unsafe path',relative));continue
    if relative in seen:
        failed.append(('duplicate',relative));continue
    seen.add(relative)
    if not path.is_file():
        failed.append(('missing',relative));continue
    h=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1<<20),b''):
            h.update(block)
    if h.hexdigest()!=sha:
        failed.append(('hash mismatch',relative))

actual={str(p.relative_to(base)).replace('\\','/') for p in base.rglob('*') if p.is_file() and not any(n in {'.git', '.vscode', '__pycache__', '.pytest_cache', '.venv', 'reproduced_results', 'validation_reports'} for n in p.relative_to(base).parts) and p.name!='SHA256SUMS.txt'}
missing_from_listing=actual-seen
if missing_from_listing:
    failed.extend(('not in manifest',x) for x in sorted(missing_from_listing))
print(f'Integrity: {len(seen)} records; {len(failed)} failures')
for issue in failed[:25]:print(*issue)
sys.exit(1 if failed else 0)
