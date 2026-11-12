"""Strict pairing for historical generation records; no model dependencies."""
import json
import math
from pathlib import Path


def select_baseline(runs):
    """Require one explicit beta-zero configuration, independent of filenames.

    Multiple seeds/baselines need an explicit comparison design; selecting the
    first sorted filename would silently decide that scientific question.
    """
    baselines = []
    for name, result in runs.items():
        beta = result.get('config', {}).get('beta')
        if type(beta) not in (int, float) or not math.isfinite(beta) or beta < 0:
            raise ValueError(f'{name}: beta must be a finite nonnegative number')
        if beta == 0:
            baselines.append(name)
    if len(baselines) != 1:
        raise ValueError('require exactly one beta-zero baseline; '
                         f'found {len(baselines)}; specify a comparison design')
    return baselines[0]


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('duplicate JSON key')
        obj[key] = value
    return obj


def _reject_constant(value):
    raise ValueError('non-finite JSON constant')


def load_generations(path):
    """Load nonempty JSONL with unique exact prompts and finite numeric RM.

    The historical writer persists prompt text, not stable sample IDs. Exact
    text is therefore the only available pairing key. Repeated prompts are
    ambiguous and fail closed; do not silently normalize text or invent IDs.
    """
    records = []
    seen = set()
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                item = json.loads(line, object_pairs_hook=_unique_object,
                                  parse_constant=_reject_constant)
                if not isinstance(item, dict):
                    raise ValueError('record must be an object')
                prompt = item.get('prompt')
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError('prompt must be a nonblank string')
                if prompt in seen:
                    raise ValueError('ambiguous duplicate prompt')
                score = item.get('rm')
                if type(score) not in (int, float) or not math.isfinite(score):
                    raise ValueError('rm must be a finite number, not bool/string')
                seen.add(prompt)
                records.append(item)
            except (ValueError, OverflowError) as exc:
                raise ValueError(f'{path}: line {line_number}: {exc}') from exc
    if not records:
        raise ValueError(f'{path}: no generation records')
    return records


def winrate_vs_base(candidate, baseline):
    """Compare complete, already validated sets by exact prompt, not row order.

    Strict wins use >, preserving the historical convention that ties are not
    wins. This is a descriptive proportion, not a confidence/quality claim.
    """
    def index(records):
        result = {}
        for item in records:
            prompt, score = item['prompt'], item['rm']
            if (not isinstance(prompt, str) or not prompt.strip()
                    or prompt in result or type(score) not in (int, float)
                    or not math.isfinite(score)):
                raise ValueError('invalid/ambiguous generation record')
            result[prompt] = score
        if not result:
            raise ValueError('no generation records')
        return result

    mine, base = index(candidate), index(baseline)
    if mine.keys() != base.keys():
        raise ValueError('prompt sets differ: '
                         f'{len(base.keys() - mine.keys())} missing, '
                         f'{len(mine.keys() - base.keys())} extra')
    return sum(mine[p] > base[p] for p in base) / len(base)
