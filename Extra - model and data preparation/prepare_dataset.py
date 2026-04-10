import json
import re

INPUT  = r'e:\indic\all_songs.json'
OUTPUT = r'e:\indic\training_data.jsonl'


def normalize(s):
    """Normalize Hindi text for consistent comparison."""
    s = s.strip()
    s = s.replace('\u0901', '\u0902')  # chandrabindu → anuswar
    return s


def get_rhyme_key(line, n=3):
    """Last n chars after stripping punctuation — used as rhyme fingerprint."""
    clean = re.sub(r'[।,\.!?\s]+$', '', normalize(line))
    chars = list(clean)
    return ''.join(chars[-n:]) if len(chars) >= n else ''.join(chars)


def assign_rhyme_groups(lines):
    """
    Assign A/B/C rhyme groups to each line in a verse.
    Tries 3-char match first, falls back to 2-char.
    Returns (groups_list, scheme_string).
    """
    if not lines:
        return [], 'free'

    rhyme_map = {}      # key -> label
    groups    = []
    next_idx  = 0
    labels    = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    for line in lines:
        if not line.strip():
            groups.append('A')
            continue

        key3 = get_rhyme_key(line, 3)
        key2 = get_rhyme_key(line, 2)

        if key3 in rhyme_map:
            groups.append(rhyme_map[key3])
        elif key2 in rhyme_map:
            # 2-char match — same rhyme group, register 3-char alias too
            rhyme_map[key3] = rhyme_map[key2]
            groups.append(rhyme_map[key3])
        else:
            label = labels[min(next_idx, len(labels) - 1)]
            rhyme_map[key3] = label
            rhyme_map[key2] = label   # so future lines with same 2-char ending match
            next_idx += 1
            groups.append(label)

    scheme = ''.join(groups)
    return groups, scheme


def get_prev_end(line, n=4):
    """Last n chars of line (after stripping punctuation) for the prev_end field."""
    clean = re.sub(r'[।,\.!?\s]+$', '', line.strip())
    chars = list(clean)
    return ''.join(chars[-n:]) if chars else ''


def detect_verse_type(verses, idx):
    """
    Chorus  : first line of this verse appears as first line in at least one other verse.
    Bridge  : very short block (≤2 lines) sandwiched between other blocks.
    Verse   : everything else.
    """
    current_lines = [normalize(l) for l in verses[idx] if l.strip()]
    if not current_lines:
        return 'verse'

    first_line = current_lines[0]
    matches = sum(
        1 for v in verses
        if [normalize(l) for l in v if l.strip()][:1] == [first_line]
    )
    if matches > 1:
        return 'chorus'

    non_empty = [l for l in verses[idx] if l.strip()]
    if len(non_empty) <= 2 and 0 < idx < len(verses) - 1:
        return 'bridge'

    return 'verse'


# ── Main ────────────────────────────────────────────────────────────────────

with open(INPUT, 'r', encoding='utf-8') as f:
    songs = json.load(f)

records = []
empty_verses = 0

for song_idx, song in enumerate(songs):
    song_id = song['song_id']
    genre   = song['genre']
    verses  = song.get('verses', [])

    print(f"[{song_idx + 1:>3}/{len(songs)}] {song_id} ({genre})  —  {len(verses)} verses")

    for verse_idx, verse in enumerate(verses):
        lines = [l.strip() for l in verse if l.strip()]
        if not lines:
            empty_verses += 1
            continue

        verse_type              = detect_verse_type(verses, verse_idx)
        rhyme_groups, rhyme_scheme = assign_rhyme_groups(lines)

        prev_end = 'none'

        for line_idx, line in enumerate(lines):
            rhyme_group = rhyme_groups[line_idx] if line_idx < len(rhyme_groups) else 'A'

            record = {
                'song_id'      : song_id,
                'genre'        : genre,
                'verse_idx'    : verse_idx,
                'verse_type'   : verse_type,
                'rhyme_scheme' : rhyme_scheme,
                'rhyme_group'  : rhyme_group,
                'prev_end'     : prev_end,
                'literal'      : '',     # filled by add_literals.py
                'output'       : line,
            }
            records.append(record)
            prev_end = get_prev_end(line)

with open(OUTPUT, 'w', encoding='utf-8') as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f'\nDone!  {len(records)} training examples  →  {OUTPUT}')
print(f'Empty verses skipped: {empty_verses}')
print('\nNext step: run add_literals.py to fill in the "literal" field using IndicTrans2.')
