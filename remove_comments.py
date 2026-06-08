import re
import sys
from pathlib import Path

def remove_python_comments(source: str) -> str:
    result = []
    in_string = False
    string_char = None
    i = 0

    while i < len(source):
        c = source[i]

        if in_string:
            result.append(c)
            if c == '\\':
                i += 1
                if i < len(source):
                    result.append(source[i])
            elif string_char in ('"""', "'''"):
                if source[i:i+3] == string_char:
                    result.append(source[i+1])
                    result.append(source[i+2])
                    i += 2
                    in_string = False
                    string_char = None
            elif c == string_char:
                in_string = False
                string_char = None
            i += 1
            continue

        if source[i:i+3] in ('"""', "'''"):
            triple = source[i:i+3]
            in_string = True
            string_char = triple
            result.append(triple)
            i += 3
            continue

        if c in ('"', "'"):
            in_string = True
            string_char = c
            result.append(c)
            i += 1
            continue

        if c == '#':
            while i < len(source) and source[i] != '\n':
                i += 1
            continue

        result.append(c)
        i += 1

    return ''.join(result)

def clean_blank_lines(source: str) -> str:
    lines = source.splitlines()
    cleaned = []
    prev_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank
    return '\n'.join(cleaned).strip() + '\n'

def process_file(path: Path) -> bool:
    try:
        original = path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"skip {path}: {e}")
        return False

    cleaned = remove_python_comments(original)
    cleaned = clean_blank_lines(cleaned)

    if cleaned != original:
        path.write_text(cleaned, encoding='utf-8')
        print(f"cleaned {path}")
        return True

    return False

def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    files = list(root.rglob('*.py'))
    changed = 0
    for f in files:
        if any(part.startswith('.') for part in f.parts):
            continue
        if process_file(f):
            changed += 1
    print(f"\n{changed}/{len(files)} files changed")

if __name__ == '__main__':
    main()
