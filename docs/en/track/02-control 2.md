# 02 — Control: decisions and repetition

## `if` / `elif` / `else`

```tri
let grade = 7;

if grade >= 7 {
    print("pass");
} elif grade >= 4 {
    print("retake");
} else {
    print("fail");
}
```

`▸ output`: `pass`

Comparisons: `==` `!=` `<` `<=` `>` `>=`.
Logic: `and`, `or`, `not`.

```tri
let age = 20;
let has_invite = true;

if age >= 18 and has_invite {
    print("come in");
}

if not false {
    print("true");
}
```

## `for`: two shapes

```tri
// 1. counting
for i in range(3) {
    print(i);
}
// ▸ 0 1 2

// 2. walking
for name in ["ana", "bia"] {
    print("hi " + name);
}
// ▸ hi ana / hi bia
```

`range(5)` → 0..4. `range(2, 8)` → 2..7.

## `while`, `break`, `continue`

```tri
let x = 0;
while x < 10 {
    x = x + 1;
    if x == 3 {
        continue;   // skips 3
    }
    if x == 6 {
        break;      // stops at 6
    }
    print(x);
}
// ▸ 1 2 4 5
```

## f-strings: text with math inside

```tri
let total = 250;
print(f"sum 0..99 = {total}");
print(f"double = {total * 2}");
```

`▸ output`

```text
sum 0..99 = 250
double = 500
```

Everything inside `{ }` is computed on the spot.

✎ try it: print the 7 times table with `for` + f-string.

Next: [03 — functions](03-functions.md).
