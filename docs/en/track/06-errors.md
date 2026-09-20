# 06 — Errors: breaking gracefully

## `try` / `catch` / `finally`

```
  try {
      risk it here
  } catch error {
      if it falls, it falls here (error holds the message)
  } finally {
      always passes here
  }
```

```tri
try {
    throw "boom";
} catch e {
    print("caught: " + str(e));
} finally {
    print("always runs");
}
```

`▸ output`

```text
caught: boom
always runs
```

`throw` throws any value (text, number, dict).

## `assert`: lock in your certainties

```tri
fn divide(a, b) {
    assert b != 0;
    return a / b;
}

print(divide(10, 2));   // 5.0
// divide(10, 0) aborts with an assertion error
```

`assert` is executable documentation: if the condition fails, the
program stops there and points at the line.

## `match`: choose by value

```tri
let v = 2;

match v {
    case 1 {
        print("one");
    }
    case 2 {
        print("two");
    }
}
// ▸ two
```

Each `case` tests one value; its block runs.

## The emergency trio

```
  something broke? ──► throw "reason"
  might break? ──────► try / catch
  must never? ───────► assert condition
```

✎ try it: `fn root(x)` that throws if `x < 0`, else returns
`math.sqrt(x)` (needs `import math;`).

Next: [07 — modules](07-modules.md).
