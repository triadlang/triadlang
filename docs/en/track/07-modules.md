# 07 — Modules: importing the world

## The three shapes

```tri
import math;                    // whole module
import math as m;               // with a nickname
from math import sqrt, pi;      // only what you need
```

## Essential stdlib (always available)

```
  math         sqrt sin cos log exp floor ceil pi e min max
  random       random randint choice seed shuffle uniform
  string       split join replace lower upper strip
  json         parse stringify
  fs           read_text write_text exists listdir join tempdir
  time         now sleep
  collections  len range enumerate sorted zip map filter…
  datetime     dates and times
  regex        regular expressions
  plot         charts (triad.plot is the same)
  io           print input
```

One example together:

```tri
import math;
import random;

random.seed(0);
print(math.sqrt(25.0));       // 5.0
print(random.randint(1, 10)); // 7 (with seed 0)
```

## Files (`fs`)

```tri
import fs;

let path = fs.join(fs.tempdir(), "hi.txt");
fs.write_text(path, "Hello file!");
print(fs.read_text(path));   // Hello file!
print(fs.exists(path));      // true
```

## Your own modules

File `util.tri`:

```tri
fn double(n) {
    return n * 2;
}
```

File `app.tri`, in the same folder:

```tri
import util;

print(util.double(21));   // 42
```

## Packages (bigger projects)

```sh
./triad init myapp        # create project
./triad install           # install dependencies
./triad publish           # publish to the local registry
./triad list              # list installed
```

Dependencies live in `triad_modules/`.

✎ try it: a `hello.tri` module with `fn hi(name)` and a program
that imports and uses it.

Next: [08 — arrays](08-arrays.md).
