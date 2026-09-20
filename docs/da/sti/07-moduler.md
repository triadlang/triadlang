# 07 — Moduler: importer verden

## De tre former

```tri
import math;                    // hele modulet
import math as m;               // med kælenavn
from math import sqrt, pi;      // kun det nødvendige
```

## Kerne-stdlib (altid inde)

```
  math         sqrt sin cos log exp floor ceil pi e min max
  random       random randint choice seed shuffle uniform
  string       split join replace lower upper strip
  json         parse stringify
  fs           read_text write_text exists listdir join tempdir
  time         now sleep
  collections  len range enumerate sorted zip map filter…
  datetime     datoer og tider
  regex        regulære udtryk
  plot         diagrammer (triad.plot er samme)
  io           print input
```

Et eksempel samlet:

```tri
import math;
import random;

random.seed(0);
print(math.sqrt(25.0));       // 5.0
print(random.randint(1, 10)); // 7 (med seed 0)
```

## Filer (`fs`)

```tri
import fs;

let sti = fs.join(fs.tempdir(), "hej.txt");
fs.write_text(sti, "Hello fil!");
print(fs.read_text(sti));   // Hello fil!
print(fs.exists(sti));      // true
```

## Egne moduler

Filen `util.tri`:

```tri
fn dobbelt(n) {
    return n * 2;
}
```

Filen `app.tri`, i samme mappe:

```tri
import util;

print(util.dobbelt(21));   // 42
```

## Pakker (større projekter)

```sh
./triad init minapp        # lav projekt
./triad install           # installer afhængigheder
./triad publish           # publicér til lokalt register
./triad list              # list installerede
```

Afhængigheder bor i `triad_modules/`.

✎ prøv: et modul `hils.tri` med `fn hej(navn)` og et
program, der importerer og bruger det.

Næste: [08 — arrays](08-arrays.md).
