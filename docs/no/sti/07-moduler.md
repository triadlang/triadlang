# 07 — Moduler: importer verden

## De tre formene

```tri
import math;                    // hele modulen
import math as m;               // med kallenavn
from math import sqrt, pi;      // bare det som trengs
```

## Kjerne-stdlib (alltid inne)

```
  math         sqrt sin cos log exp floor ceil pi e min max
  random       random randint choice seed shuffle uniform
  string       split join replace lower upper strip
  json         parse stringify
  fs           read_text write_text exists listdir join tempdir
  time         now sleep
  collections  len range enumerate sorted zip map filter…
  datetime     datoer og tider
  regex        regulære uttrykk
  plot         diagram (triad.plot er samme)
  io           print input
```

Ett eksempel samlet:

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

let sti = fs.join(fs.tempdir(), "hei.txt");
fs.write_text(sti, "Hello fil!");
print(fs.read_text(sti));   // Hello fil!
print(fs.exists(sti));      // true
```

## Egne moduler

Filen `util.tri`:

```tri
fn dobbel(n) {
    return n * 2;
}
```

Filen `app.tri`, i samme mappe:

```tri
import util;

print(util.dobbel(21));   // 42
```

## Pakker (større prosjekter)

```sh
./triad init minapp        # lag prosjekt
./triad install           # installer avhengigheter
./triad publish           # publiser til lokalt register
./triad list              # list installerte
```

Avhengigheter bor i `triad_modules/`.

✎ prøv: en modul `hils.tri` med `fn hei(navn)` og et
program som importerer og bruker den.

Neste: [08 — arrayer](08-arrays.md).
