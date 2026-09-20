# 07 — Moduler: importera världen

## De tre formerna

```tri
import math;                    // hela modulen
import math as m;               // med smeknamn
from math import sqrt, pi;      // bara det som behövs
```

## Kärn-stdlib (alltid inne)

```
  math         sqrt sin cos log exp floor ceil pi e min max
  random       random randint choice seed shuffle uniform
  string       split join replace lower upper strip
  json         parse stringify
  fs           read_text write_text exists listdir join tempdir
  time         now sleep
  collections  len range enumerate sorted zip map filter…
  datetime     datum och tider
  regex        reguljära uttryck
  plot         diagram (triad.plot är samma)
  io           print input
```

Ett exempel ihop:

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

let sokvag = fs.join(fs.tempdir(), "hej.txt");
fs.write_text(sokvag, "Hello fil!");
print(fs.read_text(sokvag));   // Hello fil!
print(fs.exists(sokvag));      // true
```

## Egna moduler

Filen `util.tri`:

```tri
fn dubbel(n) {
    return n * 2;
}
```

Filen `app.tri`, i samma mapp:

```tri
import util;

print(util.dubbel(21));   // 42
```

## Paket (större projekt)

```sh
./triad init minapp        # skapa projekt
./triad install           # installera beroenden
./triad publish           # publicera till lokala registret
./triad list              # lista installerade
```

Beroenden bor i `triad_modules/`.

✎ prova: en modul `halsa.tri` med `fn hej(namn)` och ett
program som importerar och använder den.

Nästa: [08 — arrayer](08-arrayer.md).
