# 07 — Module: die Welt importieren

## Die drei Formen

```tri
import math;                    // ganzes Modul
import math as m;               // mit Spitzname
from math import sqrt, pi;      // nur was nötig ist
```

## Kern-Stdlib (immer da)

```
  math         sqrt sin cos log exp floor ceil pi e min max
  random       random randint choice seed shuffle uniform
  string       split join replace lower upper strip
  json         parse stringify
  fs           read_text write_text exists listdir join tempdir
  time         now sleep
  collections  len range enumerate sorted zip map filter…
  datetime     Daten und Zeiten
  regex        reguläre Ausdrücke
  plot         Diagramme (triad.plot ist dasselbe)
  io           print input
```

Ein Beispiel zusammen:

```tri
import math;
import random;

random.seed(0);
print(math.sqrt(25.0));       // 5.0
print(random.randint(1, 10)); // 7 (mit seed 0)
```

## Dateien (`fs`)

```tri
import fs;

let pfad = fs.join(fs.tempdir(), "hi.txt");
fs.write_text(pfad, "Hello Datei!");
print(fs.read_text(pfad));   // Hello Datei!
print(fs.exists(pfad));      // true
```

## Eigene Module

Datei `util.tri`:

```tri
fn doppel(n) {
    return n * 2;
}
```

Datei `app.tri`, im selben Ordner:

```tri
import util;

print(util.doppel(21));   // 42
```

## Pakete (größere Projekte)

```sh
./triad init meineapp     # Projekt anlegen
./triad install           # Abhängigkeiten installieren
./triad publish           # ins lokale Register
./triad list              # Installiertes auflisten
```

Abhängigkeiten wohnen in `triad_modules/`.

✎ probier es: Ein Modul `gruss.tri` mit `fn hi(name)` und ein
Programm, das es importiert und nutzt.

Weiter: [08 — Arrays](08-arrays.md).
