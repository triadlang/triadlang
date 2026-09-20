# 00 — Willkommen

## Was TriadLang ist, in 30 Sekunden

```
   .tri ──► ./triad ──► Ergebnis
    │           │
    │           ├── direkte Ausführung (run)
    │           ├── Typprüfung (check)
    │           ├── echtes Python innen (import numpy, flask…)
    │           └── echte Physik innen (Solver, SAT, Qubits)
    │
    └── oder physikalische DSL: reg / ring / OBSERVE
```

TriadLang ist eine vollständige Programmiersprache (Variablen,
Funktionen, Klassen, Fehler, Module) mit zwei ungewöhnlichen Türen:
Sie importiert das echte Python-Ökosystem und bringt eine Engine für
integrale Felddynamik (den TRIAD-Solver), die Dinge wie SAT und
Quantenschaltkreise löst.

## Prüfen, ob alles stimmt

```sh
./triad doctor
```

`▸ Ausgabe` — ein Umgebungsbericht (Python, Abhängigkeiten, nativ).

Falls etwas fehlt, behebt es das geführte Setup:

```sh
./triad setup
```

## Der erste Run (2 Minuten)

Datei `hi.tri`:

```tri
print("Hello from TriadLang");
```

```sh
./triad run hi.tri
```

`▸ Ausgabe`

```text
Hello from TriadLang
```

Geklappt? Dann kannst du TriadLang schon ausführen. Der Rest ist Sprache.

## Die zwei Prüfungen für immer

```sh
./triad check hi.tri     # prüft nur Typen, führt nichts aus
./triad fmt hi.tri       # formatiert die Datei
```

## Mentale Karte ab hier

```
  Schritte 1–7    Alltagssprache (wie jede Sprache lernen)
       │
  Schritte 8–9    Arrays + Python (wo triadlang die Welt trifft)
       │
  Schritte 10–11  Solver + DSL (wo triadlang Physik wird)
       │
  Schritte 12–13  Werkzeuge + nativ (Produktion und Kernel)
```

✎ probier es: Starte `./triad repl`, tippe `print(1 + 1);`, dann `exit`.

Weiter: [01 — erste Schritte](01-erste-schritte.md).
