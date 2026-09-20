# 01 — Erste Schritte

## Variablen: `let` und `const`

```tri
let x = 10;
let name = "Triad";
let aktiv = true;
const PI2 = 3.14159 * 2;

print(x + 20);      // 30
print(name);        // Triad
print(aktiv);       // true
print(PI2);         // 6.28318
```

- `let` erzeugt eine Variable, die sich ändern darf: `x = 99;` gilt.
- `const` erzeugt eine Konstante: Ändern ist ein Fehler.
- Das Semikolon beendet den Satz. Immer.

## Die vier Typen für 99 % der Zeit

```
  10            3.14           "Text"          true / false / none
  └── Int      └── Float      └── String       └── Bool (none = leer)
```

```tri
print(10 + 3.14);            // Zahl mit Zahl geht
print("hi " + "dev");        // Text mit Text klebt
print("n = " + str(42));     // Zahl wird Text mit str()
```

`▸ Ausgabe`

```text
13.14
hi dev
n = 42
```

## `print` zeigt, `str` wandelt, `type` verrät

```tri
print(str(3.14));     // "3.14"
print(type(10));   // <class 'int'>
print(type("hi"));   // <class 'str'>
```

## Kommentare

```tri
// eine Zeile

print("gilt"); // am Zeilenende auch
```

## Das REPL ist dein Notizblock

```sh
./triad repl
```

```text
>>> print(2 * 21);
42
>>> exit
```

✎ probier es: Erstelle `ich.tri`, das deinen Namen und `2 ** 10`
druckt (`**` ist Potenz). Starte mit `./triad run ich.tri`.

Weiter: [02 — Steuerung](02-steuerung.md).
