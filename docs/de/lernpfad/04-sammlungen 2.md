# 04 — Sammlungen: Listen und Dicts

## Listen: Reihenfolge zählt

```tri
let nums = [10, 20, 30];

print(nums[0]);      // 10 (Index beginnt bei 0)
print(len(nums));    // 3

nums.push(40);       // hängt hinten an
print(nums);         // [10, 20, 30, 40]
```

Durchgehen:

```tri
let nums = [10, 20, 30, 40];
for n in nums {
    print(n * 2);
}
// ▸ 20 40 60 80
```

## Dicts: Namen zählen

```tri
let person = {"name": "Ana", "alter": 30};

print(person["name"]);   // Ana
person["stadt"] = "SP";  // erzeugt neuen Schlüssel
print(person["stadt"]);  // SP

del person["alter"];     // entfernt Schlüssel
print(person);           // {"name": "Ana", "stadt": "SP"}
```

Schlüssel durchgehen:

```tri
let person = {"name": "Ana", "stadt": "SP"};
for key in person {
    print(str(key) + ": " + str(person[key]));
}
```

## Text ist eine Trickkiste (`string`)

```tri
import string;

print(string.upper("triad"));              // TRIAD
print(string.split("hi dev triad"));       // ["hi", "dev", "triad"]
print(string.join("-", ["a", "b"]));       // a-b
```

## JSON rein und raus (`json`)

```tri
import json;

let daten = {"name": "TriadLang", "n": 1};
let text = json.stringify(daten);
print(text);                     // {"name": "TriadLang", "n": 1}

let zurueck = json.parse(text);
print(zurueck["name"]);          // TriadLang
```

## Schnellkarte

```
  Reihenfolge nötig ──► Liste [...] + push + Index
  Namen nötig ────────► Dict {...} + ["key"]
  Text nötig ─────────► string.split/join/upper/...
  Austausch nötig ────► json.stringify / json.parse
```

✎ probier es: Eine Einkaufsliste (Dict mit Name und Preis pro
Artikel), die die Summe druckt.

Weiter: [05 — Typen und Klassen](05-typen-klassen.md).
