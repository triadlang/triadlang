# 04 — Samlinger: lister og dicts

## Lister: rækkefølge betyder noget

```tri
let nums = [10, 20, 30];

print(nums[0]);      // 10 (indeks starter ved 0)
print(len(nums));    // 3

nums.push(40);       // lægger til sidst
print(nums);         // [10, 20, 30, 40]
```

Gå igennem:

```tri
let nums = [10, 20, 30, 40];
for n in nums {
    print(n * 2);
}
// ▸ 20 40 60 80
```

## Dicts: navne betyder noget

```tri
let person = {"navn": "Ana", "alder": 30};

print(person["navn"]);   // Ana
person["by"] = "SP";     // laver ny nøgle
print(person["by"]);     // SP

del person["alder"];     // fjerner nøgle
print(person);           // {"navn": "Ana", "by": "SP"}
```

Gå igennem nøgler:

```tri
let person = {"navn": "Ana", "by": "SP"};
for noegle in person {
    print(str(noegle) + ": " + str(person[noegle]));
}
```

## Tekst er en trickkasse (`string`)

```tri
import string;

print(string.upper("triad"));              // TRIAD
print(string.split("hej dev triad"));      // ["hej", "dev", "triad"]
print(string.join("-", ["a", "b"]));       // a-b
```

## JSON ind og ud (`json`)

```tri
import json;

let data = {"navn": "TriadLang", "n": 1};
let tekst = json.stringify(data);
print(tekst);                    // {"navn": "TriadLang", "n": 1}

let tilbage = json.parse(tekst);
print(tilbage["navn"]);          // TriadLang
```

## Hurtigkort

```
  behøver rækkefølge ─► liste [...] + push + indeks
  behøver navne ──────► dict {...} + ["nøgle"]
  behøver tekst ──────► string.split/join/upper/...
  behøver udveksling ─► json.stringify / json.parse
```

✎ prøv: en indkøbsliste (dict med navn og pris per vare),
der udskriver totalen.

Næste: [05 — typer og klasser](05-typer-klasser.md).
