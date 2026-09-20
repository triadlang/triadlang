# 04 — Samlinger: lister og dicter

## Lister: rekkefølge betyr noe

```tri
let nums = [10, 20, 30];

print(nums[0]);      // 10 (indeks starter på 0)
print(len(nums));    // 3

nums.push(40);       // legger til sist
print(nums);         // [10, 20, 30, 40]
```

Gå gjennom:

```tri
let nums = [10, 20, 30, 40];
for n in nums {
    print(n * 2);
}
// ▸ 20 40 60 80
```

## Dicter: navn betyr noe

```tri
let person = {"navn": "Ana", "alder": 30};

print(person["navn"]);   // Ana
person["by"] = "SP";     // lager ny nøkkel
print(person["by"]);     // SP

del person["alder"];     // fjerner nøkkel
print(person);           // {"navn": "Ana", "by": "SP"}
```

Gå gjennom nøkler:

```tri
let person = {"navn": "Ana", "by": "SP"};
for nokkel in person {
    print(str(nokkel) + ": " + str(person[nokkel]));
}
```

## Tekst er en triksekasse (`string`)

```tri
import string;

print(string.upper("triad"));              // TRIAD
print(string.split("hei dev triad"));      // ["hei", "dev", "triad"]
print(string.join("-", ["a", "b"]));       // a-b
```

## JSON inn og ut (`json`)

```tri
import json;

let data = {"navn": "TriadLang", "n": 1};
let tekst = json.stringify(data);
print(tekst);                    // {"navn": "TriadLang", "n": 1}

let tilbake = json.parse(tekst);
print(tilbake["navn"]);          // TriadLang
```

## Hurtigkart

```
  trenger rekkefølge ─► liste [...] + push + indeks
  trenger navn ───────► dict {...} + ["nøkkel"]
  trenger tekst ──────► string.split/join/upper/...
  trenger utveksling ─► json.stringify / json.parse
```

✎ prøv: en handleliste (dict med navn og pris per vare)
som skriver ut totalen.

Neste: [05 — typer og klasser](05-typer-klasser.md).
