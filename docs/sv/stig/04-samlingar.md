# 04 — Samlingar: listor och dicts

## Listor: ordning spelar roll

```tri
let nums = [10, 20, 30];

print(nums[0]);      // 10 (index börjar på 0)
print(len(nums));    // 3

nums.push(40);       // lägger till sist
print(nums);         // [10, 20, 30, 40]
```

Gå igenom:

```tri
let nums = [10, 20, 30, 40];
for n in nums {
    print(n * 2);
}
// ▸ 20 40 60 80
```

## Dicts: namn spelar roll

```tri
let person = {"namn": "Ana", "alder": 30};

print(person["namn"]);   // Ana
person["stad"] = "SP";   // skapar ny nyckel
print(person["stad"]);   // SP

del person["alder"];     // tar bort nyckel
print(person);           // {"namn": "Ana", "stad": "SP"}
```

Gå igenom nycklar:

```tri
let person = {"namn": "Ana", "stad": "SP"};
for nyckel in person {
    print(str(nyckel) + ": " + str(person[nyckel]));
}
```

## Text är en tricklåda (`string`)

```tri
import string;

print(string.upper("triad"));              // TRIAD
print(string.split("hej dev triad"));      // ["hej", "dev", "triad"]
print(string.join("-", ["a", "b"]));       // a-b
```

## JSON in och ut (`json`)

```tri
import json;

let data = {"namn": "TriadLang", "n": 1};
let text = json.stringify(data);
print(text);                     // {"namn": "TriadLang", "n": 1}

let tillbaka = json.parse(text);
print(tillbaka["namn"]);         // TriadLang
```

## Snabbkarta

```
  behöver ordning ──► lista [...] + push + index
  behöver namn ─────► dict {...} + ["nyckel"]
  behöver text ─────► string.split/join/upper/...
  behöver utbyte ───► json.stringify / json.parse
```

✎ prova: en inköpslista (dict med namn och pris per vara)
som skriver ut totalen.

Nästa: [05 — typer och klasser](05-typer-klasser.md).
