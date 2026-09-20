# 04 — Collections: lists and dicts

## Lists: order matters

```tri
let nums = [10, 20, 30];

print(nums[0]);      // 10 (index starts at 0)
print(len(nums));    // 3

nums.push(40);       // appends at the end
print(nums);         // [10, 20, 30, 40]
```

Walking:

```tri
let nums = [10, 20, 30, 40];
for n in nums {
    print(n * 2);
}
// ▸ 20 40 60 80
```

## Dicts: names matter

```tri
let person = {"name": "Ana", "age": 30};

print(person["name"]);   // Ana
person["city"] = "SP";   // creates a new key
print(person["city"]);   // SP

del person["age"];       // removes a key
print(person);           // {"name": "Ana", "city": "SP"}
```

Walking keys:

```tri
let person = {"name": "Ana", "city": "SP"};
for key in person {
    print(str(key) + ": " + str(person[key]));
}
```

## Text is a bag of tricks (`string`)

```tri
import string;

print(string.upper("triad"));              // TRIAD
print(string.split("hi dev triad"));       // ["hi", "dev", "triad"]
print(string.join("-", ["a", "b"]));       // a-b
```

## JSON in and out (`json`)

```tri
import json;

let data = {"name": "TriadLang", "n": 1};
let text = json.stringify(data);
print(text);                     // {"name": "TriadLang", "n": 1}

let back = json.parse(text);
print(back["name"]);             // TriadLang
```

## Quick map

```
  need order ──► list [...] + push + index
  need names ──► dict {...} + ["key"]
  need text ───► string.split/join/upper/...
  need exchange ► json.stringify / json.parse
```

✎ try it: a shopping list (dict with name and price per item)
that prints the total.

Next: [05 — types and classes](05-types-classes.md).
