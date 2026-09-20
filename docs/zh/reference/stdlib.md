# 标准库 — 简单模块

用 `import 名字;` 导入。签名完整；没有隐藏。

## math

```tri
import math;
math.sqrt(25.0)  math.sin(0.0)  math.cos(0.0)  math.tan(1.0)
math.log(2.7)    math.log10(100.0)  math.exp(1.0)
math.floor(3.7)  math.ceil(3.2)     math.pow(2, 10)
math.pi  math.e
math.abs(-3)  math.min(1, 2)  math.max(1, 2)  math.clamp(9, 0, 5)
```

## random

```tri
import random;
random.seed(0)
random.random()        // 0..1 浮点
random.randint(1, 10)  // 整数，两端都含
random.uniform(0.0, 1.0)
random.choice(["a", "b"])
random.shuffle([3, 1, 2])
```

## string

```tri
import string;
string.split("a b")         // ["a", "b"]
string.join("-", ["a","b"]) // "a-b"
string.replace("aaa", "a", "b")
string.lower("X")  string.upper("x")  string.strip("  x  ")
string.starts_with("triad", "tri")  string.ends_with("triad", "ad")
```

## json

```tri
import json;
print(json.stringify({"a": 1}));   // {"a": 1}
let t = json.stringify({"a": 1});
print(json.parse(t));              // {"a": 1}
// encode/decode 是 stringify/parse 的别名
```

## fs

```tri
import fs;
print(fs.join("a", "b"));              // "a/b"
let tmp = fs.join(fs.tempdir(), "r.txt");
fs.write_text(tmp, "x");
print(fs.read_text(tmp));              // x
print(fs.exists(tmp));                 // true
```

## time · collections · datetime · regex · plot · io

```tri
import time;
print(time.now() > 0);   // true
time.sleep(0.1);

import collections;
print(collections.len([1, 2]));       // 2
print(collections.range(3));          // [0, 1, 2]
print(collections.sorted([2, 1]));    // [1, 2]
print(collections.enumerate(["a"]));  // [[0, "a"]]
print(collections.zip([1], [2]));     // [[1, 2]]
print(collections.reversed([1, 2]));  // [2, 1]
fn is_pos(n) { return n > 0; }
print(collections.map(is_pos, [0, 1]));      // [false, true]
print(collections.filter(is_pos, [-1, 2]));  // [2]

import datetime;
datetime.now()        // "2026-09-19T…" (iso)
datetime.today()      // "2026-09-19"
datetime.timestamp()  // epoch 秒
datetime.year()  datetime.month()  datetime.day()

import regex;
regex.match("[0-9]+", "abc123")     // false（match = 从头）
regex.search("[0-9]+", "abc123")    // "123"
regex.findall("[0-9]", "a1b22")     // ["1", "2", "2"]
regex.replace("[0-9]", "#", "a1b")  // "a#b"

// plot：matplotlib 习惯（triad.plot 是同一个）
import plot;
plot.plot([0, 1, 2], [0, 1, 4]);
plot.title("抛物线");
plot.savefig("chu.png");
import io;
io.print("x");   // print 作函数（io.input 读 stdin）
```
