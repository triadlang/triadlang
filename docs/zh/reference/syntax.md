# 语法 — 一页看懂整门语言

## 句子和值

```tri-frag
let x = 10;          const K = 2;          // 可变 / 冻结
10  3.14  "文本"  true  false  none       // Int Float String Bool none
"a" + "b"  2 ** 10  7 % 3                 // 拼接、乘方、取模
f"x={x} {x * 2}"                          // f-string 里算
str(42)  type(x)  len([1])                // 转换 / 揭示 / 计数
```

## 判断和循环

```tri-frag
if a { } elif b { } else { }              // and or not
for i in range(3) { }                     // 0 1 2
for v in [1, 2] { }                       // 遍历可遍历的
while ok { }  break;  continue;           // 经典三件套
match v { case 1 { } case 2 { } }         // 按值选择
```

## 函数和错误

```tri-frag
fn f(a, b=2, *args, **kw) { return a; }   // 默认 + 溢出
yield v;                                  // 暂停并交付（在 fn 里）
try { } catch e { } finally { }           // 冒险 / 倒下 / 永远
throw "原因";  assert ok;                 // 扔 / 锁
async fn f() { }  await g();              // await 只在 async 里
```

## 数据形状

```tri
let l = [1, 2]; let d = {"k": 1, "n": 0};
[1, 2]  l[0]  l.push(3)                   // 列表
{"k": 1}  d["k"]  d["n"] = 2  del d["k"]  // 字典
type P { mingzi: String; nianling: Int; } // 结构模具
class C { fn __init__(self) {} }          // 行为模具
class D inherits C { }                    // 复用 + 变化
```

## 模块

```tri
import math;  import math as m;           // 整个 / 别名
from math import sqrt, pi;                // 挑选
import triad.ntri as np;                  // triad.* 家族（39）
import numpy;  import flask;               // 真正 Python（安全名单）
```

## 物理 DSL（同一运行器）

```tri
@T(18.0)
reg a : anti_collapse = 4;
reg b : anti_collapse = 4;
ring(a, b) kappa=-2.5 for T=18.0;
OBSERVE a k_star, crystallinity, peak, atom_count;
```

## 关键字（全部）

```text
let const fn return if else elif for in while break continue
type class super import from as true false none and or not
try catch finally throw self match case yield async await with
reg entity world couple pair ring observe OBSERVE run evolve
sequence via each_for substrate composed_of assert persistent
extended structurally_open mem_memory atomic anti_collapsed
over_seeds is pass del inherits
```
