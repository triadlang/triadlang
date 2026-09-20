# 05 — Types and classes

## `type`: a simple data mold

```tri
type Person {
    name: String;
    age: Int;
}

let p = Person(name="Ana", age=30);
print(p.name);    // Ana
print(p.age);     // 30
```

`type` is pure structure: typed fields, no behavior.

## `class`: data + behavior

```
  class Name {
      fn __init__(self, ...) { ... }   // born here
      fn method(self, ...) { ... }     // lives here
  }
```

```tri
class Account {
    fn __init__(self, owner) {
        self.owner = owner;
        self.balance = 0;
    }

    fn deposit(self, amount) {
        self.balance = self.balance + amount;
    }
}

let c = Account("Ana");
c.deposit(100);
print(c.balance);   // 100
```

- `self` is the instance itself, always the first parameter.
- `__init__` runs at creation.

## Inheritance: `inherits`

```tri
class Animal {
    fn __init__(self, name) {
        self.name = name;
    }
    fn speak(self) {
        return "...";
    }
}

class Cat inherits Animal {
    fn speak(self) {
        return "meow";
    }
}

let g = Cat("Tom");
print(g.name + " says " + g.speak());
// ▸ Tom says meow
```

`Cat` inherits `name` and overrides `speak` — the rest comes from dad.

## When to use each

```
  only holding fields ──► type
  holding + doing ──────► class
  reusing + varying ────► inherits
```

✎ try it: `class Rectangle` with `width/height` and `fn area(self)`.

Next: [06 — errors](06-errors.md).
