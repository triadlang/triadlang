# 05 — Tipos y clases

## `type`: un molde simple de datos

```tri
type Persona {
    nombre: String;
    edad: Int;
}

let p = Persona(nombre="Ana", edad=30);
print(p.nombre);    // Ana
print(p.edad);      // 30
```

`type` es estructura pura: campos con tipo, sin comportamiento.

## `class`: datos + comportamiento

```
  class Nombre {
      fn __init__(self, ...) { ... }   // nace aquí
      fn metodo(self, ...) { ... }     // vive aquí
  }
```

```tri
class Cuenta {
    fn __init__(self, titular) {
        self.titular = titular;
        self.saldo = 0;
    }

    fn depositar(self, valor) {
        self.saldo = self.saldo + valor;
    }
}

let c = Cuenta("Ana");
c.depositar(100);
print(c.saldo);   // 100
```

- `self` es la propia instancia, siempre el primer parámetro.
- `__init__` corre en la creación.

## Herencia: `inherits`

```tri
class Animal {
    fn __init__(self, nombre) {
        self.nombre = nombre;
    }
    fn habla(self) {
        return "...";
    }
}

class Gato inherits Animal {
    fn habla(self) {
        return "miau";
    }
}

let g = Gato("Tom");
print(g.nombre + " dice " + g.habla());
// ▸ Tom dice miau
```

`Gato` hereda `nombre` y cambia `habla` — el resto viene del padre.

## Cuándo cada uno

```
  solo guardar campos ──► type
  guardar + hacer ──────► class
  reusar y variar ──────► inherits
```

✎ prueba: `class Rectangulo` con `ancho/alto` y `fn area(self)`.

Siguiente: [06 — errores](06-errores.md).
