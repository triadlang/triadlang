# 04 — Colecciones: listas y diccionarios

## Listas: el orden importa

```tri
let nums = [10, 20, 30];

print(nums[0]);      // 10 (el índice empieza en 0)
print(len(nums));    // 3

nums.push(40);       // agrega al final
print(nums);         // [10, 20, 30, 40]
```

Recorriendo:

```tri
let nums = [10, 20, 30, 40];
for n in nums {
    print(n * 2);
}
// ▸ 20 40 60 80
```

## Diccionarios: el nombre importa

```tri
let persona = {"nombre": "Ana", "edad": 30};

print(persona["nombre"]);   // Ana
persona["ciudad"] = "SP";   // crea clave nueva
print(persona["ciudad"]);   // SP

del persona["edad"];        // quita clave
print(persona);             // {"nombre": "Ana", "ciudad": "SP"}
```

Recorriendo claves:

```tri
let persona = {"nombre": "Ana", "ciudad": "SP"};
for clave in persona {
    print(str(clave) + ": " + str(persona[clave]));
}
```

## El texto es caja de trucos (`string`)

```tri
import string;

print(string.upper("triad"));              // TRIAD
print(string.split("hola dev triad"));     // ["hola", "dev", "triad"]
print(string.join("-", ["a", "b"]));       // a-b
```

## JSON entra y sale (`json`)

```tri
import json;

let datos = {"nombre": "TriadLang", "n": 1};
let texto = json.stringify(datos);
print(texto);                    // {"nombre": "TriadLang", "n": 1}

let vuelta = json.parse(texto);
print(vuelta["nombre"]);         // TriadLang
```

## Mapa rápido

```
  necesito orden ──► lista [...] + push + índice
  necesito nombre ─► dict {...} + ["clave"]
  necesito texto ──► string.split/join/upper/...
  necesito cambio ─► json.stringify / json.parse
```

✎ prueba: una lista de compras (dict con nombre y precio por ítem)
que imprima el total.

Siguiente: [05 — tipos y clases](05-tipos-clases.md).
