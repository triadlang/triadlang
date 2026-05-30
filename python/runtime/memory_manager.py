from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class RegisterSlot:
    name: str
    substrate_id: int
    bit_width: int = 1
    role: str = 'register'
    encoded_type: str = 'int'
    first_segment: int = -1
    last_segment: int = -1

class MemoryManager:

    def __init__(self):
        self.slots: dict[str, RegisterSlot] = {}

    def allocate(self, name: str, substrate_id: int, *, bit_width: int=1, role: str='register', encoded_type: str='int') -> RegisterSlot:
        if name in self.slots:
            raise ValueError(f'slot {name!r} already allocated')
        slot = RegisterSlot(name=name, substrate_id=substrate_id, bit_width=bit_width, role=role, encoded_type=encoded_type)
        self.slots[name] = slot
        return slot

    def get(self, name: str) -> RegisterSlot:
        if name not in self.slots:
            raise KeyError(f'unknown slot: {name!r}')
        return self.slots[name]

    def all_substrates(self) -> list[int]:
        return [s.substrate_id for s in self.slots.values()]

    def by_role(self, role: str) -> list[RegisterSlot]:
        return [s for s in self.slots.values() if s.role == role]

    def __contains__(self, name: str) -> bool:
        return name in self.slots

    def __iter__(self):
        return iter(self.slots.values())

    def __len__(self):
        return len(self.slots)