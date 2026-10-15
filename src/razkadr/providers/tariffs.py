"""Cost accounting by manually entered tariffs from the configuration.

Tariff section (per task, e.g. [asr.tariff]):

    units = "seconds"        # billed unit kind: seconds | tokens | images
    price = 0.0059           # price per unit
    currency = "USD"         # ISO 4217; accounting currency is USD (naryad 033)
    tariff_date = "2026-10-01"

Tariffs are entered by hand (naryad 004 collects the real ones); the
cost of a call is units(billed kind) * price, rounded to 6 decimals.
Without a tariff section the call is recorded with zero cost and the
"no-tariff" marker date - visible, not silently mispriced.
"""

from __future__ import annotations

NO_TARIFF_DATE = "1970-01-01"


class Tariff:
    def __init__(self, section: dict | None) -> None:
        self.units: str | None = section.get("units") if section else None
        self.price: float = float(section.get("price", 0.0)) if section else 0.0
        self.currency: str = section.get("currency", "USD") if section else "USD"
        self.tariff_date: str = section.get("tariff_date", NO_TARIFF_DATE) if section else NO_TARIFF_DATE

    def cost(self, units: list[dict]) -> float:
        if self.units is None:
            return 0.0
        billed = sum(float(u["value"]) for u in units if u.get("kind") == self.units)
        return round(billed * self.price, 6)


def tariff_from_config(task_section: dict) -> Tariff:
    return Tariff(task_section.get("tariff") if isinstance(task_section, dict) else None)
