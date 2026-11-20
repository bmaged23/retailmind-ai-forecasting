"""Features available at a forecast origin; no future sales or prices."""
from datetime import timedelta
from functools import lru_cache
import math
import holidays


@lru_cache(maxsize=64)
def _bank_holidays(year):
    # Country aggregation is unavailable in the target panel: this is an
    # England-and-Wales proxy, not a customer-specific holiday calendar.
    return holidays.country_holidays('GB', subdiv='ENG', years=[year, year + 1])


@lru_cache(maxsize=8192)
def calendar_features(day):
    weekday = day.weekday()
    season = (day.month % 12) // 3  # winter, spring, summer, autumn
    bank = _bank_holidays(day.year)
    distance = next((n for n in range(29) if day + timedelta(days=n) in bank), 29)
    return (math.sin(2 * math.pi * weekday / 7), math.cos(2 * math.pi * weekday / 7),
            math.sin(2 * math.pi * day.month / 12), math.cos(2 * math.pi * day.month / 12),
            season / 3, float(day in bank), distance / 29)


def demand_features(history):
    if len(history) < 28:
        raise ValueError('At least 28 historical days are required')
    last7, last28 = history[-7:], history[-28:]
    mean7 = sum(last7) / 7
    mean28 = sum(last28) / 28
    std28 = math.sqrt(sum((value - mean28) ** 2 for value in last28) / 28)
    since_sale = next((i for i, value in enumerate(reversed(last28)) if value > 0), 28)
    return [history[-1], history[-7], history[-28], mean7, mean28, std28,
            sum(value > 0 for value in last7) / 7, sum(value > 0 for value in last28) / 28, since_sale]
