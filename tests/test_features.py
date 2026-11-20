from datetime import date
import pytest
from retailmind.features import calendar_features, demand_features


def test_bank_holiday_is_date_known_and_distinct():
    assert calendar_features(date(2011, 4, 29))[5] == 1  # royal wedding bank holiday
    assert calendar_features(date(2011, 5, 3))[5] == 0


def test_demand_features_use_past_and_zero_frequency():
    values = [0] * 27 + [7]
    result = demand_features(values)
    assert result[0] == 7 and result[7] == pytest.approx(1 / 28) and result[-1] == 0
    assert demand_features([0] * 28)[-1] == 28
