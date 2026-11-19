"""Height provenance shared by the building audit and raster features."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


HEIGHT_CATEGORIES = ("height_attribute", "floor_based", "rule_estimated", "unattributed", "missing")
RULE_SOURCES = {"type_area_variation", "default", "fixed", "estimated", "inferred", "fallback"}


def positive_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


@dataclass(frozen=True)
class BuildingHeight:
    height_m: float | None
    source: str
    category: str
    num_floors: float | None


def building_height(properties: Mapping[str, Any]) -> BuildingHeight:
    """Read attributes without inventing a storey height or a fallback height.

    An explicit height attribute describes provenance, not survey accuracy.
    A floor-only record remains missing in metres even though its category is
    floor_based. Unknown source labels are not assumed to be measurements.
    """
    height = positive_number(properties.get("height_m"))
    native_height = positive_number(properties.get("height"))
    floors = positive_number(properties.get("num_floors"))
    source = str(properties.get("height_source") or "").strip()
    if height is None and native_height is not None:
        height = native_height
        source = source or "height"
    if height is None:
        category = "floor_based" if floors is not None else "missing"
        source = source or ("num_floors" if floors is not None else "missing")
    elif source == "height":
        category = "height_attribute"
    elif source == "num_floors":
        category = "floor_based"
    elif source in RULE_SOURCES:
        category = "rule_estimated"
    else:
        category = "unattributed"
        source = source or "unknown"
    return BuildingHeight(height, source, category, floors)
