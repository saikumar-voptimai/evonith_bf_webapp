"""Pure frontend adapters from canonical V-Board JSON to plotter inputs."""

from __future__ import annotations

from typing import Any


JsonDict = dict[str, Any]


def longitudinal_temperature_arrays(contours: JsonDict) -> tuple[list[list[Any]], list[list[Any]], list[list[Any]]]:
    """Return Q1-Q4 mean/max/min arrays ordered by elevation."""

    means = [[], [], [], []]
    maxima = [[], [], [], []]
    minima = [[], [], [], []]
    for level in contours.get("temperature", {}).get("levels", []):
        by_quadrant = {item["quadrant_id"]: item for item in level.get("quadrants", [])}
        for idx in range(4):
            item = by_quadrant.get(f"Q{idx + 1}", {})
            means[idx].append(item.get("mean"))
            maxima[idx].append(item.get("maximum"))
            minima[idx].append(item.get("minimum"))
    return means, maxima, minima


def circumferential_heatload_rows(catalog: JsonDict, contours: JsonDict) -> tuple[list[list[list[Any]]], list[str]]:
    rows_by_id = {
        row.get("row_id"): row
        for row in contours.get("heatload", {}).get("rows", [])
    }
    values: list[list[list[Any]]] = []
    titles: list[str] = []
    for row in catalog.get("rows", []):
        row_id = row["id"]
        response_row = rows_by_id.get(row_id, {})
        by_quadrant = {
            item["quadrant_id"]: item
            for item in response_row.get("quadrants", [])
        }
        values.append(_stats_for_quadrants(by_quadrant))
        titles.append(row_id)
    return values, titles


def circumferential_temperature_groups(catalog: JsonDict, contours: JsonDict) -> list[JsonDict]:
    levels_by_id = {
        level.get("level_id"): level
        for level in contours.get("temperature", {}).get("levels", [])
    }
    labels_by_id = {
        level.get("id"): level.get("label")
        for level in catalog.get("temperature_levels", [])
    }
    groups: list[JsonDict] = []
    for group in catalog.get("circumferential_temperature_groups", []):
        field_values = []
        titles = []
        for level_id in group.get("level_ids", []):
            response_level = levels_by_id.get(level_id, {})
            by_quadrant = {
                item["quadrant_id"]: item
                for item in response_level.get("quadrants", [])
            }
            field_values.append(_stats_for_quadrants(by_quadrant))
            titles.append(f"At {labels_by_id.get(level_id, level_id)}")
        groups.append(
            {
                "id": group["id"],
                "title": group["title"],
                "field_values": field_values,
                "titles": titles,
            }
        )
    return groups


def _stats_for_quadrants(by_quadrant: dict[str, JsonDict]) -> list[list[Any]]:
    means = []
    maxima = []
    minima = []
    for idx in range(4):
        item = by_quadrant.get(f"Q{idx + 1}", {})
        means.append(item.get("mean"))
        maxima.append(item.get("maximum"))
        minima.append(item.get("minimum"))
    return [means, maxima, minima]
