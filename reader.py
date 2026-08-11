
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd

READER_VERSION = "9.0.0-global-trace"

STATION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "NHK総合": (182, 188, 195),
    "日本テレビ": (254, 216, 120),
    "テレビ朝日": (246, 162, 228),
    "TBS": (100, 148, 240),
    "テレビ東京": (89, 113, 175),
    "フジテレビ": (243, 124, 124),
}

@dataclass(frozen=True)
class Calibration:
    graph_left: float
    graph_right: float
    graph_top: float
    graph_bottom: float
    y_zero: float
    pixels_per_percent: float
    pixels_per_minute: float
    start_time: datetime
    end_time: datetime
    duration_minutes: int
    grid_percent_step: float
    grid_gap_pixels: float
    time_axis_method: str = "unknown"
    time_line_inliers: int = 0

def _group_consecutive(values: np.ndarray | list[int], max_gap: int = 1) -> list[list[int]]:
    groups: list[list[int]] = []
    for raw in values:
        value = int(raw)
        if not groups or value - groups[-1][-1] > max_gap:
            groups.append([value])
        else:
            groups[-1].append(value)
    return groups

def _neutral_gray_mask(
    pixels: np.ndarray,
    *,
    max_spread: int,
    min_brightness: float,
    max_brightness: float,
) -> np.ndarray:
    work = pixels.astype(np.int16)
    spread = work.max(axis=2) - work.min(axis=2)
    brightness = work.mean(axis=2)
    return (
        (spread <= max_spread)
        & (brightness >= min_brightness)
        & (brightness <= max_brightness)
    )

def _find_thin_horizontal_lines(arr: np.ndarray) -> list[int]:
    h, w, _ = arr.shape
    x0 = int(w * 0.08)
    x1 = int(w * 0.95)

    crop = arr[:, x0:x1]
    mask = _neutral_gray_mask(
        crop,
        max_spread=20,
        min_brightness=195,
        max_brightness=253,
    )
    row_score = mask.mean(axis=1)

    candidates = np.where(row_score >= 0.65)[0]
    groups = _group_consecutive(candidates, max_gap=1)

    thin_lines = [
        int(round(np.mean(group)))
        for group in groups
        if 1 <= len(group) <= 4
        and int(h * 0.18) <= int(round(np.mean(group))) <= int(h * 0.99)
    ]

    # TVALは低視聴率帯で0,2,4の3本だけになることがある。
    if len(thin_lines) < 3:
        raise RuntimeError(
            "水平グリッド線を3本以上検出できませんでした。"
            "グラフ全体と0%線が入った画像を使ってください。"
        )
    return thin_lines

def _select_regular_grid(
    lines: list[int],
    image_height: int,
    min_lines: int = 3,
) -> tuple[list[int], float]:
    ys = np.array(sorted(set(lines)), dtype=float)
    if len(ys) < min_lines:
        raise RuntimeError("水平グリッド線が不足しています。")

    # 固定45～220pxを廃止し、画像高さに比例させる。
    min_gap = max(20.0, image_height * 0.015)
    max_gap = max(min_gap + 1.0, image_height * 0.45)

    gap_candidates: list[float] = []
    for i in range(len(ys)):
        for j in range(i + 1, len(ys)):
            difference = ys[j] - ys[i]
            for steps in range(1, 13):
                gap = difference / steps
                if min_gap <= gap <= max_gap:
                    gap_candidates.append(float(gap))

    best_score = -1e18
    best_values: list[float] | None = None

    for gap in gap_candidates:
        tolerance = max(3.0, gap * 0.035)

        for anchor in ys:
            step_numbers = np.round((ys - anchor) / gap)
            residuals = np.abs(ys - (anchor + step_numbers * gap))
            matched_steps = np.unique(
                step_numbers[residuals <= tolerance].astype(int)
            )
            if len(matched_steps) < min_lines:
                continue

            step_groups = _group_consecutive(matched_steps, max_gap=1)
            longest_steps = max(step_groups, key=len)
            if len(longest_steps) < min_lines:
                continue

            values: list[float] = []
            residual_sum = 0.0
            for step in longest_steps:
                target = anchor + step * gap
                index = int(np.argmin(np.abs(ys - target)))
                values.append(float(ys[index]))
                residual_sum += abs(float(ys[index]) - target)

            span = max(values) - min(values)
            score = len(values) * 1000.0 + span - residual_sum * 10.0

            if score > best_score:
                best_score = score
                best_values = values

    if best_values is None or len(best_values) < min_lines:
        raise RuntimeError("規則的な水平グリッドを特定できませんでした。")

    grid = sorted(set(int(round(value)) for value in best_values))
    gap = float(np.median(np.diff(grid)))

    if np.max(np.abs(np.diff(grid) - gap)) > max(4.0, gap * 0.06):
        raise RuntimeError("水平グリッドの間隔が不規則です。")

    return grid, gap

def _find_rough_graph_x_bounds(arr: np.ndarray, grid: list[int]) -> tuple[int, int]:
    h, w, _ = arr.shape
    spans: list[tuple[int, int]] = []
    rows_to_use = grid[: min(3, len(grid))]

    for y in rows_to_use:
        row = arr[y:y + 1]
        mask = _neutral_gray_mask(
            row,
            max_spread=25,
            min_brightness=190,
            max_brightness=253,
        )[0]

        xs = np.where(mask)[0]
        if len(xs) == 0:
            continue

        groups = _group_consecutive(xs, max_gap=max(6, int(w * 0.008)))
        long_groups = [
            group for group in groups
            if group[-1] - group[0] + 1 >= max(80, int(w * 0.05))
        ]
        if not long_groups:
            continue

        left = min(group[0] for group in long_groups)
        right = max(group[-1] for group in long_groups)

        if right - left >= w * 0.55:
            spans.append((left, right))

    if not spans:
        raise RuntimeError("グラフ左右端を検出できませんでした。")

    left = int(round(np.median([span[0] for span in spans])))
    right = int(round(np.median([span[1] for span in spans])))

    if right - left < w * 0.55:
        raise RuntimeError("検出したグラフ幅が狭すぎます。")

    return left, right

def _find_vertical_hour_lines(
    arr: np.ndarray,
    graph_top: int,
    graph_bottom: int,
    rough_left: int,
    rough_right: int,
) -> list[int]:
    roi = arr[graph_top:graph_bottom + 1]
    mask = _neutral_gray_mask(
        roi,
        max_spread=20,
        min_brightness=195,
        max_brightness=253,
    )
    column_score = mask.mean(axis=0)

    w = arr.shape[1]
    search_left = max(0, rough_left - max(30, int(w * 0.03)))
    search_right = min(w - 1, int(w * 0.98))
    nms_distance = max(10, int(w * 0.008))

    candidates: list[tuple[int, float]] = []
    for x in range(max(1, search_left), min(len(column_score) - 2, search_right) + 1):
        score = float(column_score[x])
        if not (0.30 <= score <= 0.79):
            continue
        if score >= column_score[x - 1] and score >= column_score[x + 1]:
            candidates.append((x, score))

    selected: list[tuple[int, float]] = []
    for x, score in sorted(candidates, key=lambda item: item[1], reverse=True):
        if all(abs(x - old_x) > nms_distance for old_x, _ in selected):
            selected.append((x, score))

    return sorted(x for x, _ in selected)

def _match_time_axis_model(
    peaks: list[int],
    expected_offsets: list[int],
    rough_left: int,
    rough_right: int,
    duration_minutes: int,
    image_width: int,
) -> tuple[float, float, float, list[tuple[int, float]], list[int]]:
    if len(peaks) < 2:
        raise RuntimeError("縦の時刻目盛りを2本以上検出できませんでした。")
    if len(expected_offsets) < 2:
        raise RuntimeError("表示期間内の正時候補が不足しています。")

    rough_ppm = (rough_right - rough_left) / float(duration_minutes)
    rough_hour_spacing = rough_ppm * 60.0

    max_hour_steps = max(1, int(np.ceil(duration_minutes / 60.0)))
    spacing_candidates: list[float] = [rough_hour_spacing]

    min_peak_difference = max(60.0, image_width * 0.04)
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            difference = float(peaks[j] - peaks[i])
            if difference < min_peak_difference:
                continue
            for hour_steps in range(1, max_hour_steps + 1):
                spacing = difference / hour_steps
                if rough_hour_spacing * 0.55 <= spacing <= rough_hour_spacing * 1.45:
                    spacing_candidates.append(spacing)

    best: dict[str, object] | None = None

    for hour_spacing in spacing_candidates:
        ppm = hour_spacing / 60.0
        tolerance = max(7.0, hour_spacing * 0.035)

        for peak in peaks:
            for offset in expected_offsets:
                candidate_start = float(peak) - float(offset) * ppm
                predicted = np.array(
                    [candidate_start + expected * ppm for expected in expected_offsets],
                    dtype=float,
                )

                pairs: list[tuple[float, int, int, float]] = []
                for pred_index, predicted_x in enumerate(predicted):
                    for peak_index, detected_x in enumerate(peaks):
                        distance = abs(float(detected_x) - float(predicted_x))
                        if distance <= tolerance:
                            pairs.append(
                                (distance, pred_index, peak_index, float(detected_x))
                            )

                pairs.sort(key=lambda item: item[0])
                used_pred: set[int] = set()
                used_peak: set[int] = set()
                matches: list[tuple[int, float]] = []
                residual_sum = 0.0

                for distance, pred_index, peak_index, detected_x in pairs:
                    if pred_index in used_pred or peak_index in used_peak:
                        continue
                    used_pred.add(pred_index)
                    used_peak.add(peak_index)
                    matches.append((expected_offsets[pred_index], detected_x))
                    residual_sum += distance

                if len(matches) < 2:
                    continue

                candidate_end = candidate_start + duration_minutes * ppm
                boundary_penalty = (
                    abs(candidate_start - rough_left)
                    + abs(candidate_end - rough_right)
                )
                spacing_penalty = abs(ppm - rough_ppm) * 60.0

                score = (
                    len(matches) * 100000.0
                    - residual_sum * 100.0
                    - boundary_penalty * 2.0
                    - spacing_penalty * 10.0
                )

                if candidate_start < -40 or candidate_end > image_width + 40:
                    score -= 100000.0

                if best is None or score > float(best["score"]):
                    best = {
                        "score": score,
                        "matches": matches,
                        "start": candidate_start,
                        "ppm": ppm,
                    }

    if best is None:
        raise RuntimeError("縦の時刻目盛りから時間軸を較正できませんでした。")

    matches = list(best["matches"])  # type: ignore[arg-type]
    offsets = np.array([pair[0] for pair in matches], dtype=float)
    detected = np.array([pair[1] for pair in matches], dtype=float)
    design = np.column_stack([np.ones(len(offsets)), offsets])
    start_x, pixels_per_minute = np.linalg.lstsq(
        design, detected, rcond=None
    )[0]
    start_x = float(start_x)
    pixels_per_minute = float(pixels_per_minute)
    end_x = start_x + duration_minutes * pixels_per_minute

    predicted_matches = start_x + offsets * pixels_per_minute
    residuals = np.abs(predicted_matches - detected)
    hour_spacing = pixels_per_minute * 60.0
    allowed_residual = max(8.0, hour_spacing * 0.04)

    if float(np.max(residuals)) > allowed_residual:
        raise RuntimeError(
            "正時線の対応後も残差が大きく、安全に時間軸を確定できませんでした。"
        )

    allowed_boundary = max(50.0, rough_hour_spacing * 0.22)
    if (
        start_x < rough_left - allowed_boundary
        or end_x > rough_right + allowed_boundary
    ):
        raise RuntimeError("推定した時間軸がグラフ領域から大きく外れています。")

    matched_peak_values = {int(round(pair[1])) for pair in matches}
    outliers = [peak for peak in peaks if peak not in matched_peak_values]

    return start_x, end_x, pixels_per_minute, matches, outliers

def _color_distance(roi: np.ndarray, target_rgb: Tuple[int, int, int]) -> np.ndarray:
    work = roi.astype(np.int16)
    target = np.array(target_rgb, dtype=np.int16)
    return np.max(np.abs(work - target), axis=2)

def _station_color_mask(
    roi: np.ndarray,
    station: str,
    target_rgb: Tuple[int, int, int],
    tolerance: int,
) -> np.ndarray:
    distance = _color_distance(roi, target_rgb)
    if station == "NHK総合":
        work = roi.astype(np.int16)
        spread = work.max(axis=2) - work.min(axis=2)
        brightness = work.mean(axis=2)
        return (
            (distance <= max(55, tolerance))
            & (brightness < 220)
            & (spread <= 30)
        )
    return distance <= tolerance

def _find_colored_plot_bounds(
    arr: np.ndarray,
    graph_top: int,
    graph_bottom: int,
    rough_left: int,
    rough_right: int,
    tolerance: int = 50,
) -> tuple[int, int]:
    x0 = max(0, rough_left - 120)
    x1 = min(arr.shape[1] - 1, rough_right + 120)
    roi = arr[graph_top:graph_bottom + 1, x0:x1 + 1]

    left_edges: list[int] = []
    right_edges: list[int] = []

    for station, color in STATION_COLORS.items():
        mask = _station_color_mask(roi, station, color, tolerance)
        columns = np.where(mask.sum(axis=0) > 0)[0]
        if len(columns) < 10:
            continue

        groups = _group_consecutive(columns, max_gap=8)
        longest = max(groups, key=lambda group: group[-1] - group[0] + 1)

        if longest[-1] - longest[0] + 1 < arr.shape[1] * 0.25:
            continue

        left_edges.append(x0 + longest[0])
        right_edges.append(x0 + longest[-1])

    if len(left_edges) < 3:
        raise RuntimeError("色線から時間軸範囲を推定できませんでした。")

    left = int(round(float(np.median(left_edges))))
    right = int(round(float(np.median(right_edges))))

    if right - left < arr.shape[1] * 0.45:
        raise RuntimeError("色線から得た時間軸範囲が狭すぎます。")

    return left, right

def _calibrate_time_axis(
    arr: np.ndarray,
    grid: list[int],
    rough_left: int,
    rough_right: int,
    end_time: datetime,
    duration_minutes: int,
) -> tuple[float, float, float, str, int]:
    peaks = _find_vertical_hour_lines(
        arr, grid[0], grid[-1], rough_left, rough_right
    )

    start_time = end_time - timedelta(minutes=duration_minutes)
    first_hour_offset = (-start_time.minute) % 60
    expected_offsets = list(range(first_hour_offset, duration_minutes + 1, 60))

    if len(peaks) >= 2 and len(expected_offsets) >= 2:
        try:
            start_x, end_x, ppm, matches, _ = _match_time_axis_model(
                peaks=peaks,
                expected_offsets=expected_offsets,
                rough_left=rough_left,
                rough_right=rough_right,
                duration_minutes=duration_minutes,
                image_width=arr.shape[1],
            )
            return float(start_x), float(end_x), float(ppm), "hour-lines", int(len(matches))
        except RuntimeError:
            pass

    color_left, color_right = _find_colored_plot_bounds(
        arr, grid[0], grid[-1], rough_left, rough_right, tolerance=50
    )
    start_x = float(color_left)
    end_x = float(color_right)
    ppm = (end_x - start_x) / float(duration_minutes)
    if ppm <= 0:
        raise RuntimeError("時間軸の横幅が不正です。")
    return start_x, end_x, ppm, "color-range-fallback", 0

def _find_cursor_x(
    arr: np.ndarray,
    grid: list[int],
    rough_left: int,
    rough_right: int,
) -> int | None:
    try:
        color_left, color_right = _find_colored_plot_bounds(
            arr, grid[0], grid[-1], rough_left, rough_right, tolerance=50
        )
    except RuntimeError:
        color_left, color_right = rough_left, rough_right

    roi = arr[grid[0]:grid[-1] + 1]
    mask = _neutral_gray_mask(
        roi,
        max_spread=35,
        min_brightness=150,
        max_brightness=253,
    )
    scores = mask.mean(axis=0)

    x_axis = np.arange(arr.shape[1])
    candidates = np.where(
        (scores >= 0.82)
        & (x_axis >= color_left - 10)
        & (x_axis <= color_right + 10)
    )[0]
    groups = _group_consecutive(candidates, max_gap=2)
    thin_groups = [group for group in groups if 1 <= len(group) <= 8]
    if not thin_groups:
        return None

    chosen = max(thin_groups, key=lambda group: float(np.max(scores[group])))
    return int(round(float(np.mean(chosen))))

def _find_cursor_station_y(
    arr: np.ndarray,
    cursor_x: int,
    graph_top: int,
    graph_bottom: int,
    target_rgb: Tuple[int, int, int],
    tolerance: int,
) -> float | None:
    window = 14
    x0 = max(0, cursor_x - window)
    x1 = min(arr.shape[1] - 1, cursor_x + window)

    roi = arr[graph_top:graph_bottom + 1, x0:x1 + 1]
    distance = _color_distance(roi, target_rgb)
    color_mask = distance <= tolerance
    row_counts = color_mask.sum(axis=1)

    maximum = int(row_counts.max())
    if maximum < 2:
        return None

    threshold = max(2, int(np.ceil(maximum * 0.45)))
    rows = np.where(row_counts >= threshold)[0]
    groups = _group_consecutive(rows, max_gap=2)
    if not groups:
        return None

    chosen = max(groups, key=lambda group: int(row_counts[group].sum()))
    y_values = np.array(chosen, dtype=float)
    weights = row_counts[chosen].astype(float)
    return float(np.average(y_values, weights=weights) + graph_top)

def _vertical_local_min(distance: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return distance.astype(np.float32)

    h, w = distance.shape
    padded = np.pad(
        distance.astype(np.float32),
        ((radius, radius), (0, 0)),
        mode="constant",
        constant_values=255.0,
    )
    views = [padded[offset:offset + h] for offset in range(radius * 2 + 1)]
    return np.minimum.reduce(views)

def _choose_anchor_from_color(
    local_distance: np.ndarray,
    *,
    preferred_x: int,
    tolerance: int,
) -> tuple[int, float]:
    h, w = local_distance.shape
    preferred_x = int(np.clip(preferred_x, 0, w - 1))

    for radius in range(w):
        for x in (preferred_x - radius, preferred_x + radius):
            if x < 0 or x >= w:
                continue
            column = local_distance[:, x]
            y = int(np.argmin(column))
            if float(column[y]) <= tolerance:
                return x, float(y)

    x = preferred_x
    y = int(np.argmin(local_distance[:, x]))
    if float(local_distance[y, x]) > max(90.0, tolerance * 2.0):
        raise RuntimeError("安定した色線アンカーを見つけられませんでした。")
    return x, float(y)

def _beam_trace_direction(
    emission: np.ndarray,
    *,
    anchor_x: int,
    anchor_y: float,
    direction: int,
    beam_width: int,
    smoothness: float,
    anchor_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = emission.shape
    if direction not in (-1, 1):
        raise ValueError("directionは-1または1です。")

    columns = list(range(anchor_x, w if direction == 1 else -1, direction))
    n = len(columns)
    k = min(beam_width, h)

    beam_y = np.empty((n, k), dtype=np.int16)
    parent_dtype = np.int8 if k <= 127 else np.int16
    parents = np.zeros((n, k), dtype=parent_dtype)

    y_axis_1d = np.arange(h, dtype=np.float32)
    first_x = columns[0]
    first_cost = (
        emission[:, first_x].astype(np.float32)
        + anchor_strength * np.abs(y_axis_1d - float(anchor_y))
    )
    selected = np.argpartition(first_cost, k - 1)[:k]
    selected = selected[np.argsort(first_cost[selected])]
    beam_y[0] = selected
    costs = first_cost[selected].astype(np.float32)
    costs -= costs.min()

    y_axis = y_axis_1d[:, None]

    for step_index in range(1, n):
        x = columns[step_index]
        prev_y = beam_y[step_index - 1].astype(np.float32)
        transition = (
            costs[None, :]
            + smoothness * np.abs(y_axis - prev_y[None, :])
        )
        best_parent = np.argmin(transition, axis=1)
        best_cost = (
            transition[np.arange(h), best_parent]
            + emission[:, x]
        )

        selected = np.argpartition(best_cost, k - 1)[:k]
        selected = selected[np.argsort(best_cost[selected])]
        beam_y[step_index] = selected
        parents[step_index] = best_parent[selected]

        costs = best_cost[selected].astype(np.float32)
        costs -= costs.min()

    slot = int(np.argmin(costs))
    path_local = np.empty(n, dtype=np.float32)
    for step_index in range(n - 1, -1, -1):
        path_local[step_index] = float(beam_y[step_index, slot])
        if step_index:
            slot = int(parents[step_index, slot])

    x_values = np.array(columns, dtype=int)
    return x_values, path_local

def _refine_path(
    raw_distance: np.ndarray,
    path: np.ndarray,
    tolerance: int,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = raw_distance.shape
    refined = path.astype(np.float32).copy()
    observed = np.zeros(w, dtype=bool)

    search_radius = max(2, radius + 2)
    for x in range(w):
        center = int(round(float(path[x])))
        y0 = max(0, center - search_radius)
        y1 = min(h - 1, center + search_radius)
        rows = np.arange(y0, y1 + 1)
        distances = raw_distance[y0:y1 + 1, x]
        good = rows[distances <= tolerance]

        if len(good):
            # 近傍の同色線の中心を使う。単発ノイズより線幅を優先。
            groups = _group_consecutive(good, max_gap=1)
            chosen = min(
                groups,
                key=lambda group: abs(float(np.mean(group)) - float(path[x])),
            )
            refined[x] = float(np.mean(chosen))
            observed[x] = True

    return refined, observed

def _trace_station_global(
    arr: np.ndarray,
    station: str,
    color: Tuple[int, int, int],
    trace_left: int,
    trace_right: int,
    graph_top: int,
    graph_bottom: int,
    tolerance: int,
    cursor_x: int | None = None,
    cursor_y: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | str]]:
    x0 = max(0, int(trace_left))
    x1 = min(arr.shape[1] - 1, int(trace_right))
    roi = arr[graph_top:graph_bottom + 1, x0:x1 + 1]

    raw_distance = _color_distance(roi, color)
    line_radius = max(1, min(4, int(round((graph_bottom - graph_top + 1) / 400.0))))
    local_distance = _vertical_local_min(raw_distance, line_radius)

    # 距離を二値化せずソフトコストとして使う。
    emission = np.minimum(local_distance.astype(np.float32), 120.0)

    anchor_tolerance = max(tolerance, 55 if station == "NHK総合" else tolerance)
    preferred_local_x = (x1 - x0) // 2
    suppressed_cursor_band: tuple[int, int] | None = None

    # NHKだけは灰色カーソル自身がNHK色候補になり得る。
    # カーソル中心のごく狭い帯を「色証拠なし」として扱い、
    # 左右の実際のNHK線から経路を橋渡しする。
    if station == "NHK総合":
        work = roi.astype(np.int16)
        brightness = work.mean(axis=2)
        spread = work.max(axis=2) - work.min(axis=2)
        grid_like = (brightness > 210) & (spread <= 18)
        emission += grid_like.astype(np.float32) * 12.0

        if cursor_x is not None and x0 <= cursor_x <= x1:
            local_cursor_x = int(cursor_x - x0)
            suppress_radius = max(4, int(round(arr.shape[1] * 0.0025)))
            band_left = max(0, local_cursor_x - suppress_radius)
            band_right = min(emission.shape[1], local_cursor_x + suppress_radius + 1)
            emission[:, band_left:band_right] = 70.0
            suppressed_cursor_band = (band_left, band_right)

            if band_left <= preferred_local_x < band_right:
                preferred_local_x = min(
                    emission.shape[1] - 1,
                    band_right + max(6, suppress_radius),
                )

    anchor_used = "color"
    if (
        station != "NHK総合"
        and cursor_x is not None
        and cursor_y is not None
        and x0 <= cursor_x <= x1
    ):
        anchor_x = int(cursor_x - x0)
        anchor_y = float(cursor_y - graph_top)
        anchor_used = "cursor"
    else:
        anchor_x, anchor_y = _choose_anchor_from_color(
            local_distance,
            preferred_x=preferred_local_x,
            tolerance=anchor_tolerance,
        )

        # NHKのアンカーが抑制帯へ入った場合は、その外側から取り直す。
        if (
            suppressed_cursor_band is not None
            and suppressed_cursor_band[0] <= anchor_x < suppressed_cursor_band[1]
        ):
            retry_x = min(
                emission.shape[1] - 1,
                suppressed_cursor_band[1] + max(6, suppressed_cursor_band[1] - suppressed_cursor_band[0]),
            )
            anchor_x, anchor_y = _choose_anchor_from_color(
                local_distance,
                preferred_x=retry_x,
                tolerance=anchor_tolerance,
            )

    right_x, right_path = _beam_trace_direction(
        emission,
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        direction=1,
        beam_width=24,
        smoothness=0.45,
        anchor_strength=6.0,
    )
    left_x, left_path = _beam_trace_direction(
        emission,
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        direction=-1,
        beam_width=24,
        smoothness=0.45,
        anchor_strength=6.0,
    )

    path = np.full(x1 - x0 + 1, np.nan, dtype=np.float32)
    path[right_x] = right_path
    path[left_x] = left_path

    if np.isnan(path).any():
        raise RuntimeError(f"{station}の全幅経路を復元できませんでした。")

    refined, observed = _refine_path(
        raw_distance,
        path,
        tolerance=anchor_tolerance,
        radius=line_radius,
    )

    # NHKカーソル帯は、灰色カーソルを「実測NHK」と誤認させない。
    # DPが左右のNHK線から求めた経路をそのまま採用する。
    if suppressed_cursor_band is not None:
        band_left, band_right = suppressed_cursor_band
        refined[band_left:band_right] = path[band_left:band_right]
        observed[band_left:band_right] = False

    # 非NHKの局色カーソル円は信頼できるため、中心を1点アンカーとして戻す。
    # 従来のNHKのように十数pxを平坦化せず、カーソル直近3pxだけを固定する。
    if (
        station != "NHK総合"
        and cursor_x is not None
        and cursor_y is not None
        and x0 <= cursor_x <= x1
    ):
        local_cursor_x = int(cursor_x - x0)
        left = max(0, local_cursor_x - 1)
        right = min(len(refined), local_cursor_x + 2)
        refined[left:right] = float(cursor_y - graph_top)
        observed[left:right] = True

    xs = np.arange(x0, x1 + 1)
    ys = refined + float(graph_top)

    missing_groups = _group_consecutive(np.where(~observed)[0], max_gap=1)
    max_gap_pixels = max((len(group) for group in missing_groups), default=0)
    coverage = float(observed.mean())

    quality = "A" if coverage >= 0.97 else ("B" if coverage >= 0.90 else "C")

    diagnostics: dict[str, float | int | str] = {
        "coverage": coverage,
        "max_gap_pixels": int(max_gap_pixels),
        "rejected_jumps": 0,
        "quality": quality,
        "tracking_method": "global-dp",
        "anchor": anchor_used,
        "inferred_pixels": int((~observed).sum()),
    }
    return xs, ys, observed, diagnostics

def analyze_image(
    arr: np.ndarray,
    end_time: datetime,
    duration_minutes: int = 180,
    tolerance: int = 42,
    grid_percent_step: float = 2.0,
) -> tuple[pd.DataFrame, Calibration, dict[str, dict[str, float | int | str]]]:
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("RGB画像を指定してください。")
    if duration_minutes <= 0:
        raise ValueError("表示時間は1分以上にしてください。")
    if grid_percent_step <= 0:
        raise ValueError("縦軸1目盛りは0より大きい値にしてください。")

    thin_lines = _find_thin_horizontal_lines(arr)
    grid, grid_gap = _select_regular_grid(thin_lines, arr.shape[0], min_lines=3)

    graph_top = int(grid[0])
    y_zero = int(grid[-1])
    pixels_per_percent = grid_gap / float(grid_percent_step)

    rough_left, rough_right = _find_rough_graph_x_bounds(arr, grid)
    (
        start_x,
        end_x,
        pixels_per_minute,
        time_axis_method,
        time_line_inliers,
    ) = _calibrate_time_axis(
        arr,
        grid,
        rough_left,
        rough_right,
        end_time,
        duration_minutes,
    )

    calibration = Calibration(
        graph_left=start_x,
        graph_right=end_x,
        graph_top=float(graph_top),
        graph_bottom=float(y_zero),
        y_zero=float(y_zero),
        pixels_per_percent=float(pixels_per_percent),
        pixels_per_minute=float(pixels_per_minute),
        start_time=end_time - timedelta(minutes=duration_minutes),
        end_time=end_time,
        duration_minutes=duration_minutes,
        grid_percent_step=float(grid_percent_step),
        grid_gap_pixels=float(grid_gap),
        time_axis_method=time_axis_method,
        time_line_inliers=time_line_inliers,
    )

    cursor_x = _find_cursor_x(arr, grid, rough_left, rough_right)

    traces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    diagnostics: dict[str, dict[str, float | int | str]] = {}

    for station, color in STATION_COLORS.items():
        cursor_y = None
        # 非NHKは局色のカーソル円を強いアンカーとして使える。
        # NHKはカーソル自体が灰色なので、誤アンカー防止のため使わない。
        if cursor_x is not None and station != "NHK総合":
            cursor_y = _find_cursor_station_y(
                arr,
                cursor_x,
                graph_top,
                y_zero,
                color,
                tolerance=max(tolerance, 20),
            )

        xs, ys, observed, station_diagnostics = _trace_station_global(
            arr,
            station,
            color,
            max(0, int(np.floor(start_x)) - 4),
            min(arr.shape[1] - 1, int(np.ceil(end_x)) + 4),
            graph_top,
            y_zero,
            tolerance,
            cursor_x=cursor_x,
            cursor_y=cursor_y,
        )

        gap_minutes = (
            float(station_diagnostics["max_gap_pixels"]) / pixels_per_minute
        )
        station_diagnostics["max_gap_minutes"] = round(gap_minutes, 2)

        coverage = float(station_diagnostics["coverage"])
        if coverage >= 0.97 and gap_minutes <= 2.0:
            station_diagnostics["quality"] = "A"
        elif coverage >= 0.90 and gap_minutes <= 5.0:
            station_diagnostics["quality"] = "B"
        else:
            station_diagnostics["quality"] = "C"

        traces[station] = (xs, ys, observed)
        diagnostics[station] = station_diagnostics

    times = [
        calibration.start_time + timedelta(minutes=minute)
        for minute in range(duration_minutes + 1)
    ]
    df = pd.DataFrame({"日時": times})

    # 計算用は丸め前を保持する。
    raw_station_values: dict[str, np.ndarray] = {}
    station_status: dict[str, list[str]] = {}

    for station, (xs, ys, observed) in traces.items():
        values = np.empty(duration_minutes + 1, dtype=float)
        statuses: list[str] = []

        for minute in range(duration_minutes + 1):
            x = start_x + minute * pixels_per_minute
            y = float(np.interp(x, xs, ys))
            rating = max(0.0, (y_zero - y) / pixels_per_percent)
            values[minute] = rating

            nearest = int(np.clip(np.argmin(np.abs(xs - x)), 0, len(xs) - 1))
            statuses.append("実測" if bool(observed[nearest]) else "経路推定")

        raw_station_values[station] = values
        station_status[station] = statuses
        df[station] = np.round(values, 1)

    stations = list(STATION_COLORS)
    raw_matrix = np.column_stack([raw_station_values[s] for s in stations])
    raw_total = raw_matrix.sum(axis=1)

    df["6局合計"] = np.round(raw_total, 1)
    df["その他5局合計"] = np.round(
        raw_total - raw_station_values["NHK総合"], 1
    )

    for station in stations:
        share = np.divide(
            raw_station_values[station],
            raw_total,
            out=np.full_like(raw_total, np.nan, dtype=float),
            where=raw_total != 0,
        ) * 100.0
        df[f"{station}_シェア"] = np.round(share, 2)

    raw_rank_df = pd.DataFrame(
        {station: raw_station_values[station] for station in stations}
    )
    ranks = raw_rank_df.rank(axis=1, ascending=False, method="min")
    for station in stations:
        df[f"{station}_順位"] = ranks[station].astype("Int64")

    # 解析の透明性のため状態列を末尾へ付加。
    for station in stations:
        df[f"{station}_状態"] = station_status[station]

    return df, calibration, diagnostics
