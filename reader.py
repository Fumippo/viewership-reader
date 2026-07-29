from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd


READER_VERSION = "7.0.1-hourline-layout-fix"

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
    """
    画像下端近くまで調べ、長い薄灰色の細線を抽出する。
    以前の0.90h打ち切りは廃止。0%線が95%地点でも拾う。
    """
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

    # グラフ線は通常1～2px。UIカードなどの太い帯は除外。
    thin_lines = [
        int(round(np.mean(group)))
        for group in groups
        if 1 <= len(group) <= 4
        and int(h * 0.18) <= int(round(np.mean(group))) <= int(h * 0.99)
    ]

    if len(thin_lines) < 5:
        raise RuntimeError(
            "水平グリッド線を十分に検出できませんでした。"
            "グラフ全体と0%線が入った画像を使ってください。"
        )

    return thin_lines


def _select_regular_grid(lines: list[int], image_height: int) -> tuple[list[int], float]:
    """
    UI由来の別水平線が混ざっても、最長の等間隔列だけを採用する。
    """
    ys = np.array(sorted(set(lines)), dtype=float)
    gap_candidates: list[float] = []

    for i in range(len(ys)):
        for j in range(i + 1, len(ys)):
            difference = ys[j] - ys[i]
            for steps in range(1, 10):
                gap = difference / steps
                if 45.0 <= gap <= 220.0:
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

            if len(matched_steps) < 5:
                continue

            step_groups = _group_consecutive(matched_steps, max_gap=1)
            longest_steps = max(step_groups, key=len)

            values: list[float] = []
            residual_sum = 0.0
            for step in longest_steps:
                target = anchor + step * gap
                index = int(np.argmin(np.abs(ys - target)))
                values.append(float(ys[index]))
                residual_sum += abs(float(ys[index]) - target)

            if len(values) < 5:
                continue

            span = max(values) - min(values)
            score = len(values) * 1000.0 + span - residual_sum

            if score > best_score:
                best_score = score
                best_values = values

    if best_values is None or len(best_values) < 5:
        raise RuntimeError("規則的な水平グリッドを特定できませんでした。")

    grid = sorted(set(int(round(value)) for value in best_values))
    gap = float(np.median(np.diff(grid)))

    # 2px程度の描画揺れは許すが、崩れた列は拒否。
    if np.max(np.abs(np.diff(grid) - gap)) > max(4.0, gap * 0.06):
        raise RuntimeError("水平グリッドの間隔が不規則です。")

    return grid, gap



def _find_rough_graph_x_bounds(arr: np.ndarray, grid: list[int]) -> tuple[int, int]:
    """
    上側の水平グリッド線からグラフ左右端を推定する。

    全水平線の共通部分を取る旧方式は、下側の線が番組線で分断されると
    右端を途中で切ってしまうため廃止。
    """
    h, w, _ = arr.shape
    spans: list[tuple[int, int]] = []

    # 上側は折れ線の交差が少なく、水平線が最も長く残りやすい。
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

        groups = _group_consecutive(xs, max_gap=18)
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
    """
    縦破線の局所ピークを抽出する。

    太い左端UIや選択カーソルを一本の時線として誤認しない。
    """
    roi = arr[graph_top:graph_bottom + 1]
    mask = _neutral_gray_mask(
        roi,
        max_spread=20,
        min_brightness=195,
        max_brightness=253,
    )
    column_score = mask.mean(axis=0)

    search_left = max(0, rough_left - 60)
    search_right = min(arr.shape[1] - 1, int(arr.shape[1] * 0.98))

    candidates: list[tuple[int, float]] = []
    for x in range(max(1, search_left), min(len(column_score) - 2, search_right) + 1):
        score = float(column_score[x])
        if not (0.30 <= score <= 0.79):
            continue
        if score >= column_score[x - 1] and score >= column_score[x + 1]:
            candidates.append((x, score))

    selected: list[tuple[int, float]] = []
    for x, score in sorted(candidates, key=lambda item: item[1], reverse=True):
        if all(abs(x - old_x) > 20 for old_x, _ in selected):
            selected.append((x, score))

    return sorted(x for x, _ in selected)

def _select_hour_progression(peaks: list[int]) -> list[int]:
    if len(peaks) < 2:
        return []

    best_score = -1e18
    best: list[int] = []

    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            gap = peaks[j] - peaks[i]
            if gap < 200:
                continue

            tolerance = max(4.0, gap * 0.015)
            matched: list[int] = []

            for x in peaks:
                step = round((x - peaks[i]) / gap)
                target = peaks[i] + step * gap
                if abs(x - target) <= tolerance:
                    matched.append(x)

            matched = sorted(set(matched))
            if len(matched) < 2:
                continue

            score = len(matched) * 1000.0 + (matched[-1] - matched[0])
            if score > best_score:
                best_score = score
                best = matched

    return best




def _calibrate_time_axis(
    arr: np.ndarray,
    grid: list[int],
    rough_left: int,
    rough_right: int,
    end_time: datetime,
    duration_minutes: int,
) -> tuple[float, float, float]:
    """
    検出時線を、表示期間内に存在する正時の候補へ割り当てる。

    最初の正時線が選択カーソルと重なって消えていても、17:00・18:00
    など残った時線から開始位置を復元する。
    """
    peaks = _find_vertical_hour_lines(
        arr,
        grid[0],
        grid[-1],
        rough_left,
        rough_right,
    )
    hours = _select_hour_progression(peaks)

    if len(hours) < 2:
        raise RuntimeError("縦の時刻目盛りを2本以上検出できませんでした。")

    hour_spacing = float(np.median(np.diff(hours)))
    pixels_per_minute = hour_spacing / 60.0

    start_time = end_time - timedelta(minutes=duration_minutes)
    first_hour_offset = (-start_time.minute) % 60
    expected_offsets = list(
        range(first_hour_offset, duration_minutes + 1, 60)
    )

    best: tuple[float, float] | None = None

    for detected_x in hours:
        for offset in expected_offsets:
            candidate_start = detected_x - offset * pixels_per_minute
            predicted = np.array(
                [
                    candidate_start + expected_offset * pixels_per_minute
                    for expected_offset in expected_offsets
                ],
                dtype=float,
            )

            residual = sum(
                float(np.min(np.abs(predicted - hour_x)))
                for hour_x in hours
            )
            score = residual + 0.15 * abs(candidate_start - rough_left)

            candidate_end = (
                candidate_start
                + duration_minutes * pixels_per_minute
            )
            if candidate_start < -30 or candidate_end > arr.shape[1] + 30:
                score += 1000.0

            if best is None or score < best[0]:
                best = (score, candidate_start)

    if best is None:
        raise RuntimeError("縦の時刻目盛りから時間軸を較正できませんでした。")

    start_x = float(best[1])
    end_x = start_x + duration_minutes * pixels_per_minute

    # 時線同士が予測位置に並ぶことを最終確認。
    predicted = np.array(
        [
            start_x + expected_offset * pixels_per_minute
            for expected_offset in expected_offsets
        ],
        dtype=float,
    )
    max_residual = max(
        float(np.min(np.abs(predicted - hour_x)))
        for hour_x in hours
    )
    if max_residual > max(7.0, hour_spacing * 0.025):
        raise RuntimeError("縦の時刻目盛りの間隔が不規則です。")

    return start_x, end_x, pixels_per_minute

def _station_color_mask(
    roi: np.ndarray,
    station: str,
    target_rgb: Tuple[int, int, int],
    tolerance: int,
) -> np.ndarray:
    work = roi.astype(np.int16)
    target = np.array(target_rgb, dtype=np.int16)
    distance = np.max(np.abs(work - target), axis=2)

    if station == "NHK総合":
        # JPEG圧縮されたNHK灰色も拾う一方、明るいグリッド線は除外。
        spread = work.max(axis=2) - work.min(axis=2)
        brightness = work.mean(axis=2)
        return (
            (distance <= 48)
            & (brightness < 230)
            & (spread <= 55)
        )

    return distance <= tolerance


def _trace_station(
    arr: np.ndarray,
    station: str,
    color: Tuple[int, int, int],
    trace_left: int,
    trace_right: int,
    graph_top: int,
    graph_bottom: int,
    pixels_per_percent: float,
    tolerance: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """
    時間軸の中央付近から左右へ追跡する。

    左端の選択カーソル丸を開始点にして別の灰色へ張り付く事故を防ぐ。
    """
    x0 = max(0, int(trace_left))
    x1 = min(arr.shape[1] - 1, int(trace_right))

    roi = arr[
        graph_top:graph_bottom + 1,
        x0:x1 + 1,
    ]
    mask = _station_color_mask(roi, station, color, tolerance)

    column_counts = mask.sum(axis=0)

    # NHKの灰色カーソル丸は周辺列まで灰色が多いため、専用の低い上限を使う。
    # これでカーソル円周を折れ線として読む事故を防ぐ。
    vertical_limit = (
        24
        if station == "NHK総合"
        else max(80, int(mask.shape[0] * 0.15))
    )
    mask[:, column_counts > vertical_limit] = False

    xs = np.arange(x0, x1 + 1)
    ys = np.full(len(xs), np.nan)
    center = len(xs) // 2

    anchor_index: int | None = None
    anchor_y: float | None = None

    # 中央から外へ探し、候補が一つだけの安定した列を開始点にする。
    for radius in range(len(xs)):
        test_indices = [center - radius, center + radius]
        for index in test_indices:
            if index < 0 or index >= len(xs):
                continue

            positions = np.where(mask[:, index])[0]
            if len(positions) == 0:
                continue

            groups = _group_consecutive(positions, max_gap=1)
            strong = [group for group in groups if len(group) >= 2]

            if len(strong) == 1:
                anchor_index = index
                anchor_y = float(np.median(strong[0])) + graph_top
                break

        if anchor_index is not None:
            break

    if anchor_index is None or anchor_y is None:
        raise RuntimeError(f"{station}の追跡開始点を特定できませんでした。")

    ys[anchor_index] = anchor_y
    rejected_jumps = 0
    max_single_column_jump = max(70.0, pixels_per_percent * 1.5)

    for direction in (1, -1):
        previous_y = anchor_y
        index = anchor_index + direction

        while 0 <= index < len(xs):
            positions = np.where(mask[:, index])[0]

            if len(positions):
                groups = _group_consecutive(positions, max_gap=1)
                candidates = [
                    (
                        float(np.median(group)) + graph_top,
                        len(group),
                    )
                    for group in groups
                ]

                chosen = min(
                    candidates,
                    key=lambda candidate:
                        abs(candidate[0] - previous_y)
                        - min(candidate[1], 20) * 0.1,
                )

                if abs(chosen[0] - previous_y) <= max_single_column_jump:
                    ys[index] = chosen[0]
                    previous_y = chosen[0]
                else:
                    rejected_jumps += 1

            index += direction

    good = ~np.isnan(ys)
    coverage = float(good.mean())

    missing_groups = _group_consecutive(np.where(~good)[0], max_gap=1)
    max_gap_pixels = max((len(group) for group in missing_groups), default=0)

    if coverage < 0.85:
        raise RuntimeError(
            f"{station}の線検出率が{coverage * 100:.1f}%しかありません。"
            "この画像では安全に数値化できません。"
        )

    if max_gap_pixels > 60:
        raise RuntimeError(
            f"{station}で{max_gap_pixels}pxの連続欠損が発生しました。"
            "長距離補間を避けるため解析を中止しました。"
        )

    interpolated = np.interp(xs, xs[good], ys[good])

    diagnostics: dict[str, float | int] = {
        "coverage": coverage,
        "max_gap_pixels": int(max_gap_pixels),
        "rejected_jumps": int(rejected_jumps),
    }
    return xs, interpolated, diagnostics

def analyze_image(
    arr: np.ndarray,
    end_time: datetime,
    duration_minutes: int = 180,
    tolerance: int = 42,
) -> tuple[pd.DataFrame, Calibration, dict[str, dict[str, float | int]]]:
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("RGB画像を指定してください。")

    thin_lines = _find_thin_horizontal_lines(arr)
    grid, grid_gap = _select_regular_grid(thin_lines, arr.shape[0])

    graph_top = int(grid[0])
    y_zero = int(grid[-1])
    pixels_per_percent = grid_gap / 2.0

    rough_left, rough_right = _find_rough_graph_x_bounds(arr, grid)
    start_x, end_x, pixels_per_minute = _calibrate_time_axis(
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
    )

    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    diagnostics: dict[str, dict[str, float | int]] = {}

    for station, color in STATION_COLORS.items():
        xs, ys, station_diagnostics = _trace_station(
            arr,
            station,
            color,
            max(0, int(np.floor(start_x)) - 4),
            min(arr.shape[1] - 1, int(np.ceil(end_x)) + 4),
            graph_top,
            y_zero,
            pixels_per_percent,
            tolerance,
        )
        traces[station] = (xs, ys)
        diagnostics[station] = station_diagnostics

    times = [
        calibration.start_time + timedelta(minutes=minute)
        for minute in range(duration_minutes + 1)
    ]
    df = pd.DataFrame({"日時": times})

    for station, (xs, ys) in traces.items():
        values: list[float] = []

        for minute in range(duration_minutes + 1):
            x = start_x + minute * pixels_per_minute
            y = float(np.interp(x, xs, ys))
            rating = (y_zero - y) / pixels_per_percent
            values.append(round(max(0.0, rating), 1))

        df[station] = values

    stations = list(STATION_COLORS)
    df["6局合計"] = df[stations].sum(axis=1).round(1)
    df["その他5局合計"] = (df["6局合計"] - df["NHK総合"]).round(1)

    for station in stations:
        df[f"{station}_シェア"] = (
            df[station]
            / df["6局合計"].replace(0, np.nan)
            * 100
        ).round(2)

    ranks = df[stations].rank(axis=1, ascending=False, method="min")
    for station in stations:
        df[f"{station}_順位"] = ranks[station].astype("Int64")

    return df, calibration, diagnostics
