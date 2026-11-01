from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Tuple
import numpy as np
import pandas as pd
READER_VERSION='9.2.0-optimized-trace'  # App.pyとの互換性を維持
READER_PATCH_VERSION='9.2.1-adaptive-range'
STRICT_EVIDENCE_THRESHOLD=24.0
RELAXED_EVIDENCE_THRESHOLD=35.0
HIGH_RANGE_PERCENT=18.0
# 動作は固定値だけで決めない。高レンジかつ色証拠が弱い場合だけ緩和する。
STATION_COLORS: Dict[str, Tuple[int,int,int]]={
 'NHK総合':(182,188,195),'日本テレビ':(254,216,120),'テレビ朝日':(246,162,228),
 'TBS':(100,148,240),'テレビ東京':(89,113,175),'フジテレビ':(243,124,124)}
# 旧9.2版と完全に同じ27列・同じ順番を守る。診断情報はCSV本体に追加しない。
CSV_COLUMNS = (
    ['日時'] + list(STATION_COLORS) + ['6局合計', 'その他5局合計']
    + [f'{station}_シェア' for station in STATION_COLORS]
    + [field for station in STATION_COLORS for field in (f'{station}_順位', f'{station}_状態')]
)
DEFAULT_CURSOR_RGB_TOLERANCE=20
DIRECT_EVIDENCE_THRESHOLD=STRICT_EVIDENCE_THRESHOLD  # App.pyの表示用。自動調整後の実使用閾値は診断表に表示。
FALLBACK_COLOR_TOLERANCE=50
REFERENCE_PIXELS_PER_PERCENT=20.0
OTHER_SMOOTHNESS_AT_REFERENCE=0.50
NHK_SMOOTHNESS_AT_REFERENCE=0.62
@dataclass(frozen=True)
class Calibration:
    graph_left:float
    graph_right:float
    graph_top:float
    graph_bottom:float
    y_zero:float
    pixels_per_percent:float
    pixels_per_minute:float
    start_time:datetime
    end_time:datetime
    duration_minutes:int
    grid_percent_step:float
    grid_gap_pixels:float
    time_axis_method:str='unknown'
    time_line_inliers:int=0

def _group_consecutive(values,max_gap=1):
    groups=[]
    for raw in values:
        value=int(raw)
        if not groups or value-groups[-1][-1]>max_gap:groups.append([value])
        else:groups[-1].append(value)
    return groups

def _neutral_gray_mask(pixels,*,max_spread,min_brightness,max_brightness):
    work=pixels.astype(np.int16)
    spread=work.max(axis=2)-work.min(axis=2)
    brightness=work.mean(axis=2)
    return (spread<=max_spread)&(brightness>=min_brightness)&(brightness<=max_brightness)

def _find_thin_horizontal_lines(arr):
    h,w,_=arr.shape
    crop=arr[:,int(w*.08):int(w*.95)]
    mask=_neutral_gray_mask(crop,max_spread=20,min_brightness=195,max_brightness=253)
    candidates=np.where(mask.mean(axis=1)>=.65)[0]
    groups=_group_consecutive(candidates,max_gap=1)
    thin_lines=[int(round(np.mean(g))) for g in groups
                if 1<=len(g)<=max(7, min(12, int(h * .006))) and int(h*.18)<=int(round(np.mean(g)))<=int(h*.99)] # JPEG線の太さを画像高さに応じて許容
    if len(thin_lines)<3:raise RuntimeError('not enough horizontal gridlines')
    return thin_lines

def _select_regular_grid(lines,image_height,min_lines=3):
    ys=np.array(sorted(set(lines)),dtype=float)
    if len(ys)<min_lines:raise RuntimeError('not enough grid')
    min_gap=max(20.,image_height*.015)
    max_gap=max(min_gap+1.,image_height*.45)
    gap_candidates=[]
    for i in range(len(ys)):
        for j in range(i+1,len(ys)):
            difference=ys[j]-ys[i]
            for steps in range(1,13):
                gap=difference/steps
                if min_gap<=gap<=max_gap:gap_candidates.append(float(gap))
    best_score=-1e18; best_values=None
    for gap in gap_candidates:
        tolerance=max(3.,gap*.035)
        for anchor in ys:
            step_numbers=np.round((ys-anchor)/gap)
            residuals=np.abs(ys-(anchor+step_numbers*gap))
            matched_steps=np.unique(step_numbers[residuals<=tolerance].astype(int))
            if len(matched_steps)<min_lines:continue
            longest_steps=max(_group_consecutive(matched_steps,max_gap=1),key=len)
            if len(longest_steps)<min_lines:continue
            values=[];residual_sum=0.
            for step in longest_steps:
                target=anchor+step*gap
                index=int(np.argmin(np.abs(ys-target)))
                values.append(float(ys[index]));residual_sum+=abs(float(ys[index])-target)
            score=len(values)*1000.+(max(values)-min(values))-residual_sum*10.
            if score>best_score:best_score,best_values=score,values
    if best_values is None:raise RuntimeError('no regular grid')
    grid=sorted(set(int(round(v)) for v in best_values))
    gap=float(np.median(np.diff(grid)))
    if np.max(np.abs(np.diff(grid)-gap))>max(4.,gap*.06):raise RuntimeError('irregular grid')
    return grid,gap

def _find_rough_graph_x_bounds(arr,grid):
    _,w,_=arr.shape;spans=[]
    for y in grid[:min(3,len(grid))]:
        mask=_neutral_gray_mask(arr[y:y+1],max_spread=25,min_brightness=190,max_brightness=253)[0]
        xs=np.where(mask)[0]
        if len(xs)==0:continue
        groups=_group_consecutive(xs,max_gap=max(6,int(w*.008)))
        long_groups=[g for g in groups if g[-1]-g[0]+1>=max(80,int(w*.05))]
        if not long_groups:continue
        left=min(g[0] for g in long_groups);right=max(g[-1] for g in long_groups)
        if right-left>=w*.55:spans.append((left,right))
    if not spans:raise RuntimeError('no rough x')
    left=int(round(np.median([s[0] for s in spans])))
    right=int(round(np.median([s[1] for s in spans])))
    if right-left<w*.55:raise RuntimeError('x too narrow')
    return left,right

def _find_vertical_hour_lines(arr,graph_top,graph_bottom,rough_left,rough_right):
    roi=arr[graph_top:graph_bottom+1]
    mask=_neutral_gray_mask(roi,max_spread=20,min_brightness=195,max_brightness=253)
    scores=mask.mean(axis=0);w=arr.shape[1]
    search_left=max(0,rough_left-max(30,int(w*.03)))
    search_right=min(w-1,int(w*.98))
    nms_distance=max(10,int(w*.008))
    candidates=[]
    for x in range(max(1,search_left),min(len(scores)-2,search_right)+1):
        score=float(scores[x])
        if .30<=score<=.79 and score>=scores[x-1] and score>=scores[x+1]:candidates.append((x,score))
    selected=[]
    for x,score in sorted(candidates,key=lambda item:item[1],reverse=True):
        if all(abs(x-old_x)>nms_distance for old_x,_ in selected):selected.append((x,score))
    return sorted(x for x,_ in selected)

def _match_time_axis_model(peaks,expected_offsets,rough_left,rough_right,duration_minutes,image_width):
    if len(peaks)<2 or len(expected_offsets)<2:raise RuntimeError('no hourlines')
    rough_ppm=(rough_right-rough_left)/float(duration_minutes)
    rough_hour_spacing=rough_ppm*60.;max_hour_steps=max(1,int(np.ceil(duration_minutes/60.)))
    spacing_candidates=[rough_hour_spacing];min_peak_difference=max(60.,image_width*.04)
    for i in range(len(peaks)):
        for j in range(i+1,len(peaks)):
            difference=float(peaks[j]-peaks[i])
            if difference<min_peak_difference:continue
            for hour_steps in range(1,max_hour_steps+1):
                spacing=difference/hour_steps
                if rough_hour_spacing*.55<=spacing<=rough_hour_spacing*1.45:spacing_candidates.append(spacing)
    best=None
    for hour_spacing in spacing_candidates:
        ppm=hour_spacing/60.
        tolerance=max(7.,hour_spacing*.035)
        for peak in peaks:
            for offset in expected_offsets:
                candidate_start=float(peak)-float(offset)*ppm
                predicted=np.array([candidate_start+expected*ppm for expected in expected_offsets],dtype=float)
                pairs=[]
                for pred_index,predicted_x in enumerate(predicted):
                    for peak_index,detected_x in enumerate(peaks):
                        distance=abs(float(detected_x)-float(predicted_x))
                        if distance<=tolerance:pairs.append((distance,pred_index,peak_index,float(detected_x)))
                pairs.sort(key=lambda item:item[0]);used_pred=set();used_peak=set();matches=[];residual_sum=0.
                for distance,pred_index,peak_index,detected_x in pairs:
                    if pred_index in used_pred or peak_index in used_peak:continue
                    used_pred.add(pred_index);used_peak.add(peak_index)
                    matches.append((expected_offsets[pred_index],detected_x));residual_sum+=distance
                if len(matches)<2:continue
                candidate_end=candidate_start+duration_minutes*ppm
                boundary_penalty=abs(candidate_start-rough_left)+abs(candidate_end-rough_right)
                spacing_penalty=abs(ppm-rough_ppm)*60.
                score=len(matches)*100000.-residual_sum*100.-boundary_penalty*2.-spacing_penalty*10.
                if candidate_start< -40 or candidate_end>image_width+40:score-=100000.
                if best is None or score>best['score']:
                    best={'score':score,'matches':matches,'start':candidate_start,'ppm':ppm}
    if best is None:raise RuntimeError('no x calibration')
    matches=list(best['matches']);offsets=np.array([m[0] for m in matches],dtype=float);detected=np.array([m[1] for m in matches],dtype=float)
    design=np.column_stack([np.ones(len(offsets)),offsets])
    start_x,pixels_per_minute=np.linalg.lstsq(design,detected,rcond=None)[0]
    start_x,pixels_per_minute=float(start_x),float(pixels_per_minute)
    end_x=start_x+duration_minutes*pixels_per_minute
    residuals=np.abs(start_x+offsets*pixels_per_minute-detected)
    hour_spacing=pixels_per_minute*60.
    if float(np.max(residuals))>max(8.,hour_spacing*.04):raise RuntimeError('large residual')
    if start_x<rough_left-max(50.,rough_hour_spacing*.22) or end_x>rough_right+max(50.,rough_hour_spacing*.22):raise RuntimeError('outside bounds')
    matched_peaks={int(round(m[1])) for m in matches}
    return start_x,end_x,pixels_per_minute,matches,[p for p in peaks if p not in matched_peaks]

def _color_distance(roi,target_rgb):
    work=roi.astype(np.int16);target=np.array(target_rgb,dtype=np.int16)
    return np.max(np.abs(work-target),axis=2)

def _station_color_mask(roi,station,target_rgb,tolerance):
    distance=_color_distance(roi,target_rgb)
    if station=='NHK総合':
        work=roi.astype(np.int16);spread=work.max(axis=2)-work.min(axis=2);brightness=work.mean(axis=2)
        return (distance<=max(55,tolerance))&(brightness<220)&(spread<=30)
    return distance<=tolerance

def _find_colored_plot_bounds(arr,graph_top,graph_bottom,rough_left,rough_right,tolerance=FALLBACK_COLOR_TOLERANCE):
    x0=max(0,rough_left-120);x1=min(arr.shape[1]-1,rough_right+120)
    roi=arr[graph_top:graph_bottom+1,x0:x1+1]
    left_edges=[];right_edges=[]
    for station,color in STATION_COLORS.items():
        mask=_station_color_mask(roi,station,color,tolerance)
        columns=np.where(mask.sum(axis=0)>0)[0]
        if len(columns)<10:continue
        groups=_group_consecutive(columns,max_gap=8);longest=max(groups,key=lambda g:g[-1]-g[0]+1)
        if longest[-1]-longest[0]+1<arr.shape[1]*.25:continue
        left_edges.append(x0+longest[0]);right_edges.append(x0+longest[-1])
    if len(left_edges)<3:raise RuntimeError('no colored bounds')
    left=int(round(float(np.median(left_edges))));right=int(round(float(np.median(right_edges))))
    if right-left<arr.shape[1]*.45:raise RuntimeError('narrow colored bounds')
    return left,right

def _calibrate_time_axis(arr,grid,rough_left,rough_right,end_time,duration_minutes):
    peaks=_find_vertical_hour_lines(arr,grid[0],grid[-1],rough_left,rough_right)
    start_time=end_time-timedelta(minutes=duration_minutes)
    first_hour_offset=(-start_time.minute)%60
    expected_offsets=list(range(first_hour_offset,duration_minutes+1,60))
    if len(peaks)>=2 and len(expected_offsets)>=2:
        try:
            start_x,end_x,ppm,matches,_=_match_time_axis_model(peaks,expected_offsets,rough_left,rough_right,duration_minutes,arr.shape[1])
            return float(start_x),float(end_x),float(ppm),'hour-lines',len(matches)
        except RuntimeError:pass
    color_left,color_right=_find_colored_plot_bounds(arr,grid[0],grid[-1],rough_left,rough_right)
    start_x,end_x=float(color_left),float(color_right);ppm=(end_x-start_x)/float(duration_minutes)
    if ppm<=0:raise RuntimeError('negative ppm')
    return start_x,end_x,ppm,'color-range-fallback',0

def _find_cursor_x(arr,graph_top,graph_bottom,plot_left,plot_right):
    roi=arr[graph_top:graph_bottom+1]
    mask=_neutral_gray_mask(roi,max_spread=35,min_brightness=150,max_brightness=253)
    scores=mask.mean(axis=0);x_axis=np.arange(arr.shape[1])
    candidates=np.where((scores>=.82)&(x_axis>=int(np.floor(plot_left))-10)&(x_axis<=int(np.ceil(plot_right))+10))[0]
    groups=_group_consecutive(candidates,max_gap=2)
    thin_groups=[g for g in groups if 1<=len(g)<=8]
    if not thin_groups:return None
    best_global=max(float(np.max(scores[g])) for g in thin_groups)
    # JPEG縮小後は、19:00目盛り線の灰色一致率がカーソルを0.4%程度上回ることがある。
    # 右端近傍にほぼ同程度の縦線があれば、画面更新時刻のカーソルを優先する。
    near_right=[g for g in thin_groups
                if abs(float(np.mean(g))-float(plot_right)) <= max(25.,(plot_right-plot_left)*.025)
                and float(np.max(scores[g])) >= best_global-.025]
    chosen=max(near_right,key=lambda g:float(np.max(scores[g]))) if near_right else max(thin_groups,key=lambda g:float(np.max(scores[g])))
    return int(round(float(np.mean(chosen))))

def _find_cursor_station_y(arr,cursor_x,graph_top,graph_bottom,target_rgb,tolerance):
    window=14
    x0,x1=max(0,cursor_x-window),min(arr.shape[1]-1,cursor_x+window)
    roi=arr[graph_top:graph_bottom+1,x0:x1+1]
    row_counts=(_color_distance(roi,target_rgb)<=tolerance).sum(axis=1)
    maximum=int(row_counts.max())
    if maximum<2:return None
    threshold=max(2,int(np.ceil(maximum*.45)))
    groups=_group_consecutive(np.where(row_counts>=threshold)[0],max_gap=2)
    if not groups:return None
    chosen=max(groups,key=lambda g:int(row_counts[g].sum()))
    y_values=np.array(chosen,dtype=float);weights=row_counts[chosen].astype(float)
    return float(np.average(y_values,weights=weights)+graph_top)

def _precompute_station_evidence_costs(roi,*,alpha_penalty=35.):
    work=roi.astype(np.float32)
    r,g,b=work[:,:,0],work[:,:,1],work[:,:,2]
    stations=list(STATION_COLORS)
    h,w=roi.shape[:2]
    stack=np.empty((len(stations),h,w),dtype=np.float32)
    smallest=np.full((h,w),np.inf,dtype=np.float32)
    second=np.full((h,w),np.inf,dtype=np.float32)
    argmin=np.full((h,w),-1,dtype=np.int8)
    for i,(station,target_rgb) in enumerate(STATION_COLORS.items()):
        d0=np.float32(target_rgb[0]-255);d1=np.float32(target_rgb[1]-255);d2=np.float32(target_rgb[2]-255)
        denominator=np.float32(d0*d0+d1*d1+d2*d2)
        if denominator<=1e-9:raise ValueError('white color')
        alpha=((r-255.)*d0+(g-255.)*d1+(b-255.)*d2)/denominator
        np.clip(alpha,0.,1.,out=alpha)
        residual=np.abs(r-(255.+alpha*d0))
        residual=np.maximum(residual,np.abs(g-(255.+alpha*d1)))
        residual=np.maximum(residual,np.abs(b-(255.+alpha*d2)))
        cost=(residual+np.float32(alpha_penalty)*(1.-alpha)).astype(np.float32,copy=False)
        stack[i]=cost
        better=cost<smallest
        second[better]=smallest[better]
        np.minimum(second,cost,out=second,where=~better)
        smallest[better]=cost[better]
        argmin[better]=i
    spread=(roi.max(axis=2).astype(np.int16)-roi.min(axis=2).astype(np.int16)).astype(np.float32)
    for i,station in enumerate(stations):
        own=stack[i]
        min_other=np.where(argmin==i,second,smallest)
        own+=np.maximum(0.,own-min_other+3.)*4.
        if station=='NHK総合':own+=np.maximum(0.,spread-25.)*1.6
    return {station:stack[i] for i,station in enumerate(stations)}

def _vertical_local_min(distance,radius):
    if radius<=0:return distance.astype(np.float32)
    h,_=distance.shape
    padded=np.pad(distance.astype(np.float32),((radius,radius),(0,0)),mode='constant',constant_values=255.)
    return np.minimum.reduce([padded[offset:offset+h] for offset in range(radius*2+1)])

def _choose_anchor_from_color(local_distance,*,preferred_x,tolerance):
    _,w=local_distance.shape
    preferred_x=int(np.clip(preferred_x,0,w-1))
    for radius in range(w):
        for x in (preferred_x-radius,preferred_x+radius):
            if x<0 or x>=w:continue
            column=local_distance[:,x];y=int(np.argmin(column))
            if float(column[y])<=tolerance:return x,float(y)
    y=int(np.argmin(local_distance[:,preferred_x]))
    if float(local_distance[y,preferred_x])>max(90.,tolerance*2.):raise RuntimeError('no stable anchor')
    return preferred_x,float(y)

def _beam_trace_direction(emission,*,anchor_x,anchor_y,direction,beam_width,smoothness,anchor_strength):
    h,w=emission.shape
    if direction not in (-1,1):raise ValueError('direction')
    columns=list(range(anchor_x,w if direction==1 else -1,direction))
    n,k=len(columns),min(beam_width,h)
    beam_y=np.empty((n,k),dtype=np.int16)
    parent_dtype=np.int8 if k<=127 else np.int16
    parents=np.zeros((n,k),dtype=parent_dtype)
    y_axis_1d=np.arange(h,dtype=np.float32)
    first_cost=emission[:,columns[0]].astype(np.float32)+anchor_strength*np.abs(y_axis_1d-float(anchor_y))
    selected=np.argpartition(first_cost,k-1)[:k]
    selected=selected[np.argsort(first_cost[selected])]
    beam_y[0]=selected
    costs=first_cost[selected].astype(np.float32);costs-=costs.min()
    y_axis=y_axis_1d[:,None]
    for step_index in range(1,n):
        x=columns[step_index]
        prev_y=beam_y[step_index-1].astype(np.float32)
        transition=costs[None,:]+smoothness*np.abs(y_axis-prev_y[None,:])
        best_parent=np.argmin(transition,axis=1)
        best_cost=transition[np.arange(h),best_parent]+emission[:,x]
        selected=np.argpartition(best_cost,k-1)[:k]
        selected=selected[np.argsort(best_cost[selected])]
        beam_y[step_index]=selected
        parents[step_index]=best_parent[selected]
        costs=best_cost[selected].astype(np.float32);costs-=costs.min()
    slot=int(np.argmin(costs));path_local=np.empty(n,dtype=np.float32)
    for step_index in range(n-1,-1,-1):
        path_local[step_index]=float(beam_y[step_index,slot])
        if step_index:slot=int(parents[step_index,slot])
    return np.array(columns,dtype=int),path_local

def _refine_path(evidence_cost,path,*,direct_threshold,radius):
    h,w=evidence_cost.shape
    refined=path.astype(np.float32).copy();observed=np.zeros(w,dtype=bool)
    search_radius=max(3,radius+3)
    for x in range(w):
        center=int(round(float(path[x])))
        y0,y1=max(0,center-search_radius),min(h-1,center+search_radius)
        rows=np.arange(y0,y1+1);costs=evidence_cost[y0:y1+1,x]
        good=rows[costs<=direct_threshold]
        if not len(good):continue
        groups=_group_consecutive(good,max_gap=1)
        def group_score(group):
            group_arr=np.array(group,dtype=int)
            local_cost=float(np.mean(evidence_cost[group_arr,x]));center_distance=abs(float(np.mean(group_arr))-float(path[x]))
            return local_cost+center_distance*1.5-min(len(group),6)*.35
        chosen=min(groups,key=group_score)
        refined[x]=float(np.mean(chosen));observed[x]=True
    return refined,observed

def _trace_station_global(arr,station,evidence_cost,trace_left,trace_right,graph_top,graph_bottom,pixels_per_percent,cursor_x=None,cursor_y=None,grid_rows=None, *, evidence_threshold=STRICT_EVIDENCE_THRESHOLD, nhk_chroma_guard=False):
    x0=max(0,int(trace_left));x1=min(arr.shape[1]-1,int(trace_right))
    roi=arr[graph_top:graph_bottom+1,x0:x1+1]
    if evidence_cost.shape!=roi.shape[:2]:raise RuntimeError('evidence roi mismatched')
    line_radius=max(1,min(4,int(round((graph_bottom-graph_top+1)/400.))))
    local_cost=_vertical_local_min(evidence_cost,line_radius)
    emission=np.minimum(local_cost.astype(np.float32),150.)
    direct_threshold=float(evidence_threshold)
    preferred_local_x=(x1-x0)//2
    suppressed_cursor_band=None
    if station=='NHK総合':
        if grid_rows:
            grid_radius=max(2,int(round(arr.shape[0]*.0015)))
            for global_y in grid_rows:
                local_y=int(global_y-graph_top)
                y0,y1=max(0,local_y-grid_radius),min(emission.shape[0],local_y+grid_radius+1)
                if y0<y1:emission[y0:y1,:]+=38.
        work=roi.astype(np.int16)
        brightness=work.mean(axis=2)
        spread=work.max(axis=2)-work.min(axis=2)
        emission+=((brightness>215)&(spread<=18)).astype(np.float32)*24.
        if nhk_chroma_guard:
            emission+=np.maximum(0.,spread.astype(np.float32)-12.)*2.5 # 高レンジ・灰色線の青線誤追跡対策
        if cursor_x is not None and x0<=cursor_x<=x1:
            local_cursor_x=int(cursor_x-x0)
            suppress_radius=max(4,int(round(arr.shape[1]*.0025)))
            band_left=max(0,local_cursor_x-suppress_radius)
            band_right=min(emission.shape[1],local_cursor_x+suppress_radius+1)
            emission[:,band_left:band_right]=np.maximum(emission[:,band_left:band_right],85.)
            suppressed_cursor_band=(band_left,band_right)
            if band_left<=preferred_local_x<band_right:
                preferred_local_x=min(emission.shape[1]-1,band_right+max(6,suppress_radius))
    anchor_used='color'
    if station!='NHK総合' and cursor_x is not None and cursor_y is not None and x0<=cursor_x<=x1:
        anchor_x=int(cursor_x-x0);anchor_y=float(cursor_y-graph_top);anchor_used='cursor'
    else:
        anchor_x,anchor_y=_choose_anchor_from_color(local_cost,preferred_x=preferred_local_x,tolerance=int(round(direct_threshold)))
        if suppressed_cursor_band is not None and suppressed_cursor_band[0]<=anchor_x<suppressed_cursor_band[1]:
            retry_x=min(emission.shape[1]-1,suppressed_cursor_band[1]+max(6,suppressed_cursor_band[1]-suppressed_cursor_band[0]))
            anchor_x,anchor_y=_choose_anchor_from_color(local_cost,preferred_x=retry_x,tolerance=int(round(direct_threshold)))
    base_smoothness=NHK_SMOOTHNESS_AT_REFERENCE if station=='NHK総合' else OTHER_SMOOTHNESS_AT_REFERENCE
    scale=REFERENCE_PIXELS_PER_PERCENT/max(float(pixels_per_percent),1e-6)
    smoothness=base_smoothness*float(np.clip(scale,.60,1.80))
    beam_width=32 if station=='NHK総合' else 24
    right_x,right_path=_beam_trace_direction(emission,anchor_x=anchor_x,anchor_y=anchor_y,direction=1,beam_width=beam_width,smoothness=smoothness,anchor_strength=7.)
    left_x,left_path=_beam_trace_direction(emission,anchor_x=anchor_x,anchor_y=anchor_y,direction=-1,beam_width=beam_width,smoothness=smoothness,anchor_strength=7.)
    path=np.full(x1-x0+1,np.nan,dtype=np.float32)
    path[right_x]=right_path;path[left_x]=left_path
    if np.isnan(path).any():raise RuntimeError('path missing')
    refined,observed=_refine_path(evidence_cost,path,direct_threshold=direct_threshold,radius=line_radius)
    if suppressed_cursor_band is not None:
        band_left,band_right=suppressed_cursor_band
        refined[band_left:band_right]=path[band_left:band_right]
        observed[band_left:band_right]=False
    if station!='NHK総合' and cursor_x is not None and cursor_y is not None and x0<=cursor_x<=x1:
        local_cursor_x=int(cursor_x-x0);left,right=max(0,local_cursor_x-1),min(len(refined),local_cursor_x+2)
        refined[left:right]=float(cursor_y-graph_top);observed[left:right]=True
    # 閾値35の「弱い色証拠」と、元の閾値24の「強い色証拠」を区別して診断。
    _, strict_observed = _refine_path(evidence_cost,refined,direct_threshold=STRICT_EVIDENCE_THRESHOLD,radius=line_radius)
    _, relaxed_observed = _refine_path(evidence_cost,refined,direct_threshold=RELAXED_EVIDENCE_THRESHOLD,radius=line_radius)
    strict_gaps = _group_consecutive(np.where(~strict_observed)[0], max_gap=1)
    strict_max_gap_pixels = max((len(g) for g in strict_gaps), default=0)
    xs=np.arange(x0,x1+1);ys=refined+float(graph_top)
    inferred_groups=_group_consecutive(np.where(~observed)[0],max_gap=1)
    max_gap_pixels=max((len(g) for g in inferred_groups),default=0)
    coverage=float(observed.mean())
    path_rows=np.rint(np.clip(refined,0,evidence_cost.shape[0]-1)).astype(int)
    path_costs=evidence_cost[path_rows,np.arange(evidence_cost.shape[1])]
    diagnostics={'coverage':coverage,'max_gap_pixels':int(max_gap_pixels),'quality':'A' if coverage>=.97 else ('B' if coverage>=.90 else 'C'),
      'tracking_method':'global-dp-antialias-optimized','anchor':anchor_used,'inferred_pixels':int((~observed).sum()),
      'mean_evidence_cost':round(float(np.mean(path_costs)),2),'p95_evidence_cost':round(float(np.percentile(path_costs,95)),2),
      'smoothness':round(float(smoothness),4),
      'strict_24_coverage':round(float(strict_observed.mean()),6),
      'strict_24_max_gap_pixels':int(strict_max_gap_pixels),
      'relaxed_35_coverage':round(float(relaxed_observed.mean()),6),
      'evidence_threshold':float(direct_threshold),
      'nhk_chroma_guard':int(nhk_chroma_guard),
      'long_jump_count':int(np.sum(np.abs(np.diff(refined))>max(8.,float(pixels_per_percent)*1.0))),
      'max_step_pp':round(float(np.max(np.abs(np.diff(refined))))/float(pixels_per_percent),3)}
    return xs,ys,observed,diagnostics,strict_observed,relaxed_observed

def analyze_image(arr, end_time, duration_minutes=180, tolerance=DEFAULT_CURSOR_RGB_TOLERANCE,
                  grid_percent_step=2., *, evidence_mode='auto', return_traces=False):
    """TVAL graph RGB -> time series.

    evidence_mode: auto / strict / high-range.
    return_traces=True なら追加の4番目の戻り値として描画検証用の経路を返す。
    従来の App.py とは3戻り値のまま互換。
    """
    if evidence_mode not in ('auto', 'strict', 'high-range'):
        raise ValueError('evidence_mode は auto / strict / high-range から選択してください')
    if arr.ndim!=3 or arr.shape[2]!=3:raise ValueError('RGB only')
    if duration_minutes<=0 or grid_percent_step<=0 or tolerance<1:raise ValueError('invalid args')
    thin_lines=_find_thin_horizontal_lines(arr)
    grid,grid_gap=_select_regular_grid(thin_lines,arr.shape[0],min_lines=3)
    graph_top,y_zero=int(grid[0]),int(grid[-1])
    pixels_per_percent=grid_gap/float(grid_percent_step)
    rough_left,rough_right=_find_rough_graph_x_bounds(arr,grid)
    start_x,end_x,pixels_per_minute,time_axis_method,time_line_inliers=_calibrate_time_axis(arr,grid,rough_left,rough_right,end_time,duration_minutes)
    calibration=Calibration(start_x,end_x,float(graph_top),float(y_zero),float(y_zero),float(pixels_per_percent),float(pixels_per_minute),end_time-timedelta(minutes=duration_minutes),end_time,duration_minutes,float(grid_percent_step),float(grid_gap),time_axis_method,time_line_inliers)
    cursor_x=_find_cursor_x(arr,graph_top,y_zero,start_x,end_x)
    trace_left=max(0,int(np.floor(start_x))-4)
    trace_right=min(arr.shape[1]-1,int(np.ceil(end_x))+4)
    trace_roi=arr[graph_top:y_zero+1,trace_left:trace_right+1]
    evidence_costs=_precompute_station_evidence_costs(trace_roi)
    # 処理対象はグラフの見た目の「最高視聴率」でなく縦軸上限。
    # 例: 0～24%は high-range、0～12%なら normal。
    axis_upper_percent=(y_zero-graph_top)/pixels_per_percent
    # 「経路を追った結果」ではなく、各列のどこかに強い色が存在する割合で
    # 厳格閾値が画像に適合しているか検査する。値が低いだけでは誤追跡とは断定しない。
    color_availability={s:float(np.mean(np.min(cost,axis=0)<=STRICT_EVIDENCE_THRESHOLD))
                        for s,cost in evidence_costs.items()}
    high_range=axis_upper_percent>=HIGH_RANGE_PERCENT
    weak_color_stations=sum(v<.88 for v in color_availability.values())
    adaptive_soft=(evidence_mode=='high-range' or
                   (evidence_mode=='auto' and high_range and weak_color_stations>=2))
    evidence_threshold=(RELAXED_EVIDENCE_THRESHOLD if adaptive_soft else STRICT_EVIDENCE_THRESHOLD)
    # ふだんの0～12%画像を変更しない。高レンジ画像でも色証拠が十分なら厳格なまま。
    nhk_chroma_guard=adaptive_soft
    traces={};diagnostics={}

    for station,color in STATION_COLORS.items():
        cursor_y=None
        if cursor_x is not None and station!='NHK総合':
            cursor_y=_find_cursor_station_y(arr,cursor_x,graph_top,y_zero,color,tolerance=max(int(tolerance),1))
        xs,ys,observed,station_diag,_strong_mask,_soft_mask=_trace_station_global(arr,station,evidence_costs[station],trace_left,trace_right,graph_top,y_zero,pixels_per_percent,cursor_x=cursor_x,cursor_y=cursor_y,grid_rows=grid,
                evidence_threshold=evidence_threshold,nhk_chroma_guard=nhk_chroma_guard)
        gap_minutes=float(station_diag['max_gap_pixels'])/pixels_per_minute
        station_diag['max_gap_minutes']=round(gap_minutes,2)
        coverage=float(station_diag['coverage'])
        station_diag['quality']='A' if coverage>=.97 and gap_minutes<=2. else 'B' if coverage>=.90 and gap_minutes<=5. else 'C'
        strict_cov=float(station_diag['strict_24_coverage'])
        strict_gap=float(station_diag['strict_24_max_gap_pixels'])/pixels_per_minute
        # strict_24 は従来からの閾値。新たな緩和判定と分けて表示する。
        station_diag['quality_strict']='A' if strict_cov>=.97 and strict_gap<=2 else ('B' if strict_cov>=.90 and strict_gap<=5 else 'C')
        station_diag['quality_basis']='緩和35' if adaptive_soft else '厳格24'
        station_diag['reader_patch']=READER_PATCH_VERSION
        station_diag['axis_upper_percent']=round(float(axis_upper_percent),2)
        station_diag['range_mode']=('forced-high' if evidence_mode=='high-range' else 'high-adaptive') if adaptive_soft else ('forced-strict' if evidence_mode=='strict' else ('high-strict' if high_range else 'normal-strict'))
        station_diag['color_availability_strict']=round(float(color_availability[station]),6)
        traces[station]=(xs,ys,observed);diagnostics[station]=station_diag
    minute_indices=np.arange(duration_minutes+1,dtype=np.float64)
    times=[calibration.start_time+timedelta(minutes=int(m)) for m in minute_indices]
    sample_x=start_x+minute_indices*pixels_per_minute
    df=pd.DataFrame({'日時':times})
    raw_station_values={};station_status={}
    for station,(xs,ys,observed) in traces.items():
        sample_y=np.interp(sample_x,xs,ys)
        values=np.maximum(0.,(y_zero-sample_y)/pixels_per_percent)
        nearest=np.rint(sample_x-float(xs[0])).astype(np.int64)
        np.clip(nearest,0,len(xs)-1,out=nearest)
        raw_station_values[station]=values
        station_status[station]=np.where(observed[nearest],'実測','経路推定').tolist()
        # 色証拠の強弱は diagnostics にのみ保存し、視聴率CSVには追加しない。
        df[station]=np.round(values,1)
    stations=list(STATION_COLORS)
    raw_matrix=np.column_stack([raw_station_values[s] for s in stations])
    raw_total=raw_matrix.sum(axis=1)
    df['6局合計']=np.round(raw_total,1)
    df['その他5局合計']=np.round(raw_total-raw_station_values['NHK総合'],1)
    for station in stations:
        share=np.divide(raw_station_values[station],raw_total,out=np.full_like(raw_total,np.nan,dtype=float),where=raw_total!=0)*100.
        df[f'{station}_シェア']=np.round(share,2)
    raw_rank_df=pd.DataFrame({s:raw_station_values[s] for s in stations})
    ranks=raw_rank_df.rank(axis=1,ascending=False,method='min')
    for station in stations:
        df[f'{station}_順位']=ranks[station].astype('Int64')
        df[f'{station}_状態']=station_status[station]
    # 旧9.2互換: 列が増減・並び替わった場合は黙って出力せず検証する。
    if set(df.columns) != set(CSV_COLUMNS):
        raise RuntimeError(f"CSV列が旧版と一致しません: 取得={list(df.columns)}")
    df = df.loc[:, CSV_COLUMNS]
    if return_traces:
        return df,calibration,diagnostics,traces
    return df,calibration,diagnostics

