"""Export simulator reference semantics, NOT model predictions.

CLI: python btr_semantic.py --root btr_out
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA_PATH = Path(__file__).with_name('btr_semantic_schema_v1.json')


def validate(document, schema=None):
    """Validate all assertions used by the supplied draft-07 schema.

    No dependency or claim of implementing the full JSON Schema standard.
    Unsupported assertion keywords fail closed, rather than being ignored.
    """
    root = schema or json.loads(SCHEMA_PATH.read_text(encoding='utf-8-sig'))
    annotations = {'$schema', '$id', 'title', 'description', 'examples', 'default', 'definitions'}
    supported = {'$ref', 'type', 'required', 'properties', 'additionalProperties',
                 'items', 'enum', 'minimum', 'maximum'}
    def check(value, rule, path):
        unknown = set(rule)-annotations-supported
        if unknown:
            raise ValueError(f'{path}: unsupported schema keywords: {sorted(unknown)}')
        if '$ref' in rule:
            ref = rule['$ref']
            if not ref.startswith('#/'):
                raise ValueError('Only local schema references are supported')
            resolved = root
            for part in ref[2:].split('/'):
                resolved = resolved[part.replace('~1','/').replace('~0','~')]
            check(value, resolved, path)
            return
        def matches(kind):
            return {'object':isinstance(value,dict), 'array':isinstance(value,list),
                    'string':isinstance(value,str), 'boolean':type(value) is bool,
                    'integer':type(value) is int,
                    'number':type(value) in (int,float) and math.isfinite(value),
                    'null':value is None}.get(kind,False)
        types = rule.get('type')
        if types and not any(matches(t) for t in ([types] if isinstance(types,str) else types)):
            raise ValueError(f'{path}: expected {types}')
        if 'enum' in rule and value not in rule['enum']:
            raise ValueError(f'{path}: invalid enum value')
        if type(value) in (int,float):
            if not math.isfinite(value) or value < rule.get('minimum',-math.inf) or value > rule.get('maximum',math.inf):
                raise ValueError(f'{path}: invalid numeric range')
        if isinstance(value,dict):
            for key in rule.get('required',[]):
                if key not in value:
                    raise ValueError(f'{path}.{key}: required')
            for key, val in value.items():
                child = rule.get('properties',{}).get(key)
                if child is None:
                    child = rule.get('additionalProperties',True)
                if child is False:
                    raise ValueError(f'{path}.{key}: unexpected property')
                if isinstance(child,dict):
                    check(val,child,path+'.'+key)
        if isinstance(value,list) and isinstance(rule.get('items'),dict):
            for i,val in enumerate(value):
                check(val,rule['items'],f'{path}[{i}]')
    check(document,root,'$')


def save(path, document):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(document,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def envelope(value, method, rule, confidence):
    return dict(value=value,method=method,rule_id=rule,confidence=confidence)


def evaluate_path(target, time_s, radius):
    """Same global-t=0 path semantics as the simulator; no mutable globals."""
    start = tuple(target['start'])
    if target.get('path_type') == 'curve':
        from btr_gui_server import parse_curve_expr
        lat,lon = parse_curve_expr(target['curve_func'])(time_s)
        return float(lat),float(lon)
    speed = float(target['speed_kn'])*.514444
    remaining = time_s
    point = start
    for destination in target.get('waypoints') or []:
        p,q = map(math.radians,(point[0],destination[0]))
        h = math.sin((q-p)/2)**2 + math.cos(p)*math.cos(q)*math.sin(math.radians(destination[1]-point[1])/2)**2
        distance = 2*radius*math.asin(min(1,math.sqrt(h)))
        if distance < .001:
            continue
        duration = distance/max(speed,1e-6)
        if remaining <= duration:
            fraction = remaining/duration
            return tuple(a+fraction*(b-a) for a,b in zip(point,destination))
        remaining -= duration
        point = tuple(destination)
    return point


def position(latlon):
    return dict(lat=float(latlon[0]),lon=float(latlon[1]))


def build_semantic(sample_dir, params=None):
    directory = Path(sample_dir)
    if params is None:
        meta = json.loads((directory/'meta.json').read_text(encoding='utf-8-sig'))
        if meta.get('state') != 'completed':
            raise ValueError('样本尚未完成，不能生成完整语义参考数据')
        params = meta['params']
    p = params
    duration = float(p.get('T_MAX',600))
    radius = float(p.get('EARTH_R',6371000))
    warnings = ['生成端真值及规则参考结论，不是模型识别结果；禁止作为盲测输入。',
                'TXT方位起止/CPA沿用原文件口径（通常为全仿真时段），并非仅有效观测窗。',
                '规则置信度为人工设定的规则强度，不是经过校准的正确概率。']
    platform_blocks, source_files, parsed, generated = [], {}, [], None
    all_cpas, cross_ids, target_ids = [], set(), set()
    for plat in p['platforms']:
        name = plat['name']
        contents = {}
        source_files[name] = {}
        for kind in ('track','hull','tow'):
            rel = f'truth/btr_{name}_{kind}.txt'
            source_files[name][kind] = rel
            contents[kind] = (directory/rel).read_text(encoding='utf-8-sig').splitlines()
            parsed.append(rel)
        if generated is None:
            line = next((x for x in contents['track'] if x.startswith('生成时间:')),None)
            generated = line.split(':',1)[1].strip() if line else ''
        def read_observation(kind, tid):
            lines = contents[kind]
            for number,line in enumerate(lines,1):
                if line.split('|',1)[0].strip() != tid:
                    continue
                match = re.search(r'舷角\s*([-\d.]+)°->\s*([-\d.]+)°.*?CPA=\s*([\d.]+)\s*km\s*@\s*t=\s*([\d.]+)s',line)
                if match:
                    begin,end,cpa,at = map(float,match.groups())
                    done = re.search(r'走完航点 t=([\d.]+)s',line)
                    return dict(array_name=kind,bearing_start_deg=begin,bearing_end_deg=end,
                                bearing_rate_deg_per_s=(end-begin)/duration,
                                cpa=dict(range_km=cpa,time_s=at)), f'{source_files[name][kind]}#L{number}', float(done[1]) if done else None
            raise ValueError(f'{kind} TXT缺少目标 {tid} 的观测记录')
        speed = float(plat['speed_kn'])*.514444
        heading = math.radians(float(plat['heading_deg']))
        lat0,lon0 = plat['start_ll']
        ref_lat = math.radians(float(p.get('REF_LAT',30)))
        end = (lat0+math.degrees(speed*math.cos(heading)*duration/radius),
               lon0+math.degrees(speed*math.sin(heading)*duration/(radius*math.cos(ref_lat))))
        def array(field, kind):
            x,y = map(float,plat[field])
            result = dict(type='舷侧阵' if kind=='hull' else '拖曳线列阵',
                          mount='船体固定偏移',offset_m=dict(x=x,y=y))
            if kind=='tow':
                hx,hy = plat['hull_m']
                result['baseline_m'] = math.hypot(x-hx,y-hy)
            return result
        block = dict(platform_id=name,self=dict(position_start=position((lat0,lon0)),
                     position_end=position(end),heading_deg=float(plat['heading_deg']),
                     speed_kn=float(plat['speed_kn']),speed_ms=speed,
                     motion_state='移动' if speed>0 else '静止',
                     array_hull=array('hull_m','hull'),array_tow=array('tow_m','tow'),
                     bounds=plat.get('bounds')),
                     observation=dict(t_range_s=[0,duration],dt_s=float(p.get('DT',1)),
                         bearing_convention='相对船头舷角，折叠0～180°',left_right_ambiguous=True),
                     targets=[],cross_events=[],parallax_extra=[],warnings=[])
        for target in p['targets']:
            tid = target['name']; target_ids.add(tid)
            h,href,done = read_observation('hull',tid)
            tow,tref,_ = read_observation('tow',tid)
            refs = [href,tref]
            obs = dict(hull=h,tow=tow)
            for number,line in enumerate(contents['tow'],1):
                match = re.search(r'^\s*(\S+).*?舷侧阵舷角=([\d.]+)°.*?拖曳阵舷角=([\d.]+)°.*?\|Δ\|=([\d.]+)°',line)
                if match and match[1]==tid:
                    obs['parallax_t300']=dict(at_s=300,target_label=tid,
                        hull_bearing_deg=float(match[2]),tow_bearing_deg=float(match[3]),delta_deg=float(match[4]))
                    refs.append(f'{source_files[name]["tow"]}#L{number}')
            start_pos = evaluate_path(target,0,radius)
            end_pos = evaluate_path(target,duration,radius)
            cpa = min(h['cpa']['range_km'],tow['cpa']['range_km'])
            all_cpas.append(cpa)
            crossing = cpa < .5
            for measured in (h,tow):
                if measured['cpa']['range_km'] < .5:
                    block['cross_events'].append(dict(target_id=tid,array_name=measured['array_name'],
                        cpa_km=measured['cpa']['range_km'],time_s=measured['cpa']['time_s']))
            if crossing: cross_ids.add(tid)
            on,off = float(target['t_on']),float(target['t_off'])
            twarnings=[]
            if target['amp']<.15: twarnings.append('弱目标幅度小于0.15')
            if target.get('path_type')=='curve': twarnings.append('曲线速度由公式导数决定；speed_kn/speed_ms仅为输入标称值，不是实测航速')
            if done is not None and done<=duration:
                twarnings.append('目标走完航点后停留；静止平台下后续方位恒定，距离可观性需另行评估')
            if obs.get('parallax_t300',{}).get('delta_deg',180)<.35:
                twarnings.append('t=300秒双阵视差小于0.35°，单历元距离估计不稳定')
            if 'parallax_t300' in obs and not on<=300<=off:
                twarnings.append('t=300秒不在目标有效观测窗内，此视差仅是生成端几何真值')
            distance_class = '极近' if cpa<1.2 else '近' if cpa<5 else '中' if cpa<=15 else '远'
            at = h['cpa']['time_s']
            approach = '远离中' if at<=on else '接近中' if at>=off else '最近通过（先接近后远离）'
            noise = p.get('NOISE_STD',.03)
            snr = 20*math.log10(target['amp']/noise) if target['amp']>0 and noise>0 else None
            window = off-on
            grade = None if snr is None else 'D' if snr<8 else 'C' if snr<11 or window<120 else 'B' if snr<14 or window<300 else 'A'
            # End-to-end geodesic direction, not instantaneous course for curves.
            a,b=map(math.radians,(start_pos[0],end_pos[0])); dl=math.radians(end_pos[1]-start_pos[1])
            course = None if start_pos==end_pos else math.degrees(math.atan2(math.sin(dl)*math.cos(b),math.cos(a)*math.sin(b)-math.sin(a)*math.cos(b)*math.cos(dl)))%360
            inference = {
                'target_type':envelope(None,'仅凭轨迹与航速不能可靠确定目标类型','RULE-TYPE-UNKNOWN',0),
                'motion_model':envelope('曲线' if target.get('path_type')=='curve' else '分段直线/到点停留','依据生成路径定义，不代表图像识别结果','RULE-PATH-01',1),
                'course_estimate_deg':envelope(course,'全仿真起终点反解的净位移方向，不是机动目标瞬时航向','RULE-COURSE-01',.6),
                'approach_state':envelope(approach,'按TXT全时段CPA时刻与有效观测窗比较，非逐时刻径向运动判定','RULE-APP-01',.6),
                'distance_class':envelope(distance_class,'按双阵全仿真最小CPA分档，非初始/平均距离','RULE-DIST-01',1),
                'crossing':envelope(crossing,'最小CPA<0.5km视作极近通过，不等同几何航迹交叉','RULE-CROSS-01',.6),
                'side_ambiguity':envelope(dict(mirror_candidates_deg=[(plat['heading_deg']+h['bearing_start_deg'])%360,(plat['heading_deg']-h['bearing_start_deg'])%360],note='起始折叠方位的绝对方位候选；是否可解需进一步评估'),'按航向±舷角计算','RULE-SIDE-01',1),
                'depth_estimate_m':envelope(None,'当前二维仿真不含深度观测或深度真值','RULE-DEPTH-01',0),
                'snr_est_db':envelope(snr,'20log10(amp/noise_std)的幅度比启发式指标，不是由图像测得的物理SNR','RULE-SNR-01',.5),
                'observability_grade':envelope(grade,'按幅度比指标和观测时长粗分级，非定位误差等级','RULE-OBS-01',.5),
                'threat_level':envelope(None,'无意图/身份等依据，未评估','RULE-THREAT-UNKNOWN',0)}
            kinematics = dict(speed_kn=target['speed_kn'],speed_ms=target['speed_kn']*.514444,
                position_start=position(start_pos),position_end=position(end_pos),amp=target['amp'],
                observation_window_s=[on,off],waypoints_done_s=done)
            block['targets'].append(dict(target_id=tid,raw_label=None,kinematics=kinematics,
                observations=obs,inferences=inference,warnings=twarnings,evidence=dict(txt_refs=refs),
                summary_nl=f'{tid}：有效观测窗{on:g}～{off:g}秒；全时段最小阵列CPA约{cpa:g}km。'))
        block['summary_nl']=f'平台{name}{block["self"]["motion_state"]}，包含{len(block["targets"])}个仿真目标。'
        known_ids={t['name'] for t in p['targets']}
        for line in contents['tow']:
            match=re.search(r'^\s*(\S+).*?舷侧阵舷角=([\d.]+)°.*?拖曳阵舷角=([\d.]+)°.*?\|Δ\|=([\d.]+)°',line)
            if match and match[1] not in known_ids:
                block['parallax_extra'].append(dict(at_s=300,target_label=match[1],
                    hull_bearing_deg=float(match[2]),tow_bearing_deg=float(match[3]),delta_deg=float(match[4])))
                block['warnings'].append('TXT视差标签未匹配目标清单，原样保留，未强行归属：'+match[1])
        platform_blocks.append(block)
    classes = {'近' if x<5 else '中' if x<=15 else '远' for x in all_cpas}
    motions = {b['self']['motion_state'] for b in platform_blocks}
    if len(motions)>1: warnings.append('多平台运动状态混合；受v1枚举限制，platform_state按任一平台移动汇总')
    scenario = dict(platform_state='移动' if '移动' in motions else '静止',
        distance_class=next(iter(classes)) if len(classes)==1 else '混合',
        target_count_class='少' if len(target_ids)<=2 else '多',target_total=len(target_ids),
        crossing='是' if cross_ids else '否',crossing_count=len(cross_ids),
        notes=['距离按全时段最小CPA分档；≤2个为少，>2个为多；跨平台目标编号去重；穿越按CPA<0.5km口径。',
               '与预设场景的近远/穿越标签不一定同口径，不将生成意图作为验证结论。'])
    source_meta=p.get('_meta')
    dataset=None if not source_meta else dict(scenario_code=source_meta.get('code',''),
        category=source_meta.get('category',''),sample=source_meta.get('sample'),
        scenario='_'.join(str(source_meta.get(k,'')) for k in ('dist','count','cross')),
        conditions=dict(distance=source_meta.get('dist'),count=source_meta.get('count'),crossing=source_meta.get('cross')),
        label_check=None)
    document = dict(schema_version='1.0.0',record_id=f'SEM-{directory.parent.name}-{directory.name}',
        generated_at=generated or '',summary_nl=f'仿真参考数据：{len(platform_blocks)}个平台，{len(target_ids)}个目标；不是模型预测。',
        source=dict(generator='btr_gens/simulation_reference',run_dir=str(directory.resolve()),seed=p.get('SEED'),
                    dataset=dataset,files=source_files),
        sim_settings=dict(t_max_s=duration,dt_s=p.get('DT',1),bearing_range_deg=[p.get('B_MIN',0),p.get('B_MAX',180)],
            bearing_step_deg=p.get('DB_STEP',.25),earth_radius_m=radius,
            projection_ref=dict(lat=p.get('REF_LAT',30),lon=p.get('REF_LON',122)),
            noise_std=p.get('NOISE_STD',.03),clutter_lam=p.get('CLUTTER_LAM',.012),db_range=p.get('DB_RANGE',[-5,16])),
        scenario_summary=scenario,platforms=platform_blocks,environment={},
        quality=dict(label_check=None,parsed_sources=parsed+['meta.json/params'],warnings=warnings,
            schema_validation=dict(schema=SCHEMA_PATH.name,ok=True,
                validator='builtin: all assertion keywords used in supplied schema; not full draft-07 implementation')))
    validate(document)
    return document


def export_semantic(sample_dir, params=None):
    document=build_semantic(sample_dir,params)
    save(Path(sample_dir)/'semantic.json',document)
    return document


def backfill(root):
    results=[]
    for manifest in sorted(Path(root).glob('batch_*/manifest.json')):
        data=json.loads(manifest.read_text(encoding='utf-8-sig'))
        if data.get('state') in ('running','queued','stopping'):
            results.append(dict(batch=manifest.parent.name,state='skipped_active'))
            continue
        for sample in data.get('samples',[]):
            if sample.get('state')!='completed': continue
            directory=manifest.parent/sample['sample_id']
            export_semantic(directory)
            relative=f'{directory.name}/semantic.json'
            digest=hashlib.sha256((directory/'semantic.json').read_bytes()).hexdigest()
            sample.setdefault('files',{})[relative]=digest
            meta_path=directory/'meta.json'
            meta=json.loads(meta_path.read_text(encoding='utf-8-sig'))
            meta.setdefault('files',{})[relative]=digest
            save(meta_path,meta)
            sample['meta_sha256']=hashlib.sha256(meta_path.read_bytes()).hexdigest()
            results.append(dict(batch=manifest.parent.name,sample=directory.name,state='completed'))
        save(manifest,data)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(Path(__file__).parent/'btr_out'))
    args=parser.parse_args()
    print(json.dumps(backfill(args.root),ensure_ascii=False,indent=2))
