"""User schema_v1.json / 字段字典.md export adapter. Simulation is never modified.

Only semantic output, metadata hashes and manifest hashes are updated by backfill.
"""
import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path

import btr_semantic as legacy

SCHEMA_PATH = Path(__file__).resolve().parent.parent / 'schema_v1.json'


def validate(document):
    schema = json.loads(SCHEMA_PATH.read_text(encoding='utf-8-sig'))
    legacy.validate(document, schema=schema)


def _infer(value, method, rule, confidence):
    return dict(value=value, method=method, rule_id=rule, confidence=confidence)


def _target_rules(target, params, warnings):
    k = target['kinematics']
    speed, amp = float(k['speed_kn']), float(k['amp'])
    label = params.get('raw_label') or params.get('label') or target.get('raw_label')
    target['raw_label'] = str(label) if label else None
    on, off = k['observation_window_s']
    window = off-on
    for kind in ('hull', 'tow'):
        obs = target['observations'][kind]
        obs['bearing_rate_deg_per_s'] = (obs['bearing_end_deg']-obs['bearing_start_deg'])/window if window > 0 else 0.0
    if window <= 0:
        target['warnings'].append('观测窗长度为0，方位变化率以0占位，不能据此判断运动模型。')
    hull, tow = (target['observations'][key] for key in ('hull','tow'))
    closest = min((hull,tow), key=lambda obs:obs['cpa']['range_km'])
    cpa, at = closest['cpa']['range_km'], closest['cpa']['time_s']
    crossing = cpa < .5
    if speed >= 25:
        type_name, type_rule, type_conf = '高速水面目标（疑似军用快艇）', 'RULE-TYPE-01', .6
    elif speed >= 12:
        type_name, type_rule, type_conf = '水面舰艇（商船/军舰典型航速段）', 'RULE-TYPE-02', .5
    else:
        type_name, type_rule, type_conf = '低速目标（疑似潜艇潜航，待深度佐证）', 'RULE-TYPE-03', .4
    if amp < .15:
        type_name += '；弱目标'
    type_method = f'按字段字典的航速分档（标称航速{speed:g}kn，强度{amp:g}）进行类型粗判，不作为身份真值。'
    rate = abs(hull['bearing_rate_deg_per_s'])
    if label and '机动' in str(label):
        motion, motion_rule, motion_conf = 'CT', 'RULE-MM-01', .85
        motion_method = '生成端标签含“机动”，按字段字典的协同转弯启发式规则归类。'
    elif window > 0 and rate >= .08:
        motion, motion_rule, motion_conf = 'CA', 'RULE-MM-02', .55
        motion_method = f'舷侧阵平均方位变化率绝对值{rate:g}°/s≥0.08，按字段字典归类；不等同真实加速度。'
    else:
        motion, motion_rule, motion_conf = 'CV', 'RULE-MM-03', .7 if window > 0 else 0
        motion_method = f'无机动标签且平均方位变化率绝对值{rate:g}°/s<0.08，按字段字典归类。' if window > 0 else '观测窗长度为0，CV仅为格式占位，未作有效运动模型判断。'
    if cpa < 1.2:
        distance, distance_rule = '极近', 'RULE-DIST-01'
    elif cpa <= 5:
        distance, distance_rule = '近', 'RULE-DIST-02'
    elif cpa <= 15:
        distance, distance_rule = '中', 'RULE-DIST-03'
    else:
        distance, distance_rule = '远', 'RULE-DIST-04'
    if at <= on:
        approach, approach_rule = '远离中', 'RULE-APP-03'
    elif at >= off:
        approach, approach_rule = '接近中', 'RULE-APP-02'
    else:
        approach, approach_rule = '最近通过（先接近后远离）', 'RULE-APP-01'
    # SNR is the documented amplitude ratio, not an image-measured quantity.
    noise = warnings['noise']
    snr = 20*math.log10(amp/noise) if amp > 0 and noise is not None and noise > 0 else None
    snr_method = '20·log10(amp/noise_std)；'+('noise_std取默认假设值0.030（元数据未记录）。' if warnings['assumed_noise'] else 'noise_std取本样本元数据。')+'未计杂波与条纹，不是图像实测物理SNR。'
    if snr is None:
        snr_method += '当前幅度或噪声为0，无法得到有限比值，保留null。'
    grade = None if snr is None else 'D' if snr < 8 else 'C' if snr < 11 or window < 120 else 'B' if snr < 14 or window < 300 else 'A'
    score = speed/25 + max(0,5-cpa)/5 + (.5 if crossing else 0) + (.3 if speed >= 25 else 0)
    threat = '高' if score >= 1.7 else '中' if score >= 1 else '低'
    inference = target['inferences']
    inference.update(
        target_type=_infer(type_name,type_method,type_rule,type_conf),
        motion_model=_infer(motion,motion_method,motion_rule,motion_conf),
        approach_state=_infer(approach,f'最小CPA时刻{at:g}s与观测窗[{on:g},{off:g}]s比较，按字段字典作阶段粗判。',approach_rule,.6),
        distance_class=_infer(distance,f'按双阵全时段最小CPA={cpa:g}km分档：<1.2极近、[1.2,5]近、(5,15]中、>15远。',distance_rule,.8),
        crossing=_infer(crossing,f'双阵最小CPA={cpa:g}km，按CPA<0.5km判极近通过，不等同几何航迹相交。','RULE-CROSS-01',.8),
        depth_estimate_m=_infer(None,'当前生成端未提供深度真值，按字段字典保留null，不用类型粗判代填深度。','RULE-DEPTH-02' if speed < 12 else 'RULE-DEPTH-01',0),
        snr_est_db=_infer(snr,snr_method,'RULE-SNR-01',.6 if snr is not None else 0),
        observability_grade=_infer(grade,'D=SNR<8；C=SNR<11或窗<120s；B=SNR<14或窗<300s；其余A。幅度比/时长启发式分级。','RULE-OBS-01',.7 if grade else 0),
        threat_level=_infer(threat,f'字段字典关注度评分={score:.6f}：speed/25 + max(0,5−CPA)/5 + 穿越加成0.5 + 高速类型粗判加成0.3；≥1.7高、≥1中、其余低。是规则关注度，不是已确证意图。','RULE-THREAT-01',.6))
    if crossing:
        target['warnings'].append('最小阵列CPA<0.5km，按文档口径标记穿越/极近通过。')
    target['summary_nl'] = f"{target['target_id']}：标称航速{speed:g}kn，观测窗{on:g}～{off:g}s，双阵全时段最小CPA约{cpa:g}km。"
    return cpa, crossing


def build_semantic(sample_dir, params=None):
    directory = Path(sample_dir)
    if params is None:
        meta = json.loads((directory/'meta.json').read_text(encoding='utf-8-sig'))
        if meta.get('state') != 'completed':
            raise ValueError('样本尚未完成，不能导出完整语义记录')
        params = meta['params']
    # Reuse original TXT parsing and kinematic reconstruction without changing it.
    p = copy.deepcopy(params)
    document = legacy.build_semantic(directory, p)
    settings = document['sim_settings']
    settings.update(noise_std=p.get('NOISE_STD'), clutter_lam=p.get('CLUTTER_LAM'), db_range=p.get('DB_RANGE'), bearing_step_deg=p.get('DB_STEP'))
    step = settings['bearing_step_deg']
    low, high = settings['bearing_range_deg']
    settings['beam_cells'] = math.ceil((high-low)/step-1e-10)+1 if step and step > 0 else None
    assumed_noise = p.get('NOISE_STD') is None
    noise_info = dict(noise=.03 if assumed_noise else p['NOISE_STD'], assumed_noise=assumed_noise)
    notes = document['quality']['warnings']
    notes.append('类型、CV/CA/CT与关注度按字段字典规则生成，不作为识别结果或目标身份真值。')
    notes.append('方位起止保留TXT原值；平均变化率按字段字典除以有效观测窗长度。TXT端点为全仿真端点，窗口不一致时该值是文档约定指标。')
    if assumed_noise:
        notes.append('元数据未记录NOISE_STD；sim_settings.noise_std=null，SNR规则采用默认假设0.030并在method注明。')
    targets = {target['name']:target for target in p['targets']}
    platform_params = {plat['name']:plat for plat in p['platforms']}
    all_cpas, crosses = [], set()
    for platform in document['platforms']:
        own = platform['self']; pp = platform_params[platform['platform_id']]
        for name, key in [('array_hull','sigma_hull'),('array_tow','sigma_tow')]:
            sigmas = [float(t[key]) for t in p['targets'] if t.get(key) is not None]
            own[name]['beam_sigma_deg'] = [min(sigmas),max(sigmas)] if sigmas else []
        own['array_hull']['type'] = '舷侧阵（舰壳/弦测阵）'
        own['array_hull']['mount'] = '舰体中心' if all(v==0 for v in pp['hull_m']) else '船体固定偏移'
        own['array_tow']['mount'] = '舰艉正后方，随船头航向旋转' if pp['tow_m'][0] <= 0 and pp['tow_m'][1] == 0 else '船体固定偏移，随船头航向旋转'
        if own['speed_kn'] > 0 and not pp.get('waypoints'):
            own.pop('position_end', None)
            platform['warnings'].append('按字段字典：移动平台未提供航点序列，position_end缺省；原始仿真轨迹另存于真值文件。')
        for target in platform['targets']:
            cpa,cross = _target_rules(target,targets[target['target_id']],noise_info)
            all_cpas.append(cpa)
            if cross: crosses.add(target['target_id'])
        platform['summary_nl'] = f"平台{platform['platform_id']}{own['motion_state']}，观测{len(platform['targets'])}个目标，记录{len(platform['cross_events'])}项阵列极近通过事件。"
    classes = {'近' if cpa <= 5 else '中' if cpa <= 15 else '远' for cpa in all_cpas}
    summary = document['scenario_summary']
    summary.update(distance_class=next(iter(classes)) if len(classes)==1 else '混合',crossing='是' if crosses else '否',crossing_count=len(crosses))
    summary['notes'] = ['距离按双阵全时段最小CPA分档：≤5km近，(5,15]km中，>15km远；目标级另细分<1.2km极近。',
                        '目标跨平台按编号去重；≤2个为少，>2个为多；CPA<0.5km按文档标记穿越/极近通过。',
                        '生成配置的初始距离档不是实际CPA距离档，场景摘要以实际TXT结果为准。']
    combo = p.get('_studio')
    if combo:
        document['source']['dataset'] = dict(scenario_code=str(combo.get('combination_id','')),category={'stationary':'静止','moving':'移动'}.get(combo.get('combination',{}).get('motion'),str(combo.get('combination',{}).get('motion',''))),
            scenario=str(combo.get('label','')),conditions=copy.deepcopy(combo.get('combination',{})),sample=int(directory.name.split('_')[-1]) if re.fullmatch(r'sample_\d+',directory.name) else None,label_check=None)
    environment = p.get('environment') or {}
    document['environment'] = {key:environment.get(key) for key in ('sea_state','thermocline_depth_m','sound_speed_profile','seafloor_type')}
    document['environment']['notes'] = environment.get('notes',['环境字段缺省，保留给想定系统注入。'])
    document['summary_nl'] = f"{len(document['platforms'])}个平台，{summary['target_total']}个目标；平台状态{summary['platform_state']}，距离等级{summary['distance_class']}，穿越/极近通过目标{summary['crossing_count']}个。"
    document['quality']['schema_validation'] = dict(schema=SCHEMA_PATH.name, ok=False)
    validate(document)
    document['quality']['schema_validation']['ok'] = True
    validate(document)
    return document


def export_semantic(sample_dir, params=None):
    document = build_semantic(sample_dir, params)
    legacy.save(Path(sample_dir)/'semantic.json',document)
    return document


def backfill_batch(batch_dir, backup_root):
    """Validate all records before any writes; back up and refresh only JSON exports/hashes."""
    batch = Path(batch_dir).resolve()
    backup = Path(backup_root).resolve()/batch.name
    backup = backup.resolve()
    if backup == batch or backup in batch.parents or batch in backup.parents:
        raise ValueError('备份目录不能与源批次相同或相互嵌套')
    manifest_path = batch/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if manifest.get('state') in ('running','queued','stopping'):
        raise ValueError('活动批次不执行语义回填')
    staged = []
    for sample in manifest.get('samples',[]):
        if sample.get('state') != 'completed': continue
        name = sample['sample_id']
        if not re.fullmatch(r'sample_\d+',name):raise ValueError('无效样本目录名')
        directory = batch/name
        meta_path = directory/'meta.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8-sig'))
        staged.append((sample,directory,meta,build_semantic(directory)))
    backup.mkdir(parents=True,exist_ok=True)
    paths = [manifest_path]+[d/name for _,d,_,_ in staged for name in ('meta.json','semantic.json')]
    originals = {path:path.read_bytes() if path.is_file() else None for path in paths}
    for path,data in originals.items():
        if data is not None:
            dest = backup/path.relative_to(batch)
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():dest.write_bytes(data)
    protected = {path:hashlib.sha256(path.read_bytes()).hexdigest()
                 for _,d,_,_ in staged for path in d.rglob('*')
                 if path.is_file() and path.name not in ('meta.json','semantic.json')}
    try:
        for sample,directory,meta,document in staged:
            semantic = directory/'semantic.json'
            legacy.save(semantic,document)
            relative = semantic.relative_to(batch).as_posix()
            digest = hashlib.sha256(semantic.read_bytes()).hexdigest()
            sample.setdefault('files',{})[relative] = digest
            meta.setdefault('files',{})[relative] = digest
            legacy.save(directory/'meta.json',meta)
            sample['meta_sha256'] = hashlib.sha256((directory/'meta.json').read_bytes()).hexdigest()
        legacy.save(manifest_path,manifest)
        for path,digest in protected.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('原始文件发生变化：'+str(path))
    except Exception:
        for path,data in originals.items():
            if data is None:path.unlink(missing_ok=True)
            else:path.write_bytes(data)
        raise
    return dict(batch=str(batch),updated=len(staged),backup=str(backup),protected_files=len(protected),schema=str(SCHEMA_PATH))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',required=True)
    parser.add_argument('--backup-root',required=True)
    args = parser.parse_args()
    print(json.dumps(backfill_batch(args.batch,args.backup_root),ensure_ascii=False,indent=2))
