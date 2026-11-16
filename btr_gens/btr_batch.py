"""Local batch generation. No model calls; labels never enter input/ files."""
import copy
import hashlib
import json
import math
import random
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def validate_params(p):
    if not isinstance(p, dict):
        raise ValueError('参数必须为对象')
    def number(value, lo, hi, label):
        if isinstance(value, bool):
            raise ValueError(label + '不是有效数字')
        value = float(value)
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'{label}须在 {lo}～{hi} 之间')
        return value
    duration = number(p.get('T_MAX', 600), 1, 7200, '仿真时长')
    dt = number(p.get('DT', 1), 0.1, duration, '时间步长')
    low = number(p.get('B_MIN', 0), 0, 180, '方位下限')
    high = number(p.get('B_MAX', 180), 0, 180, '方位上限')
    step = number(p.get('DB_STEP', .25), .05, 10, '方位步长')
    if low >= high or (duration/dt+2)*((high-low)/step+2) > 5_000_000:
        raise ValueError('方位范围无效或采样网格超过500万，请增大步长')
    number(p.get('NOISE_STD', .03), 0, 10, '噪声')
    number(p.get('CLUTTER_LAM', .012), 0, 10, '杂波')
    number(p.get('EARTH_R', 6371000), 1e6, 1e8, '地球半径')
    number(p.get('REF_LAT', 30), -89, 89, '基准纬度')
    number(p.get('REF_LON', 122), -180, 180, '基准经度')
    db = p.get('DB_RANGE', [-5, 16])
    if len(db) != 2 or number(db[0], -200, 200, 'dB下限') >= number(db[1], -200, 200, 'dB上限'):
        raise ValueError('dB范围无效')
    for key, limit in [('platforms', 8), ('targets', 12)]:
        items = p.get(key)
        if not isinstance(items, list) or not 1 <= len(items) <= limit:
            raise ValueError(f'{key}数量必须在1～{limit}之间')
        names = set()
        for item in items:
            name = item.get('name', '')
            if not isinstance(name, str) or not re.fullmatch(r'[\w\-]{1,40}', name) or name in names:
                raise ValueError('平台/目标名称须唯一，且仅含文字、数字、下划线或短横线')
            names.add(name)
            point = item.get('start_ll' if key == 'platforms' else 'start')
            number(point[0], -89, 89, '起点纬度')
            number(point[1], -180, 180, '起点经度')
            number(item['speed_kn'], 0, 100, '航速')
            if key == 'platforms':
                number(item['heading_deg'], -3600, 3600, '航向')
                for field in ('hull_m', 'tow_m'):
                    if len(item[field]) != 2:
                        raise ValueError('阵列偏移需要x/y')
                    for value in item[field]:
                        number(value, -100000, 100000, '阵列偏移')
            else:
                number(item['amp'], 0, 10, '强度')
                for field in ('sigma_hull', 'sigma_tow'):
                    number(item[field], .01, 90, '亮带宽度')
                on = number(item['t_on'], 0, duration, '出现时间')
                if number(item['t_off'], 0, duration, '消失时间') < on:
                    raise ValueError('消失时间不能早于出现时间')
                if item.get('path_type') == 'curve':
                    # Existing parser is mathematical-expression-only.
                    from btr_gui_server import parse_curve_expr
                    fn = parse_curve_expr(str(item.get('curve_func', '')))
                    for t in (0, duration/2, duration):
                        lat, lon = fn(t)
                        number(lat, -89, 89, '曲线纬度')
                        number(lon, -180, 180, '曲线经度')
                else:
                    for lat, lon in item.get('waypoints') or []:
                        number(lat, -89, 89, '航点纬度')
                        number(lon, -180, 180, '航点经度')


def sample_params(base, mode, seed, percent, index):
    if mode == 'scenarios':
        from gen_train_ds import SCENARIOS, build_payload
        spec = SCENARIOS[index % len(SCENARIOS)]
        # Existing S01-S16 builder, with deterministic variation per batch seed.
        p = build_payload(*spec, seed + index // len(SCENARIOS), 0)
    else:
        p = copy.deepcopy(base)
    p['SEED'] = seed + index
    p['SHOW_TRUTH_OVERLAY'] = False
    p['EXPORT_TRUTH'] = True
    if mode == 'perturb':
        rng = random.Random(seed + index)
        factor = percent/100
        scale = lambda v: float(v)*rng.uniform(1-factor, 1+factor)
        p['NOISE_STD'] = scale(p.get('NOISE_STD', .03))
        p['CLUTTER_LAM'] = scale(p.get('CLUTTER_LAM', .012))
        for pl in p['platforms']:
            pl['heading_deg'] = (pl['heading_deg'] + rng.uniform(-30*factor, 30*factor)) % 360
            pl['speed_kn'] = scale(pl['speed_kn'])
        for target in p['targets']:
            target['amp'] = scale(target['amp'])
            for field in ('sigma_hull', 'sigma_tow'):
                target[field] = scale(target[field])
            if target.get('path_type') != 'curve':
                target['speed_kn'] = scale(target['speed_kn'])
                offset = [rng.uniform(-.01*factor, .01*factor) for _ in range(2)]
                target['start'] = [v+d for v,d in zip(target['start'], offset)]
                target['waypoints'] = [[v+d for v,d in zip(wp, offset)] for wp in target.get('waypoints') or []]
    return p


class BatchManager:
    def __init__(self, root, run, run_lock):
        self.root = Path(root)
        self.run = run
        self.run_lock = run_lock
        self.lock = threading.Lock()
        self.jobs = {}

    def start(self, request):
        count = request.get('count', 10)
        seed = request.get('seed', 20260911)
        percent = request.get('percent', 15)
        mode = request.get('mode', 'noise')
        if type(count) is not int or not 1 <= count <= 1000:
            raise ValueError('样本数须为1～1000的整数')
        if type(seed) is not int or not 0 <= seed <= 2**31-1001:
            raise ValueError('随机种子须为0～2147482647的整数')
        if not isinstance(percent, (int,float)) or not math.isfinite(percent) or not 0 <= percent <= 50:
            raise ValueError('扰动百分比须为0～50')
        if mode not in ('noise', 'perturb', 'scenarios'):
            raise ValueError('未知生成模式')
        base = copy.deepcopy(request.get('params'))
        if mode != 'scenarios':
            validate_params(base)
        # Validate every derived payload before accepting a job.
        for i in range(count):
            validate_params(sample_params(base, mode, seed, percent, i))
        with self.lock:
            if any(j['state'] in ('queued','running','stopping') for j in self.jobs.values()):
                raise ValueError('已有批次运行中，请等待完成或停止后重试')
            job_id = 'batch_' + datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:6]
            directory = self.root / job_id
            directory.mkdir(parents=True)
            config = dict(count=count, seed=seed, percent=percent, mode=mode, params=base)
            write_json(directory/'batch_config.json', config)
            job = dict(id=job_id, state='queued', total=count, completed=0,
                       failed=0, current=0, stop=False, samples=[], out_dir=str(directory))
            self.jobs[job_id] = job
            self._persist(job)
        threading.Thread(target=self._worker, args=(job_id, config), daemon=True).start()
        return self.status(job_id)

    def status(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                if not re.fullmatch(r'batch_[0-9_]+[0-9a-f]{6}', job_id):
                    raise KeyError('批次不存在')
                manifest=self.root/job_id/'manifest.json'
                if not manifest.is_file():
                    raise KeyError('批次不存在')
                historical=json.loads(manifest.read_text(encoding='utf-8-sig'))
                if historical['state'] in ('queued','running','stopping'):
                    historical.update(state='interrupted',error='服务已重启，原批次未自动续跑；已完成文件仍保留。')
                return historical
            return copy.deepcopy(self.jobs[job_id])

    def stop(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            job['stop'] = True
            if job['state'] in ('queued','running'):
                job['state'] = 'stopping'
            self._persist(job)
        return self.status(job_id)

    def _persist(self, job):
        write_json(Path(job['out_dir'])/'manifest.json', job)

    def _worker(self, job_id, config):
        job = self.jobs[job_id]
        directory = Path(job['out_dir'])
        try:
            with self.lock:
                if not job['stop']:
                    job['state'] = 'running'
            for i in range(config['count']):
                with self.lock:
                    if job['stop']:
                        break
                    job['current'] = i+1
                    self._persist(job)
                sample_dir = directory / f'sample_{i+1:04d}'
                input_dir, truth_dir = sample_dir/'input', sample_dir/'truth'
                input_dir.mkdir(parents=True)
                truth_dir.mkdir()
                p = sample_params(config['params'], config['mode'], config['seed'], config['percent'], i)
                p['OUT_DIR'] = str(sample_dir)
                entry = dict(sample_id=sample_dir.name, state='failed', seed=p['SEED'])
                write_json(sample_dir/'meta.json', dict(params=p, state='running'))
                try:
                    with self.run_lock:
                        if job['stop']:
                            write_json(sample_dir/'meta.json', dict(params=p, state='cancelled'))
                            break
                        result = self.run(p)
                    for pl in result['platforms']:
                        for field in ('imgs','txts'):
                            for kind, filename in pl[field].items():
                                src = Path(filename)
                                dest = input_dir if field == 'imgs' and kind in ('hull','tow') else truth_dir
                                src.replace(dest/src.name)
                    for src in sample_dir.glob('*_truth.jsonl'):
                        src.replace(truth_dir/src.name)
                    # Only known platform geometry and observation axes enter input.
                    geometry = {key:p.get(key, default) for key,default in [
                        ('T_MAX',600),('DT',1),('B_MIN',0),('B_MAX',180),
                        ('REF_LAT',30),('REF_LON',122),('EARTH_R',6371000)]}
                    geometry['platforms'] = p['platforms']
                    geometry['platform_motion_model'] = 'constant_heading_constant_speed'
                    geometry['image_time_axis'] = 'bottom_to_top'
                    geometry['bearing_convention'] = 'relative_heading_folded_0_180'
                    geometry['left_right_ambiguous'] = True
                    write_json(input_dir/'platform.json', geometry)
                    from btr_semantic import export_semantic
                    export_semantic(sample_dir, p)
                    files = {}
                    for f in sorted(sample_dir.rglob('*')):
                        if f.is_file() and f.name != 'meta.json':
                            files[f.relative_to(directory).as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
                    entry.update(state='completed', files=files)
                    input_record = dict(sample_id=sample_dir.name,
                        platform_metadata=f'{sample_dir.name}/input/platform.json',
                        images=[f.relative_to(directory).as_posix() for f in sorted(input_dir.glob('*.png'))])
                    with (directory/'model_inputs.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps(input_record, ensure_ascii=False)+'\n')
                except Exception as exc:
                    entry['error'] = str(exc)
                write_json(sample_dir/'meta.json', dict(params=p, **entry))
                entry['meta_sha256'] = hashlib.sha256((sample_dir/'meta.json').read_bytes()).hexdigest()
                with self.lock:
                    job['samples'].append(entry)
                    job['completed' if entry['state']=='completed' else 'failed'] += 1
                    self._persist(job)
            with self.lock:
                job['state'] = 'stopped' if job['stop'] else ('completed_with_errors' if job['failed'] else 'completed')
                self._persist(job)
        except Exception as exc:
            with self.lock:
                job.update(state='failed', error=str(exc))
                self._persist(job)
