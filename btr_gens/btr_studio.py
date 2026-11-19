"""Deterministic, configurable factorial sampling for the local BTR studio.

Distance classes describe initial great-circle distance to platform A, not CPA.
All target path types use existing simulation primitives; no model is called.
"""
import copy
import itertools
import json
import math
import random
from functools import lru_cache

AXES = {
    'platform_count': list(range(1, 9)),
    'motion': ['stationary', 'moving'],
    'target_count': list(range(1, 13)),
    'distance': ['near', 'mid', 'far'],
    'path': ['straight', 'turn', 'curve'],
    'noise': ['low', 'normal', 'high'],
}
LABELS = {'stationary': '静止', 'moving': '移动', 'near': '近距', 'mid': '中距',
          'far': '远距', 'straight': '直线', 'turn': '折线', 'curve': '曲线',
          'low': '低噪声', 'normal': '中噪声', 'high': '高噪声'}
RANGE_LIMITS = {
    'target_speed': (0.1, 100), 'platform_speed': (0.1, 100),
    'heading': (0, 360), 'separation': (0, 50),
    'near': (.01, 100), 'mid': (.01, 100), 'far': (.01, 100),
    'amplitude': (.001, 10), 'sigma_hull': (.01, 90), 'sigma_tow': (.01, 90),
    'noise_low': (0, 10), 'noise_normal': (0, 10), 'noise_high': (0, 10),
    'clutter': (0, 10), 'tow_length': (0, 10000),
    'appear': (0, .9), 'disappear': (.1, 1),
}


def default_config():
    return dict(version=1, strategy='balanced', count=32, repeats=1, seed=20260911,
                duration=600, dt=1, bearing_step=.5, ref_lat=30, ref_lon=122,
                axes=dict(platform_count=[1,2], motion=['stationary','moving'],
                          target_count=[1,2,5], distance=['near','mid','far'],
                          path=['straight','turn','curve'], noise=['low','normal','high']),
                ranges=dict(target_speed=[8,28], platform_speed=[6,12], heading=[0,360],
                            separation=[.3,2], near=[1,4], mid=[4,8], far=[8,15],
                            amplitude=[.12,.32], sigma_hull=[3.5,6], sigma_tow=[1.3,2.4],
                            noise_low=[.012,.022], noise_normal=[.024,.038], noise_high=[.045,.075],
                            clutter=[.009,.016], tow_length=[100,250],
                            appear=[0,.15], disappear=[.85,1]))


def _number(v, lo, hi, name, integer=False):
    if type(v) not in (int,float) or not lo <= v <= hi or not math.isfinite(v):
        raise ValueError(f'{name}须在{lo}～{hi}之间')
    if integer and type(v) is not int:
        raise ValueError(f'{name}须为整数')
    return v


def normalize_config(raw):
    if not isinstance(raw, dict):
        raise ValueError('配置须为对象')
    c=default_config()
    for name in ('axes','ranges'):
        if name in raw:
            if not isinstance(raw[name],dict):raise ValueError(name+'须为对象')
            unknown=set(raw[name])-set(c[name])
            if unknown:raise ValueError('未知配置字段: '+','.join(sorted(unknown)))
            c[name].update(copy.deepcopy(raw[name]))
    for name in set(c)-{'axes','ranges'}:
        if name in raw:c[name]=copy.deepcopy(raw[name])
    if c['version'] != 1:raise ValueError('不支持的配置版本')
    if c['strategy'] not in ('balanced','exhaustive'):raise ValueError('未知采样方式')
    for name,lo,hi in [('count',1,10000),('repeats',1,1000),('seed',0,2147470000)]:
        _number(c[name],lo,hi,name,True)
    for name,lo,hi in [('duration',1,7200),('dt',.1,7200),('bearing_step',.05,10),
                       ('ref_lat',-80,80),('ref_lon',-170,170)]:
        _number(c[name],lo,hi,name)
    if c['dt']>c['duration']:raise ValueError('时间步长不能超过仿真时长')
    if (c['duration']/c['dt']+2)*(180/c['bearing_step']+2)>5_000_000:
        raise ValueError('采样网格超过500万，请增大时间或方位步长')
    for name,allowed in AXES.items():
        values=c['axes'][name]
        if not isinstance(values,list) or not values or len(values)>len(allowed):
            raise ValueError(name+'至少选择一项，且不能超过可用选项数')
        if any(type(v) is not type(allowed[0]) or v not in allowed for v in values):
            raise ValueError(name+'包含无效选项')
        if len(set(values)) != len(values):raise ValueError(name+'包含重复选项')
        c['axes'][name]=[v for v in allowed if v in values]
    for name,(lo,hi) in RANGE_LIMITS.items():
        pair=c['ranges'][name]
        if not isinstance(pair,list) or len(pair)!=2:raise ValueError(name+'需要最小和最大值')
        for value in pair:_number(value,lo,hi,name)
        if pair[0]>pair[1]:raise ValueError(name+'最小值不能大于最大值')
    if c['ranges']['appear'][1]>=c['ranges']['disappear'][0]:
        raise ValueError('最晚出现比例须小于最早消失比例')
    first=math.ceil(c['ranges']['appear'][1]*c['duration']/c['dt'])
    last=math.floor(c['ranges']['disappear'][0]*c['duration']/c['dt'])
    if first>last:
        raise ValueError('可见时间窗口内没有保证可用的采样点；减小时间步长或扩大出现/消失区间')
    # The simulator interpolates raw longitudes; reject regions that could cross
    # the date line rather than allowing a late failure or a 360-degree path.
    target_reach=max(c['ranges'][k][1] for k in c['axes']['distance'])
    target_reach+=c['ranges']['target_speed'][1]*.514444*c['duration']/1000*1.1
    platform_reach=c['ranges']['separation'][1] if max(c['axes']['platform_count'])>1 else 0
    if 'moving' in c['axes']['motion']:
        platform_reach+=c['ranges']['platform_speed'][1]*.514444*c['duration']/1000
    reach=max(target_reach,platform_reach)
    max_lat=abs(c['ref_lat'])+reach/111.19
    if max_lat>=89:
        raise ValueError('轨迹范围可能接近极点；降低纬度、航速、距离或仿真时长')
    longitude_margin=reach/(111.19*math.cos(math.radians(max_lat)))
    if abs(c['ref_lon'])+longitude_margin>=180:
        raise ValueError('轨迹范围可能跨越日期变更线（经度±180°）；调整参考经度、距离、航速或时长')
    total=math.prod(len(v) for v in c['axes'].values())
    if c['strategy']=='exhaustive' and total*c['repeats']>10000:
        raise ValueError(f'本批次{total*c["repeats"]:,}个样本，超过10000上限；减少组合或每组合数量')
    return c


@lru_cache(maxsize=24)
def _plan(serialized):
    c=normalize_config(json.loads(serialized))
    combinations=list(itertools.product(*(c['axes'][key] for key in AXES)))
    if c['strategy']=='balanced':random.Random(c['seed']).shuffle(combinations)
    return c,combinations


def _get_plan(raw):
    return _plan(json.dumps(raw,sort_keys=True,allow_nan=False))


def destination(point, az_deg, d_km):
    lat,lon=map(math.radians,point); az=math.radians(az_deg); d=d_km/6371
    la=math.asin(math.sin(lat)*math.cos(d)+math.cos(lat)*math.sin(d)*math.cos(az))
    lo=lon+math.atan2(math.sin(az)*math.sin(d)*math.cos(lat),math.cos(d)-math.sin(lat)*math.sin(la))
    return [math.degrees(la),math.degrees(lo)]


def distance_km(a,b):
    la,lb=math.radians(a[0]),math.radians(b[0])
    q=math.sin((lb-la)/2)**2+math.cos(la)*math.cos(lb)*math.sin(math.radians(b[1]-a[1])/2)**2
    return 12742*math.asin(min(1,math.sqrt(q)))


def describe(combo):
    return ' / '.join([f'{combo["platform_count"]}平台',LABELS[combo['motion']],
                       f'{combo["target_count"]}目标']+[LABELS[combo[k]] for k in ('distance','path','noise')])


def make_sample(raw,index):
    c,combinations=_get_plan(raw)
    if type(index) is not int or index<0:raise ValueError('样本索引无效')
    combo=dict(zip(AXES,combinations[index%len(combinations)]))
    combination_id='C'+'-'.join(str(AXES[k].index(combo[k])+1) for k in AXES)
    rng=random.Random(c['seed']*1000003+index)
    draw=lambda name:rng.uniform(*c['ranges'][name])
    origin=[c['ref_lat'],c['ref_lon']]
    platforms=[]
    for i in range(combo['platform_count']):
        point=origin if i==0 else destination(origin,rng.uniform(0,360),draw('separation'))
        platforms.append(dict(name=chr(65+i),start_ll=point,heading_deg=draw('heading'),
                              speed_kn=0 if combo['motion']=='stationary' else draw('platform_speed'),
                              hull_m=[0,0],tow_m=[-draw('tow_length'),0],color='#ffd400'))
    targets=[]
    for i in range(combo['target_count']):
        start=destination(origin,rng.uniform(0,360),draw(combo['distance']))
        speed=draw('target_speed'); az=rng.uniform(0,360)
        duration=c['duration']; length=speed*.514444*duration/1000
        path=combo['path']; waypoints=None; expr=None
        if path=='straight':waypoints=[destination(start,az,length*1.01)]
        elif path=='turn':
            mid=destination(start,az,length*.45)
            waypoints=[mid,destination(mid,az+rng.choice([-1,1])*rng.uniform(25,100),length*.56)]
        else:
            # Local drift plus lateral sinusoid. At t=0 the expression equals start.
            r=math.radians(az); v=speed*.514444
            vx,vy=v*math.sin(r),v*math.cos(r)
            period=rng.uniform(.5,1.5)*duration
            amp=v*.15*period/(2*math.pi)
            dx=f'({vx:.10f}*t+{amp*math.cos(r):.10f}*np.sin(2*np.pi*t/{period:.10f}))'
            dy=f'({vy:.10f}*t-{amp*math.sin(r):.10f}*np.sin(2*np.pi*t/{period:.10f}))'
            expr=f'({start[0]:.12f}+{dy}*{180/(math.pi*6371000):.14f}, {start[1]:.12f}+{dx}*{180/(math.pi*6371000*math.cos(math.radians(c["ref_lat"]))):.14f})'
        targets.append(dict(name=f'T{i+1}',start=start,speed_kn=speed,
                            path_type='curve' if path=='curve' else 'waypoints',waypoints=waypoints,
                            curve_func=expr,amp=draw('amplitude'),sigma_hull=draw('sigma_hull'),
                            sigma_tow=draw('sigma_tow'),t_on=duration*draw('appear'),
                            t_off=duration*draw('disappear'),color=['#ff9d00','#00ffff','#ff00cc','#aaff00'][i%4]))
    p=dict(T_MAX=c['duration'],DT=c['dt'],DB_STEP=c['bearing_step'],B_MIN=0,B_MAX=180,
           REF_LAT=c['ref_lat'],REF_LON=c['ref_lon'],NOISE_STD=draw('noise_'+combo['noise']),
           CLUTTER_LAM=draw('clutter'),DB_RANGE=[-5,16],SEED=c['seed']+index,
           SHOW_TRUTH_OVERLAY=False,EXPORT_TRUTH=True,platforms=platforms,targets=targets,
           _studio=dict(combination=combo,combination_id=combination_id,label=describe(combo),
                        distance_definition='initial_great_circle_distance_to_A_km',
                        cycle=index//len(combinations)+1))
    # Reject out-of-bounds paths early rather than silently emitting invalid samples.
    from btr_batch import validate_params
    validate_params(p)
    return p


def preview_config(raw):
    c,combinations=_get_plan(raw)
    n=len(combinations)
    total=c['count'] if c['strategy']=='balanced' else n*c['repeats']
    base,extra=divmod(total,n)
    platform_sum=base*sum(v[0] for v in combinations)+sum(v[0] for v in combinations[:extra])
    examples=[]
    for i in range(min(6,total)):
        p=make_sample(c,i)
        examples.append(dict(index=i+1,**p['_studio'],noise=round(p['NOISE_STD'],5),seed=p['SEED']))
    return dict(ok=True,config=copy.deepcopy(c),combinations=n,total=total,covered=min(n,total),
                min_per_combination=base,max_per_combination=base+bool(extra),
                input_images=2*platform_sum,all_images=3*platform_sum,
                examples=examples,output_root='btr_out',
                notes=['距离档按目标初始位置相对平台A定义，不是全程距离或CPA标签。',
                       '曲线航速是基准漂移航速；瞬时速度随曲线变化。'])


def auto_config(count, platform_count=1, target_count=2):
    """Fix platform and target counts per sample; randomize remaining dimensions."""
    import secrets
    _number(count, 1, 10000, '目标样本数量', True)
    _number(platform_count, 1, max(AXES['platform_count']), '平台数量', True)
    _number(target_count, 1, max(AXES['target_count']), '每个样本目标数量', True)
    c = default_config()
    c.update(count=count, seed=secrets.randbelow(2147470001), bearing_step=.25)
    c['axes']['platform_count'] = [platform_count]
    c['axes']['target_count'] = [target_count]
    return normalize_config(c)
