"""User-supplied schema_v1.json + field dictionary export contract."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


def fixture(directory):
    truth=directory/'truth';truth.mkdir(parents=True)
    params=dict(T_MAX=100,DT=1,DB_STEP=.25,NOISE_STD=.03,CLUTTER_LAM=.012,DB_RANGE=[-5,16],SEED=42,
        platforms=[dict(name='A',start_ll=[30,122],heading_deg=90,speed_kn=0,hull_m=[0,0],tow_m=[-150,0])],
        targets=[],_studio=dict(combination_id='C-test',label='test',combination=dict(platform_count=1,target_count=3,motion='stationary')))
    for i,speed in enumerate([25,12,11]):
        params['targets'].append(dict(name=f'T{i+1}',start=[30.01,122.01],speed_kn=speed,
            raw_label='中/机动' if i==0 else None,amp=.1 if i==0 else .3,
            sigma_hull=4+i,sigma_tow=1.4+i*.2,path_type='waypoints',waypoints=[[30.01,122.02]],
            t_on=10 if i==0 else 0,t_off=60 if i==0 else 100))
    (truth/'btr_A_track.txt').write_text('生成时间: 2026-11-16 15:00:00\n',encoding='utf-8')
    for kind in ['hull','tow']:
        lines=['header', 'T1 | 舷角 10.0°->30.0° | CPA= 0.40 km @ t= 30s',
               'T2 | 舷角 20.0°->21.0° | CPA= 5.00 km @ t= 100s',
               'T3 | 舷角 30.0°->50.0° | CPA= 16.00 km @ t= 0s']
        (truth/f'btr_A_{kind}.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (directory/'meta.json').write_text(json.dumps(dict(state='completed',params=params,files={})),encoding='utf-8')
    return params


class SemanticV1Tests(unittest.TestCase):
    def adapter(self):
        import btr_semantic_v1
        return btr_semantic_v1

    def test_schema_and_documented_rules(self):
        s=self.adapter()
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'sample_0001';p=fixture(directory)
            doc=s.build_semantic(directory,p)
            self.assertEqual(s.SCHEMA_PATH,Path(__file__).resolve().parent.parent/'schema_v1.json')
            s.validate(doc)
            self.assertEqual(doc['quality']['schema_validation'],dict(schema='schema_v1.json',ok=True))
            ts=doc['platforms'][0]['targets']
            self.assertEqual([t['inferences']['target_type']['rule_id'] for t in ts],['RULE-TYPE-01','RULE-TYPE-02','RULE-TYPE-03'])
            self.assertEqual([t['inferences']['motion_model']['value'] for t in ts],['CT','CV','CA'])
            self.assertEqual(ts[0]['observations']['hull']['bearing_rate_deg_per_s'],.4)
            self.assertEqual(ts[0]['observations']['hull']['bearing_start_deg'],10)
            self.assertEqual(ts[0]['inferences']['threat_level']['value'],'高')
            self.assertEqual(ts[0]['inferences']['threat_level']['confidence'],.6)
            self.assertEqual(ts[1]['inferences']['distance_class']['value'],'近')
            self.assertEqual(ts[2]['inferences']['distance_class']['rule_id'],'RULE-DIST-04')
            self.assertEqual(doc['source']['dataset']['scenario_code'],'C-test')
            self.assertEqual(doc['sim_settings']['beam_cells'],721)
            self.assertIn('sea_state',doc['environment'])
            self.assertEqual(doc['platforms'][0]['self']['array_hull']['beam_sigma_deg'],[4,6])
            for t in ts:
                self.assertIsNone(t['inferences']['depth_estimate_m']['value'])
                for inference in t['inferences'].values():
                    self.assertEqual(set(inference),{'value','method','rule_id','confidence'})
            self.assertNotIn('水面',doc['summary_nl'])
            self.assertNotIn('水面',doc['platforms'][0]['summary_nl'])

    def test_actual_schema_rejects_wrong_output(self):
        s=self.adapter()
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'sample_0001';p=fixture(d);doc=s.build_semantic(d,p)
            for mutate in [lambda x:x.pop('platforms'),lambda x:x['platforms'][0]['targets'][0]['inferences']['target_type'].update(confidence=2),lambda x:x['scenario_summary'].update(target_total='3')]:
                bad=copy.deepcopy(doc);mutate(bad)
                with self.assertRaises(ValueError):s.validate(bad)

    def test_unknown_and_zero_values_stay_explicit(self):
        s=self.adapter()
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'sample_0001';p=fixture(d)
            p.pop('NOISE_STD');p['platforms'][0]['speed_kn']=8
            p['targets'][0].update(t_on=20,t_off=20)
            doc=s.build_semantic(d,p)
            self.assertIsNone(doc['sim_settings']['noise_std'])
            self.assertIn('默认',doc['platforms'][0]['targets'][0]['inferences']['snr_est_db']['method'])
            self.assertNotIn('position_end',doc['platforms'][0]['self'])
            self.assertTrue(doc['platforms'][0]['warnings'])
            self.assertEqual(doc['platforms'][0]['targets'][0]['observations']['hull']['bearing_rate_deg_per_s'],0)
            p['NOISE_STD']=0
            self.assertIsNone(s.build_semantic(d,p)['platforms'][0]['targets'][0]['inferences']['snr_est_db']['value'])

    def test_both_generators_use_user_schema_export(self):
        import threading
        import btr_gui_server as server
        from btr_batch import BatchManager
        from btr_auto import StudioBatchManager
        from btr_studio import auto_config
        from test_batch import template,wait_done
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            managers=[(BatchManager(tmp,server.run_simulation,server.RUN_LOCK),dict(count=1,params=template()))]
            config=auto_config(1);config.update(duration=3,bearing_step=5)
            config['axes'].update(platform_count=[1],target_count=[1])
            managers.append((StudioBatchManager(tmp,server.run_simulation,server.RUN_LOCK),dict(mode='auto',count=1)))
            with patch('btr_studio.auto_config',return_value=config):
                for manager,request in managers:
                    job=manager.start(request);result=wait_done(manager,job['id'])
                    self.assertEqual(result['state'],'completed',result)
                    doc=json.loads((Path(result['out_dir'])/'sample_0001/semantic.json').read_text())
                    self.assertEqual(doc['quality']['schema_validation']['schema'],'schema_v1.json')
                    self.adapter().validate(doc)
                    self.assertIn(doc['platforms'][0]['targets'][0]['inferences']['motion_model']['value'],['CV','CA','CT'])

    def test_backup_must_not_overlap_source_batch(self):
        s=self.adapter()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);batch=root/'batch_20261116_150023_123abc';d=batch/'sample_0001';fixture(d)
            original=b'{"original":true}';(d/'semantic.json').write_bytes(original)
            (batch/'manifest.json').write_text(json.dumps(dict(state='completed',samples=[dict(sample_id='sample_0001',state='completed',files={})])))
            for backup in (batch.parent,batch/'nested'):
                with self.assertRaises(ValueError):s.backfill_batch(batch,backup)
                self.assertEqual((d/'semantic.json').read_bytes(),original)

    def test_backfill_updates_hashes_without_touching_inputs(self):
        s=self.adapter()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);batch=root/'batch_20261116_150023_123abc';d=batch/'sample_0001';p=fixture(d)
            inputs=d/'input';inputs.mkdir();image=inputs/'btr_A_hull.png';image.write_bytes(b'original image bytes')
            original={f:f.read_bytes() for f in d.rglob('*') if f.is_file()}
            old_semantic=d/'semantic.json';old_semantic.write_text('{"previous":true}',encoding='utf-8')
            meta=json.loads((d/'meta.json').read_text());meta['files']={'sample_0001/input/btr_A_hull.png':hashlib.sha256(image.read_bytes()).hexdigest()};(d/'meta.json').write_text(json.dumps(meta))
            manifest=dict(id=batch.name,state='completed',completed=1,total=1,samples=[dict(sample_id='sample_0001',state='completed',files=dict(meta['files']))])
            (batch/'manifest.json').write_text(json.dumps(manifest))
            backup=root/'backups'
            result=s.backfill_batch(batch,backup)
            self.assertEqual(result['updated'],1)
            self.assertTrue((backup/batch.name/'sample_0001/semantic.json').is_file())
            for f,data in original.items():
                if f.name!='meta.json':self.assertEqual(f.read_bytes(),data)
            meta=json.loads((d/'meta.json').read_text())
            self.assertEqual(meta['params'],p)
            m=json.loads((batch/'manifest.json').read_text());sample=m['samples'][0]
            self.assertEqual(sample['files']['sample_0001/semantic.json'],hashlib.sha256(old_semantic.read_bytes()).hexdigest())
            self.assertEqual(sample['meta_sha256'],hashlib.sha256((d/'meta.json').read_bytes()).hexdigest())
            (batch/'manifest.json').write_text(json.dumps(dict(manifest,state='running')))
            with self.assertRaises(ValueError):s.backfill_batch(batch,backup)


if __name__=='__main__':unittest.main()
