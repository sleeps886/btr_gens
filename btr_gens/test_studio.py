"""Configurable sample studio regression tests."""
import copy
import json
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from unittest.mock import patch
from http.server import ThreadingHTTPServer

import btr_gui_server as server
from btr_batch import BatchManager, validate_params
from test_batch import wait_done


class StudioTests(unittest.TestCase):
    def studio(self):
        import btr_studio
        return btr_studio

    def test_default_preview_counts_and_balance(self):
        s = self.studio()
        cfg = s.default_config()
        cfg['count'] = 17
        preview = s.preview_config(cfg)
        self.assertGreater(preview['combinations'], 16)
        self.assertEqual(preview['total'], 17)
        self.assertEqual(preview['covered'], 17)
        self.assertEqual(preview, s.preview_config(cfg))

    def test_every_combination_and_reproducibility(self):
        s = self.studio(); cfg = s.default_config()
        cfg['axes'] = dict(platform_count=[1,2], motion=['stationary','moving'],
                           target_count=[1,3], distance=['near'], path=['straight','turn','curve'], noise=['normal'])
        cfg.update(strategy='exhaustive', repeats=2)
        plan = s.preview_config(cfg)
        self.assertEqual(plan['combinations'],24)
        self.assertEqual(plan['total'],48)
        samples = [s.make_sample(cfg,i) for i in range(48)]
        from collections import Counter
        codes = Counter(p['_studio']['combination_id'] for p in samples)
        self.assertEqual(set(codes.values()),{2})
        for p in samples:
            validate_params(p)
            self.assertEqual(p['T_MAX'],cfg['duration'])
        self.assertEqual(samples[0],s.make_sample(cfg,0))
        cfg['seed'] += 1
        self.assertNotEqual(samples[0],s.make_sample(cfg,0))

    def test_invalid_config(self):
        s = self.studio()
        for mutate in [lambda c:c['axes'].update(path=[]),
                       lambda c:c['axes'].update(platform_count=[9]),
                       lambda c:c['axes'].update(motion=['bogus']),
                       lambda c:c.update(count=10001),
                       lambda c:c['ranges'].update(target_speed=[30,10]),
                       lambda c:c.update(duration=float('nan')),
                       lambda c:c.update(dt=0),
                       lambda c:c['ranges'].update(near=[-1,3]),
                       lambda c:c['axes'].update(target_count=[True])]:
            cfg=s.default_config(); mutate(cfg)
            with self.assertRaises(ValueError): s.preview_config(cfg)

    def test_initial_distance_and_curves(self):
        s=self.studio(); cfg=s.default_config()
        cfg['axes'].update(platform_count=[2], target_count=[3],path=['curve'],distance=['far'],motion=['moving'])
        p=s.make_sample(cfg,0)
        from gen_train_ds import offset
        for target in p['targets']:
            distance=s.distance_km(p['platforms'][0]['start_ll'],target['start'])
            self.assertGreaterEqual(distance,cfg['ranges']['far'][0]-.001)
            self.assertLessEqual(distance,cfg['ranges']['far'][1]+.001)
            fn=server.parse_curve_expr(target['curve_func'])
            self.assertAlmostEqual(fn(0)[0],target['start'][0],places=7)
            self.assertAlmostEqual(fn(0)[1],target['start'][1],places=7)
        self.assertEqual(len(p['platforms']),2)
        self.assertTrue(all(pl['speed_kn']>0 for pl in p['platforms']))

    def test_visibility_window_has_sample_and_preview_isolated(self):
        s=self.studio(); cfg=s.default_config()
        cfg.update(duration=1,dt=1)
        with self.assertRaises(ValueError):s.preview_config(cfg)
        cfg=s.default_config()
        plan=s.preview_config(cfg)
        plan['config']['ranges']['near'][0]=99
        self.assertEqual(s.preview_config(cfg)['config']['ranges']['near'][0],1)

    def test_rejects_unsupported_antimeridian_before_start(self):
        s=self.studio(); cfg=s.default_config()
        cfg.update(ref_lat=70,ref_lon=170,duration=7200,dt=10,count=100,seed=1)
        cfg['ranges']['target_speed']=[100,100]
        cfg['axes'].update(platform_count=[1],target_count=[1],path=['straight'],motion=['stationary'])
        with self.assertRaisesRegex(ValueError,'经度|日期变更线'):
            s.preview_config(cfg)

    def test_custom_batch_and_http(self):
        s=self.studio(); cfg=s.default_config()
        cfg.update(count=1,duration=3,dt=1,bearing_step=5)
        cfg['axes'].update(platform_count=[2],target_count=[2],path=['turn'],motion=['stationary'])
        with tempfile.TemporaryDirectory() as tmp:
            manager=BatchManager(tmp,server.run_simulation,server.RUN_LOCK)
            with patch.object(server,'BATCHES',manager),patch.object(server,'BATCH_ROOT',tmp):
                http=ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
                t=threading.Thread(target=http.serve_forever,daemon=True);t.start()
                origin=f'http://127.0.0.1:{http.server_port}'
                def post(path,data):
                    req=urllib.request.Request(origin+path,json.dumps(data).encode(),{'Content-Type':'application/json'})
                    with urllib.request.urlopen(req,timeout=60) as response:return json.load(response)
                try:
                    self.assertEqual(urllib.request.urlopen(origin+'/studio').status,200)
                    self.assertEqual(post('/studio/preview',cfg)['total'],1)
                    with self.assertRaises(urllib.error.HTTPError):post('/studio/preview',{'axes':{'path':[]}})
                    job=post('/batch/start',dict(mode='custom',studio=cfg))['job']
                    result=wait_done(manager,job['id'])
                    self.assertEqual(result['state'],'completed',result)
                    self.assertEqual(result['completed'],1)
                    root=Path(result['out_dir']); sample=root/'sample_0001'
                    self.assertEqual(len(list((sample/'input').glob('*.png'))),4)
                    meta=json.loads((sample/'meta.json').read_text())
                    self.assertIn('_studio',meta['params'])
                    self.assertEqual(result['samples'][0]['combination']['platform_count'],2)
                    saved=json.loads((root/'batch_config.json').read_text())
                    self.assertEqual(saved['studio']['duration'],3)
                finally:http.shutdown();http.server_close();t.join()


if __name__=='__main__':unittest.main()
