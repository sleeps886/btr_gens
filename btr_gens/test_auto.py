"""Automatic generation: fixed platform count, random parameters and bounded retries."""
import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import btr_gui_server as server
from btr_batch import validate_params
from btr_auto import StudioBatchManager as BatchManager
from test_batch import wait_done


class AutoTests(unittest.TestCase):
    def test_count_only_configuration(self):
        from btr_studio import auto_config, make_sample
        c=auto_config(7)
        self.assertEqual(c['count'],7)
        self.assertEqual(c['strategy'],'balanced')
        self.assertEqual(c['duration'],600)
        self.assertEqual(c['bearing_step'],.25)
        self.assertEqual(c['axes']['platform_count'],[1])
        for i in range(7):
            p=make_sample(c,i)
            validate_params(p)
            self.assertEqual(len(p['platforms']),1)
        with patch('secrets.randbelow',side_effect=[12,13]):
            self.assertNotEqual(auto_config(7)['seed'],auto_config(7)['seed'])
        for n in (0,-1,10001,True,1.5,'5'):
            with self.assertRaises(ValueError):auto_config(n)

    def test_fixed_platform_count_and_validation(self):
        from btr_studio import auto_config, make_sample
        for n in (2,8):
            c=auto_config(2,platform_count=n)
            self.assertEqual(c['axes']['platform_count'],[n])
            for i in range(2):
                self.assertEqual(len(make_sample(c,i)['platforms']),n)
        for n in (None,0,-1,9,True,1.5,'2'):
            with self.subTest(platform_count=n), self.assertRaisesRegex(ValueError,'平台数量'):
                auto_config(2,platform_count=n)

    def test_fixed_target_count_and_validation(self):
        from btr_studio import auto_config, make_sample
        for n in (1,4,12):
            c=auto_config(3,platform_count=2,target_count=n)
            self.assertEqual(c['axes']['target_count'],[n])
            for i in range(3):
                sample=make_sample(c,i)
                self.assertEqual(len(sample['targets']),n)
                self.assertEqual(len(sample['platforms']),2)
        for n in (None,0,-1,13,True,1.5,'2'):
            with self.subTest(target_count=n), self.assertRaisesRegex(ValueError,'目标数量'):
                auto_config(2,target_count=n)

    def test_failed_sample_is_replaced_until_success_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]
            def run(p):
                calls.append(p)
                if len(calls)==1:raise ValueError('one transient failure')
                return server.run_simulation(p)
            manager=BatchManager(tmp,run,server.RUN_LOCK)
            from btr_studio import auto_config
            c=auto_config(2,platform_count=2,target_count=4);c.update(duration=3,bearing_step=5)
            with patch('btr_studio.auto_config',return_value=c):
                job=manager.start({'mode':'auto','count':2,'platform_count':2,'target_count':4})
            result=wait_done(manager,job['id'])
            self.assertEqual(result['state'],'completed',result)
            self.assertEqual(result['completed'],2)
            self.assertEqual(result['failed'],1)
            self.assertEqual(result['attempted'],3)
            self.assertEqual(result['mode'],'auto')
            root=Path(result['out_dir'])
            self.assertEqual(len(list(root.glob('sample_*'))),2)
            self.assertEqual(len(list((root/'failed_attempts').iterdir())),1)
            records=[json.loads(s) for s in (root/'model_inputs.jsonl').read_text().splitlines()]
            self.assertEqual([r['sample_id'] for r in records],['sample_0001','sample_0002'])
            self.assertNotEqual(calls[0]['SEED'],calls[1]['SEED'])
            self.assertEqual([len(p['platforms']) for p in calls],[2,2,2])
            self.assertEqual([len(p['targets']) for p in calls],[4,4,4])
            self.assertEqual(result['target_count'],4)
            for r in records:
                self.assertEqual({Path(path).name for path in r['images']},
                                 {f'btr_{p}_{kind}.png' for p in ('A','B') for kind in ('hull','tow')})
                metadata=json.loads((root/r['platform_metadata']).read_text())
                self.assertNotIn('DB_STEP',metadata)
                self.assertEqual(len(metadata['platforms']),2)
                self.assertEqual(json.loads((root/r['sample_id']/'meta.json').read_text())['params']['DB_STEP'],5)
                self.assertEqual({p.name for p in (root/r['sample_id']/'truth').glob('*.png')},
                                 {'btr_A_track.png','btr_B_track.png'})
                for path in r['images']:self.assertTrue((root/path).is_file())

    def test_permanent_failure_stops_without_claiming_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            def broken(p):raise OSError('disk unavailable')
            manager=BatchManager(tmp,broken,threading.Lock())
            job=manager.start({'mode':'auto','count':1})
            result=wait_done(manager,job['id'])
            self.assertEqual(result['state'],'failed')
            self.assertEqual(result['completed'],0)
            self.assertLessEqual(result['attempted'],5)
            self.assertIn('disk unavailable',result['error'])

    def test_index_failure_does_not_count_as_success(self):
        from btr_studio import auto_config
        c=auto_config(1);c.update(duration=3,bearing_step=5)
        c['axes'].update(platform_count=[1],target_count=[1])
        original=Path.open
        rejected=[]
        def open_once(path,*args,**kwargs):
            if path.name=='model_inputs.jsonl' and args and args[0]=='a+' and not rejected:
                rejected.append(True)
                raise OSError('index temporarily unavailable')
            return original(path,*args,**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            manager=BatchManager(tmp,server.run_simulation,server.RUN_LOCK)
            with patch('btr_studio.auto_config',return_value=c),patch.object(Path,'open',open_once):
                job=manager.start({'mode':'auto','count':1})
                result=wait_done(manager,job['id'])
            self.assertEqual(result['state'],'completed',result)
            self.assertEqual(result['completed'],1)
            self.assertEqual(result['failed'],1)
            self.assertEqual(result['attempted'],2)
            root=Path(result['out_dir'])
            self.assertEqual(len((root/'model_inputs.jsonl').read_text().splitlines()),1)
            self.assertEqual(len(list(root.glob('sample_*'))),1)

    def test_http_platform_count_and_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            entered,release=threading.Event(),threading.Event()
            received=[]; received_targets=[]
            def waiting(p):received_targets.append(len(p['targets']));received.append(len(p['platforms']));entered.set();release.wait(5);raise ValueError('stopped test')
            manager=BatchManager(tmp,waiting,threading.Lock())
            with patch.object(server,'BATCHES',manager),patch.object(server,'BATCH_ROOT',tmp):
                http=ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
                t=threading.Thread(target=http.serve_forever,daemon=True);t.start()
                origin=f'http://127.0.0.1:{http.server_port}'
                try:
                    with urllib.request.urlopen(origin+'/studio') as r:
                        page=r.read().decode();self.assertIn('目标样本数量',page);self.assertIn('/quick.js',page)
                        self.assertIn('for="quickPlatformCount">平台数量',page)
                        self.assertIn('id="quickTargetCount"',page)
                    self.assertEqual(urllib.request.urlopen(origin+'/studio/advanced').status,200)
                    def start(n,target=4):
                        req=urllib.request.Request(origin+'/batch/start',json.dumps({'mode':'auto','count':3,'platform_count':n,'target_count':target}).encode(),{'Content-Type':'application/json'})
                        with urllib.request.urlopen(req) as r:return json.load(r)['job']
                    with self.assertRaises(urllib.error.HTTPError) as invalid:start(9)
                    with invalid.exception as response:
                        self.assertEqual(response.code,400)
                        self.assertIn('平台数量',json.load(response)['error'])
                    with self.assertRaises(urllib.error.HTTPError) as invalid_target:start(1,13)
                    with invalid_target.exception as response:
                        self.assertEqual(response.code,400)
                        self.assertIn('目标数量',json.load(response)['error'])
                    job=start(3)
                    self.assertTrue(entered.wait(5))
                    self.assertEqual(job['total'],3)
                    self.assertEqual(job['platform_count'],3)
                    self.assertEqual(job['target_count'],4)
                    self.assertEqual(received_targets,[4])
                    self.assertEqual(received,[3])
                    config=json.loads((Path(job['out_dir'])/'batch_config.json').read_text())
                    self.assertEqual(config['studio']['axes']['platform_count'],[3])
                    self.assertEqual(config['studio']['axes']['target_count'],[4])
                    manager.stop(job['id']);release.set()
                    result=wait_done(manager,job['id'])
                    self.assertEqual(result['state'],'stopped')
                    self.assertLess(result['completed'],3)
                finally:release.set();http.shutdown();http.server_close();t.join()


if __name__=='__main__':unittest.main()
