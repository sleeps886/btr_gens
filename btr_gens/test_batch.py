"""Run: python -m unittest test_batch -v (from this directory)."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import btr_gui_server as server
from btr_batch import BatchManager, sample_params, validate_params


def template():
    return dict(T_MAX=10, DT=1, DB_STEP=2, SEED=7,
        platforms=[dict(name='A',start_ll=[30,122],heading_deg=30,
                        speed_kn=0,hull_m=[0,0],tow_m=[-150,0])],
        targets=[dict(name='T1',start=[30.01,122.02],speed_kn=19,
                      amp=.3,sigma_hull=4,sigma_tow=1.6,
                      t_on=0,t_off=10,path_type='waypoints',
                      waypoints=[[30.01,122.01]])])


def wait_done(manager, job_id):
    for _ in range(1200):
        job = manager.status(job_id)
        if job['state'] not in ('queued','running','stopping'):
            return job
        time.sleep(.05)
    raise AssertionError('batch timed out')


class BatchTests(unittest.TestCase):
    def test_validation_and_reproducibility(self):
        p = template()
        validate_params(p)
        for key, value in [('DT',0), ('NOISE_STD',float('nan')), ('T_MAX',0)]:
            bad = copy.deepcopy(p); bad[key] = value
            with self.assertRaises(ValueError): validate_params(bad)
        bad = copy.deepcopy(p); bad['platforms'][0]['name']='../escape'
        with self.assertRaises(ValueError): validate_params(bad)
        a = sample_params(p,'perturb',22,15,0)
        self.assertEqual(a,sample_params(p,'perturb',22,15,0))
        self.assertNotEqual(a,sample_params(p,'perturb',23,15,0))
        self.assertEqual(p,template())
        for i in range(16):
            derived = sample_params(p,'scenarios',22,15,i)
            validate_params(derived)
            self.assertEqual(derived['_meta']['code'],f'S{i+1:02d}')

    def test_stop_and_failure(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent / 'work') as tmp:
            entered, release = threading.Event(), threading.Event()
            def broken(p):
                entered.set(); release.wait(5)
                raise ValueError('intentional failure')
            manager = BatchManager(tmp,broken,threading.Lock())
            job = manager.start(dict(count=3,params=template()))
            self.assertTrue(entered.wait(5))
            with self.assertRaises(ValueError): manager.start(dict(count=1,params=template()))
            manager.stop(job['id']); release.set()
            result = wait_done(manager,job['id'])
            self.assertEqual(result['state'],'stopped')
            self.assertEqual(result['failed'],1)
            self.assertEqual(len(result['samples']),1)
            saved=json.loads((Path(result['out_dir'])/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['state'],'stopped')

    def test_http_render_export_and_hashes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent / 'work') as tmp:
            manager = BatchManager(tmp, server.run_simulation, server.RUN_LOCK)
            with patch.object(server,'BATCHES',manager), patch.object(server,'BATCH_ROOT',tmp):
                http = ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
                thread = threading.Thread(target=http.serve_forever,daemon=True); thread.start()
                origin = f'http://127.0.0.1:{http.server_port}'
                def post(path, payload):
                    req = urllib.request.Request(origin+path,json.dumps(payload).encode(),{'Content-Type':'application/json'})
                    return json.load(urllib.request.urlopen(req,timeout=90))
                try:
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        post('/batch/start',dict(count=0,params=template()))
                    self.assertEqual(error.exception.code,400)
                    with patch.object(server.plt.Axes,'plot',autospec=True) as plot:
                        job=post('/batch/start',dict(count=2,params=template(),seed=22))['job']
                        result=wait_done(manager,job['id'])
                        self.assertEqual(result['state'],'completed',result)
                        # Only track map plots; no hull/tow truth overlays (5 calls/sample).
                        self.assertEqual(plot.call_count,10)
                    root=Path(result['out_dir'])
                    inputs=[json.loads(line) for line in (root/'model_inputs.jsonl').read_text(encoding='utf-8').splitlines()]
                    self.assertEqual(len(inputs),2)
                    for sample in result['samples']:
                        sample_dir=root/sample['sample_id']
                        semantic=json.loads((sample_dir/'semantic.json').read_text(encoding='utf-8'))
                        from btr_semantic import validate
                        validate(semantic)
                        self.assertTrue(semantic['quality']['schema_validation']['ok'])
                        self.assertEqual(semantic['scenario_summary']['target_total'],1)
                        self.assertEqual(semantic['platforms'][0]['targets'][0]['kinematics']['speed_kn'],19)
                        self.assertIsNone(semantic['platforms'][0]['targets'][0]['inferences']['depth_estimate_m']['value'])
                        bad=copy.deepcopy(semantic); bad['platforms'][0]['targets'][0]['kinematics'].pop('position_start')
                        with self.assertRaises(ValueError): validate(bad)
                        self.assertEqual(len(list((sample_dir/'input').glob('*.png'))),2)
                        self.assertEqual(len(list((sample_dir/'truth').glob('*.txt'))),3)
                        self.assertNotIn('targets',json.loads((sample_dir/'input/platform.json').read_text()))
                        truth=[json.loads(line) for line in (sample_dir/'truth/btr_A_truth.jsonl').read_text().splitlines()]
                        self.assertEqual(len(truth),11)
                        self.assertEqual(truth[-1]['time_s'],10)
                        for relative,digest in sample['files'].items():
                            self.assertEqual(hashlib.sha256((root/relative).read_bytes()).hexdigest(),digest)
                    url=origin+'/batch/files/'+job['id']+'/'+inputs[0]['images'][0]
                    self.assertEqual(urllib.request.urlopen(url).read(8),b'\x89PNG\r\n\x1a\n')
                    with self.assertRaises(urllib.error.HTTPError):
                        urllib.request.urlopen(origin+'/batch/files/%2e%2e/btr_gui_server.py')
                    status=json.load(urllib.request.urlopen(origin+'/batch/status/'+job['id']))
                    self.assertEqual(status['completed'],2)
                    # Legacy single-run endpoint still returns accessible, stable image URLs.
                    single=template(); single['OUT_DIR']=str(Path(tmp)/'single')
                    response=post('/run',single)
                    self.assertTrue(response['ok'])
                    self.assertEqual(urllib.request.urlopen(origin+response['platforms'][0]['imgs']['hull']).status,200)
                finally:
                    http.shutdown(); http.server_close(); thread.join()


if __name__ == '__main__':
    unittest.main()
