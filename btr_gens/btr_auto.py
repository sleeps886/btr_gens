"""Automatic batch orchestration; original simulation and batch modes are reused.

Only auto jobs use this worker. Other modes delegate to the existing BatchManager.
"""
import hashlib
import json
import math
import threading
import uuid
from datetime import datetime
from pathlib import Path

from btr_batch import BatchManager, platform_metadata, write_json
from btr_studio import make_sample


class StudioBatchManager(BatchManager):
    def start(self, request):
        if request.get('mode') != 'auto':
            return super().start(request)
        # Import here so tests and callers can inspect/inject the parameter factory.
        from btr_studio import auto_config
        config = auto_config(request.get('count'), platform_count=request.get('platform_count', 1),
                             target_count=request.get('target_count', 2))
        make_sample(config, 0)
        count = config['count']
        platform_count = config['axes']['platform_count'][0]
        target_count = config['axes']['target_count'][0]
        with self.lock:
            if any(j['state'] in ('queued', 'running', 'stopping') for j in self.jobs.values()):
                raise ValueError('已有批次运行中，请等待完成或停止后重试')
            job_id = 'batch_' + datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:6]
            directory = self.root / job_id
            directory.mkdir(parents=True)
            saved = dict(mode='auto', count=count, platform_count=platform_count, target_count=target_count, studio=config, seed=config['seed'])
            write_json(directory/'batch_config.json', saved)
            job = dict(id=job_id, mode='auto', state='queued', total=count, platform_count=platform_count, target_count=target_count, completed=0,
                       failed=0, current=0, attempted=0, stop=False, samples=[],
                       max_attempts=count+max(10, math.ceil(count*.2)), out_dir=str(directory))
            self.jobs[job_id] = job
            self._persist(job)
        threading.Thread(target=self._auto_worker, args=(job_id, config), daemon=True).start()
        return self.status(job_id)

    def _auto_sample(self, job, config, attempt):
        """Prepare input, call the untouched simulator, then package its real outputs."""
        root = Path(job['out_dir'])
        sample_dir = root/f'sample_{job["completed"]+1:04d}'
        entry = dict(sample_id=sample_dir.name, state='failed', attempt=attempt+1,
                     seed=config['seed']+attempt)
        params = {}
        sample_dir.mkdir()
        try:
            params = make_sample(config, attempt)
            params['OUT_DIR'] = str(sample_dir)
            entry.update(combination=params['_studio']['combination'], label=params['_studio']['label'])
            write_json(sample_dir/'meta.json', dict(params=params, state='running'))
            with self.run_lock:
                with self.lock:
                    cancelled = job['stop']
                if cancelled:
                    cancelled_dir = root/'cancelled_attempts'/f'attempt_{attempt+1:06d}'
                    cancelled_dir.parent.mkdir(exist_ok=True)
                    sample_dir.rename(cancelled_dir)
                    write_json(cancelled_dir/'meta.json', dict(params=params, state='cancelled'))
                    return None
                result = self.run(params)
            inputs, truth = sample_dir/'input', sample_dir/'truth'
            inputs.mkdir(); truth.mkdir()
            for platform in result['platforms']:
                for field in ('imgs', 'txts'):
                    for kind, filename in platform[field].items():
                        src = Path(filename)
                        dest = inputs if field == 'imgs' and kind in ('hull', 'tow') else truth
                        src.replace(dest/src.name)
            for src in sample_dir.glob('*_truth.jsonl'):
                src.replace(truth/src.name)
            write_json(inputs/'platform.json', platform_metadata(params))
            from btr_semantic_v1 import export_semantic
            export_semantic(sample_dir, params)
            files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(sample_dir.rglob('*')) if p.is_file() and p.name != 'meta.json'}
            entry.update(state='completed', files=files)
            write_json(sample_dir/'meta.json', dict(params=params, **entry))
            record = dict(sample_id=sample_dir.name,
                          platform_metadata=f'{sample_dir.name}/input/platform.json',
                          images=[p.relative_to(root).as_posix() for p in sorted(inputs.glob('*.png'))])
            # A failed append must not leave half a JSONL record in the input index.
            with (root/'model_inputs.jsonl').open('a+', encoding='utf-8') as stream:
                stream.seek(0, 2)
                offset = stream.tell()
                try:
                    stream.write(json.dumps(record, ensure_ascii=False)+'\n')
                    stream.flush()
                except Exception:
                    stream.seek(offset); stream.truncate()
                    raise
        except Exception as exc:
            entry.update(state='failed', error=str(exc))
            entry.pop('files', None)
            failed_dir = root/'failed_attempts'/f'attempt_{attempt+1:06d}'
            failed_dir.parent.mkdir(exist_ok=True)
            sample_dir.rename(failed_dir)
            sample_dir = failed_dir
            entry.update(sample_id=failed_dir.name, attempt_path=failed_dir.relative_to(root).as_posix())
            write_json(sample_dir/'meta.json', dict(params=params, **entry))
        entry['meta_sha256'] = hashlib.sha256((sample_dir/'meta.json').read_bytes()).hexdigest()
        return entry

    def _auto_worker(self, job_id, config):
        job = self.jobs[job_id]
        failures = 0
        try:
            with self.lock:
                if not job['stop']:
                    job['state'] = 'running'
                self._persist(job)
            for attempt in range(job['max_attempts']):
                with self.lock:
                    if job['stop'] or job['completed'] == job['total']:
                        break
                    job.update(current=job['completed']+1, attempted=attempt+1)
                    self._persist(job)
                entry = self._auto_sample(job, config, attempt)
                if entry is None:
                    break
                success = entry['state'] == 'completed'
                with self.lock:
                    job['samples'].append(entry)
                    job['completed' if success else 'failed'] += 1
                    self._persist(job)
                failures = 0 if success else failures+1
                if failures >= 5:
                    break
            with self.lock:
                job['state'] = 'stopped' if job['stop'] else ('completed' if job['completed'] == job['total'] else 'failed')
                if job['state'] == 'failed':
                    last = next((s['error'] for s in reversed(job['samples']) if s.get('error')), '补生成额度已用尽')
                    job['error'] = f"未达到目标数量：成功 {job['completed']}/{job['total']}。{last}"
                self._persist(job)
        except Exception as exc:
            with self.lock:
                job.update(state='failed', error=str(exc))
                self._persist(job)
