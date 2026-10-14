"""Exercise current-stage tools and file speech with the selected managed model."""
import argparse
import asyncio
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import httpx
from loguru import logger
from sqlalchemy import select
from app.adapters.llama import LlamaConfig
from app.memory_runtime import open_memory
from app.records import Records
from app.reminders import ReminderService, _NOTIFICATIONS
from app.session import AssistantSession
from app.vision_frames import VisionFrameStore
from app.voice_assistant import VoiceAssistant
from deploy import services
from tools.project import CACHE, sha256, stamp, write_json
from tools.voice_assistant_smoke import readback


def called(result, name):
    return any(c['tool'] == name and c['outcome'].get('ok') for c in result['calls'])


async def main(args):
    if args.output.exists() or args.run_dir.exists():
        raise RuntimeError('Preserve previous runs; use fresh paths')
    logger.remove(); logger.add(sys.stderr, level='WARNING')
    manifest_path = ROOT / 'models/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    entry = manifest['models']['vlm']
    expected = CACHE / 'models' / next(f['name'] for f in entry['files'] if not f['name'].startswith('mmproj'))
    with httpx.Client(timeout=5, trust_env=False) as client:
        response = client.get('http://127.0.0.1:8088/props'); response.raise_for_status()
        props = response.json()
    if Path(props['model_path']).resolve() != expected.resolve():
        raise RuntimeError('Actual model differs from the selected default')
    before = services.state(list(services.ORDER), health=True)
    if before['backend_status'] != 'READY':
        raise RuntimeError('Managed backends must be ready')
    args.run_dir.mkdir(parents=True, mode=0o700)
    prior = json.loads((ROOT / 'benchmarks/results/voice-assistant-smoke.json').read_text())
    inputs = []
    for index, label in [(2, 'time'), (3, 'image'), (4, 'reminder'), (5, 'arithmetic')]:
        path = Path(prior['artifacts']) / f'input-{index}.wav'
        assert sha256(path) == prior['cases'][index]['input_sha256']
        inputs.append((label, path))
    report = {'created_at': stamp(), 'status': 'RUNNING', 'model_manifest_sha256': sha256(manifest_path),
        'source_sha256': sha256(Path(__file__)), 'model_path': props['model_path'],
        'application_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in sorted((ROOT / 'app').rglob('*.py'))},
        'backends_before': before, 'typed': [], 'voice': [], 'errors': [], 'artifacts': str(args.run_dir),
        'scope': 'Eight current-stage typed/file-speech integration turns with real model, SDK stores and TTS/ASR. Synthetic blue RGB-D transport and synthetic WAV inputs; existing empty memory adapter retained, no long-term memory quality/capacity work. Not physical audio/AEC, natural image QA, independent tool quality, latency or JT acceptance.'}
    write_json(args.output, report)
    (args.run_dir / 'probe-source.py').write_bytes(Path(__file__).read_bytes())
    frames = VisionFrameStore(source_kind='synthetic_fixture')
    stop = asyncio.Event()
    async def feed():
        rgb = bytes((0, 0, 255)) * (640 * 480)
        depth = bytes(640 * 480 * 2)
        while not stop.is_set():
            now = time.monotonic_ns()
            frames.add(source_time_ns=now, capture_time_ns=now, arrival_time_ns=now,
                width=640, height=480, rgb_step=1920, rgb_encoding='rgb8', rgb=rgb,
                depth_step=1280, depth=depth, optical_frame='PHASE_ONE_BLUE_FIXTURE')
            await asyncio.sleep(1 / 30)
    feeder = asyncio.create_task(feed())
    checks = {}
    try:
        runtime = json.loads((ROOT / 'configs/runtime.yaml').read_text())
        config = LlamaConfig.from_manifest(manifest_path)
        with ExitStack() as stack:
            memory = stack.enter_context(open_memory(args.run_dir / 'data/memory', settings=runtime))
            records = stack.enter_context(Records(args.run_dir / 'data/records.sqlite'))
            reminders = stack.enter_context(ReminderService(args.run_dir / 'data/reminders.sqlite'))
            session = AssistantSession(config, memory, reminders, records, user_id='owner')
            try:
                await session.start()
                async def emit(_): pass
                for text in ['保存笔记，标题是阶段检查，内容是明天整理工作台。',
                             '列出我的笔记。', '五分钟后提醒我检查灯光。', '取消检查灯光的提醒。']:
                    result = await session.ask(text, emit=emit)
                    report['typed'].append(result)
                    write_json(args.output, report)
                typed = report['typed']
                checks.update(note_saved=called(typed[0], 'create_note'),
                    note_listed=called(typed[1], 'list_notes') and '阶段检查' in typed[1]['response'] and '整理工作台' in typed[1]['response'],
                    reminder_created=called(typed[2], 'create_reminder'),
                    reminder_cancelled=called(typed[3], 'cancel_reminder'))
                checks['no_active_typed_reminder'] = not reminders.list_reminders('owner')
                checks['one_note_stored'] = len(records.list_notes('owner')) == 1
                checks['typed_tool_calls_not_batched'] = all(len(row['calls']) == 1 for row in typed)
            finally:
                await session.close()
            voice = VoiceAssistant(config, memory, reminders, records, vision_frames=frames,
                output_path=args.run_dir / 'output.wav')
            try:
                await voice.start()
                for label, path in inputs:
                    result = await voice.run_file(path)
                    report['voice'].append({'label': label, 'input_sha256': sha256(path), **result})
                    write_json(args.output, report)
                    print(json.dumps({'voice': label, 'response': result['response']}, ensure_ascii=False), flush=True)
                deadline = time.monotonic() + 25
                while not voice.notifications.results:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Scheduled file notification did not finish')
                    await asyncio.sleep(.1)
                rows = report['voice']
                with reminders.engine.connect() as db:
                    notices = [dict(row) for row in db.execute(select(_NOTIFICATIONS)).mappings()]
                report['notifications'] = notices
                report['notification_receipts'] = list(voice.notifications.results)
                checks.update(spoken_time=called(rows[0], 'get_time'),
                    spoken_image=rows[1]['image'] is not None and '蓝' in rows[1]['response'],
                    spoken_reminder=called(rows[2], 'create_reminder'),
                    spoken_arithmetic=any(word in rows[3]['response'] for word in ['2', '二', '两']),
                    all_audio_outputs=all(row['speech_receipt']['bytes_written'] > 0 for row in rows),
                    reminder_file_receipt=any(n['text'] == '喝水' and n['status'] == 'fixture_delivered' and n['played_utc'] is None for n in notices),
                    cancelled_reminder_not_delivered=not any(n['text'] == '检查灯光' and n['status'] == 'fixture_delivered' for n in notices),
                    native_token_counts=all(row['prompt_tokens'] == [u['prompt_tokens'] for u in row['model_usage']] for row in [*report['typed'], *rows]),
                    no_pipeline_errors=not voice.errors,
                    voice_tool_calls_not_batched=all(len(row['calls']) <= 1 for row in rows),
                    no_long_term_facts_written=not memory.list_memories('owner'))
            finally:
                await voice.close()
        with wave.open(str(args.run_dir / 'output.wav')) as source:
            for index, row in enumerate(report['voice']):
                receipt = row['speech_receipt']
                source.setpos(receipt['first_sample'])
                pcm = source.readframes(receipt['bytes_written'] // 2)
                path = args.run_dir / f'answer-{index}.wav'
                with wave.open(str(path), 'wb') as out:
                    out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000); out.writeframes(pcm)
                row['audio_readback'] = await readback(path)
                row['output_sha256'] = sha256(path)
        checks['audio_image_readback'] = '蓝' in report['voice'][1]['audio_readback']
        checks['no_image_leases'] = frames.status()['leased_frames'] == 0
    except Exception as error:
        report['errors'].append({'type': type(error).__name__, 'message': str(error)})
    finally:
        stop.set(); await feeder
        report['backends_after'] = services.state(list(services.ORDER), health=True)
        checks['backends_ready'] = report['backends_after']['backend_status'] == 'READY'
        checks['model_unchanged'] = sha256(manifest_path) == report['model_manifest_sha256']
        report.update(checks=checks, completed_at=stamp(), status='PASS' if checks and all(checks.values()) and not report['errors'] else 'FAIL')
        write_json(args.output, report)
    print(json.dumps({'status': report['status'], 'checks': checks, 'errors': report['errors']}, ensure_ascii=False))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(asyncio.run(main(parser.parse_args())))
