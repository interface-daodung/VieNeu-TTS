# VieNeu-TTS minimal — đã cắt gọn từ https://github.com/pnnbao97/VieNeu-TTS (Apache-2.0), xem NOTICE
.PHONY: run cls

export PYTHONUTF8 = 1
export PYTHONIOENCODING = utf-8

run:
	uv run python -m apps.web_stream

cls:
	uv run python -c "import os, shutil; hs=[os.environ.get('HF_HUB_CACHE'), os.path.join(os.environ.get('HF_HOME') or '', 'hub'), os.path.expanduser('~/.cache/huggingface/hub'), 'C:/root/.cache/huggingface/hub']; [shutil.rmtree(h, ignore_errors=True) for h in hs if h]; [shutil.rmtree(d, ignore_errors=True) for r, ds, _ in list(os.walk('.')) for d in ds if d == '__pycache__' and '.venv' not in r]; [shutil.rmtree(p, ignore_errors=True) for p in ('outputs', '.pytest_cache')]; print('cache cleared')"
