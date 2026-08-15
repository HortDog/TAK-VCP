"""Run openwakeword's trainer with torchaudio I/O restored on Windows.

torchaudio >= 2.9 delegates load/save to torchcodec (FFmpeg DLLs required),
which is fragile on Windows. Every clip in this pipeline is plain 16 kHz PCM
wav, so soundfile covers it. Same CLI as openwakeword.train:

    uv run python training/train_model.py --training_config training/config/activate_tak.yml --augment_clips --train_model
"""

import runpy

import soundfile
import torch
import torchaudio


def _load(path, frame_offset=0, num_frames=-1, normalize=True,
          channels_first=True, format=None, **kwargs):
    data, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
    if frame_offset:
        data = data[frame_offset:]
    if num_frames is not None and num_frames > 0:
        data = data[:num_frames]
    return torch.from_numpy(data.T if channels_first else data), sr


def _save(path, src, sample_rate, channels_first=True, **kwargs):
    data = src.detach().cpu().numpy()
    soundfile.write(str(path), data.T if channels_first else data, sample_rate)


def _info(path, format=None, **kwargs):
    from types import SimpleNamespace

    meta = soundfile.info(str(path))
    return SimpleNamespace(
        sample_rate=meta.samplerate, num_frames=meta.frames,
        num_channels=meta.channels, bits_per_sample=16, encoding="PCM_S",
    )


def _trim_mmap(mmap_path):
    """Windows-safe openwakeword.data.trim_mmap: upstream os.remove()s the
    file while memmaps into it are still open, which Windows forbids."""
    import gc
    import os

    import numpy as np
    from numpy.lib.format import open_memmap

    src = np.load(mmap_path, mmap_mode="r")
    i = -1
    while np.all(src[i, :, :] == 0):
        i -= 1
    n_new = src.shape[0] + i + 1

    tmp_path = mmap_path + ".trim.npy"
    dst = open_memmap(tmp_path, mode="w+", dtype=np.float32,
                      shape=(n_new, src.shape[1], src.shape[2]))
    for start in range(0, n_new, 1024):
        end = min(start + 1024, n_new)
        dst[start:end] = src[start:end]
    dst.flush()
    del src, dst
    gc.collect()

    # The caller (compute_features_from_generator) still holds its own open
    # memmap of this file; it never uses it after trim_mmap returns, but
    # Windows can't delete a file with a live mapping — close them all.
    target = os.path.abspath(mmap_path)
    for obj in gc.get_objects():
        if isinstance(obj, np.memmap):
            filename = getattr(obj, "filename", None)
            if filename and os.path.abspath(filename) == target:
                try:
                    obj._mmap.close()
                except Exception:  # noqa: BLE001
                    pass

    os.remove(mmap_path)
    os.rename(tmp_path, mmap_path)


# Patch before openwakeword.train imports speechbrain/data helpers.
torchaudio.load = _load
torchaudio.save = _save
torchaudio.info = _info  # removed in torchaudio 2.11; torch_audiomentations needs it

import openwakeword.data  # noqa: E402
import openwakeword.utils  # noqa: E402

openwakeword.data.trim_mmap = _trim_mmap
openwakeword.utils.trim_mmap = _trim_mmap  # utils imported it by name

# speechbrain leaves LazyModule stubs in sys.modules for optional extras
# (e.g. k2). torch's custom-op registration walks sys.modules via inspect,
# and a mere hasattr(module, "__file__") probe triggers the lazy import and
# raises. Answer introspection probes with AttributeError instead.
from speechbrain.utils import importutils as _sb_importutils  # noqa: E402

_lazy_getattr = _sb_importutils.LazyModule.__getattr__


def _safe_lazy_getattr(self, attr):
    if attr in ("__file__", "__cached__"):
        raise AttributeError(attr)
    return _lazy_getattr(self, attr)


_sb_importutils.LazyModule.__getattr__ = _safe_lazy_getattr

import torch.utils.data as _tud  # noqa: E402

_OrigDataLoader = _tud.DataLoader


class _InProcessDataLoader(_OrigDataLoader):
    """train.py hardcodes num_workers=n_cpus, but its generator-backed
    IterableDataset can't be pickled to Windows spawn workers — load in-process."""

    def __init__(self, *args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs["persistent_workers"] = False
        kwargs.pop("prefetch_factor", None)
        super().__init__(*args, **kwargs)


_tud.DataLoader = _InProcessDataLoader

# Guarded so multiprocessing's spawned workers (which re-import this module,
# picking up the patches above) don't re-run the whole trainer.
if __name__ == "__main__":
    try:
        runpy.run_module("openwakeword.train", run_name="__main__")
    except ModuleNotFoundError as exc:
        # train.py's literal last statement converts onnx -> tflite via
        # onnx_tf/TensorFlow, which we don't install; the .onnx (the format
        # the runtime deploys) is already exported by then.
        if exc.name not in ("onnx_tf", "tensorflow"):
            raise
        print("[train_model] skipped tflite conversion (onnx is the deployed format)")
