"""14.4 秒的底乐：A 小调铺底 + 每拍一次的软脉冲 + 命中处留白 + 结尾解决。

克制为主：不开旋律，只给和声与呼吸。
    python make_music.py  -> music.wav
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 48000
DUR = 15.65
N = int(DUR * SR)
t = np.arange(N) / SR
rng = np.random.default_rng(77)

L = np.zeros(N)
R = np.zeros(N)


def bp(x, lo, hi, order=2):
    b, a = signal.butter(order, [lo / (SR / 2), hi / (SR / 2)], btype="band")
    return signal.lfilter(b, a, x)


def lp(x, hi, order=2):
    b, a = signal.butter(order, hi / (SR / 2), btype="low")
    return signal.lfilter(b, a, x)


def add(t0, sig, gain=1.0, pan=0.0):
    i = int(t0 * SR)
    if i >= N:
        return
    seg = sig[: N - i]
    L[i:i + len(seg)] += seg * gain * np.sqrt((1 - pan) / 2)
    R[i:i + len(seg)] += seg * gain * np.sqrt((1 + pan) / 2)


def pad(freqs, t0, dur, gain=1.0, attack=0.35, release=0.6):
    """一组正弦 + 轻微失谐，慢起慢落，做成 pad。"""
    n = int(dur * SR)
    tt = np.arange(n) / SR
    env = np.minimum(1.0, tt / attack) * np.minimum(1.0, (dur - tt) / release)
    env = np.clip(env, 0, 1) ** 1.3
    sig = np.zeros(n)
    for f in freqs:
        for det in (-0.35, 0.0, 0.35):
            sig += np.sin(2 * np.pi * (f + det) * tt + rng.uniform(0, 6.28)) / len(freqs) / 3
    sig = lp(sig, 1800)
    add(t0, sig * env, gain, 0.0)


A2, C3, E3, A3 = 110.0, 130.81, 164.81, 220.0
F2, A2b, C4 = 87.31, 110.0, 261.63
E2, G2, B3 = 82.41, 98.0, 246.94

# ---------------- 和声：A 小调 4 段，每段 3.6 秒，互相搭接
pad([A2, E3, A3], 0.0, 3.9, 0.095, attack=1.0, release=1.2)
pad([F2, C3, A3], 3.3, 3.9, 0.085, attack=0.8, release=1.2)
pad([C4, E3, G2], 6.6, 3.4, 0.080, attack=0.6, release=1.0)
pad([E2, B3, E3], 9.4, 5.0, 0.095, attack=1.4, release=1.6)

# ---------------- 次低频：每 1.8 秒一下，稳住底盘
for i, f in enumerate([55.0, 55.0, 65.4, 65.4, 82.4, 82.4, 41.2]):
    t0 = 0.4 + i * 1.8
    if t0 > DUR:
        break
    n = int(0.9 * SR)
    tt = np.arange(n) / SR
    env = np.minimum(1.0, tt / 0.05) * np.exp(-tt / 0.45)
    add(t0, np.sin(2 * np.pi * f * tt) * env, 0.075, 0.0)

# ---------------- 软脉冲：每拍一下，只在扫描段（3.4–7.4）
for k in range(16):
    t0 = 3.4 + 0.25 * k
    n = int(0.10 * SR)
    tt = np.arange(n) / SR
    # 纯音色：一个低沉的"嗒"，不带噪声
    body = (np.sin(2 * np.pi * 156 * tt) * 0.8
            + np.sin(2 * np.pi * 312 * tt) * 0.2) * np.exp(-tt / 0.035)
    add(t0, body, 0.030 + 0.012 * k / 15, 0.0)

# ---------------- 命中：留白 0.25 秒，然后一记很低的和弦
gap = int(0.25 * SR)
i = int(7.85 * SR)
L[i:i + gap] *= np.linspace(1, 0.15, gap)
R[i:i + gap] *= np.linspace(1, 0.15, gap)
pad([A2, E3, A3], 8.05, 2.4, 0.075, attack=0.10, release=1.2)

# ---------------- 结尾：单音 A，慢慢淡出
pad([A2, A3, E3], 12.4, 3.2, 0.070, attack=0.40, release=1.8)

# ---------------- 母带
mix = np.stack([L, R], axis=1)
mix = np.tanh(mix * 1.1) * 0.9
fi = int(0.8 * SR)
mix[:fi] *= np.linspace(0, 1, fi)[:, None]
fo = int(1.1 * SR)
mix[-fo:] *= np.linspace(1, 0, fo)[:, None]
peak = np.max(np.abs(mix)) + 1e-9
mix = mix / peak * 0.60
wavfile.write("music.wav", SR, (mix * 32767).astype(np.int16))
print("music.wav", DUR, "s")
