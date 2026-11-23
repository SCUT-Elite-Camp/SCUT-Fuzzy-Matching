"""给 14.4 秒的片子铺一条克制的底：低频床 + 启动气声 + 扫描 tick + 落定闷响 + 结果和弦 + 标题气浪。

    python make_track.py      -> soundtrack.wav
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 48000
DUR = 15.65
N = int(DUR * SR)
t = np.arange(N) / SR
rng = np.random.default_rng(2024)

L = np.zeros(N)
R = np.zeros(N)


def bp(x, lo, hi, order=2):
    b, a = signal.butter(order, [lo / (SR / 2), hi / (SR / 2)], btype="band")
    return signal.lfilter(b, a, x)


def add(t0, sig, gain=1.0, pan=0.0):
    """pan: -1 左 / +1 右"""
    i = int(t0 * SR)
    if i >= N:
        return
    seg = sig[: N - i]
    gl = gain * np.sqrt((1 - pan) / 2)
    gr = gain * np.sqrt((1 + pan) / 2)
    L[i:i + len(seg)] += seg * gl
    R[i:i + len(seg)] += seg * gr


# ---------------------------------------------------------------- 1. 低频床
bed = (0.050 * np.sin(2 * np.pi * 55.0 * t)
       + 0.034 * np.sin(2 * np.pi * 82.41 * t + 0.7)
       + 0.018 * np.sin(2 * np.pi * 110.0 * t + 1.3))
bed *= 0.72 + 0.28 * np.sin(2 * np.pi * 0.055 * t)
fi = int(1.4 * SR)
bed[:fi] *= np.linspace(0, 1, fi) ** 1.5
fo = int(0.9 * SR)
bed[-fo:] *= np.linspace(1, 0, fo) ** 1.2
add(0.0, bed, 1.0, 0.0)

# ---------------------------------------------------------------- 2. 启动气声
n = int(1.1 * SR)
tt = np.arange(n) / SR
# 用一组缓慢上行的正弦代替噪声，听起来是"气"不是"沙"
sw = np.zeros(n)
for f0, g in ((196.0, 0.5), (293.7, 0.3), (392.0, 0.2)):
    f = f0 * (1 + 0.06 * tt)
    sw += g * np.sin(2 * np.pi * np.cumsum(f) / SR)
e = np.minimum(1.0, tt / 0.35) * np.exp(-np.maximum(0, tt - 0.35) / 0.35)
add(0.12, sw * e, 0.045, -0.15)
n2 = int(0.9 * SR)
rise = np.sin(2 * np.pi * np.cumsum(np.linspace(90, 240, n2)) / SR) * np.exp(-np.linspace(0, 3, n2))
add(0.20, rise, 0.035, 0.1)

# ---------------------------------------------------------------- 3. 扫描 tick（16 下）
for k in range(16):
    t0 = 3.4 + 0.25 * k
    n = int(0.045 * SR)
    tt = np.arange(n) / SR
    click = (np.sin(2 * np.pi * 1760 * tt) * 0.7
             + np.sin(2 * np.pi * 2640 * tt) * 0.3) * np.exp(-tt / 0.006)
    g = 0.062 + 0.030 * (k / 15.0)          # 机械声加大
    add(t0, click, g, -0.35 + 0.7 * (k / 15.0))
    # 每 4 格加一记更低的棘轮声，让"对齐"有重量
    if k % 4 == 0:
        nr = int(0.09 * SR)
        tr = np.arange(nr) / SR
        ratchet = (np.sin(2 * np.pi * 220 * tr) * 0.6
                   + np.sin(2 * np.pi * 330 * tr) * 0.4) * np.exp(-tr / 0.025)
        add(t0, ratchet, 0.055, -0.2)

# ---------------------------------------------------------------- 4. 命中（软）
n = int(0.9 * SR)
tt = np.arange(n) / SR
f = 52 + 28 * np.exp(-tt * 10)
thump = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.22)
add(7.80, thump, 0.16, 0.0)
shimmer = (np.sin(2 * np.pi * 1568 * tt) * 0.5
           + np.sin(2 * np.pi * 2349 * tt) * 0.28) * np.exp(-tt / 0.30)
add(7.82, shimmer, 0.030, 0.0)

# ---------------------------------------------------------------- 5. 结果和弦
n = int(2.0 * SR)
tt = np.arange(n) / SR
chord = (0.55 * np.sin(2 * np.pi * 220.0 * tt)
         + 0.35 * np.sin(2 * np.pi * 329.63 * tt + 0.4)
         + 0.22 * np.sin(2 * np.pi * 440.0 * tt + 0.9))
env = np.minimum(1.0, tt / 0.25) * np.exp(-tt / 0.75)
add(10.30, chord * env, 0.055, 0.0)

# ---------------------------------------------------------------- 6. 标题气浪
n = int(1.5 * SR)
x = bp(rng.normal(size=n), 300, 4000)
e = np.linspace(0, 1, n) ** 2.2 * np.exp(-np.linspace(0, 2.0, n))
add(11.40, x * e, 0.075, 0.0)
n = int(1.2 * SR)
tt = np.arange(n) / SR
land = np.sin(2 * np.pi * 55.0 * tt) * np.exp(-tt / 0.35) * np.minimum(1, tt / 0.02)
add(12.72, land, 0.11, 0.0)

# ---------------------------------------------------------------- 母带
mix = np.stack([L, R], axis=1)
mix = np.tanh(mix * 1.15) * 0.86
peak = np.max(np.abs(mix)) + 1e-9
mix = mix / peak * 0.72
fo = int(0.45 * SR)
mix[-fo:] *= np.linspace(1, 0, fo)[:, None]
wavfile.write("soundtrack.wav", SR, (mix * 32767).astype(np.int16))
print("soundtrack.wav", DUR, "s  peak", round(float(np.max(np.abs(mix))), 3))
