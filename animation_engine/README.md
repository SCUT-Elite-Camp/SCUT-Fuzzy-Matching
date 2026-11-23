# MATCH ENGINE · 15 秒片

私有模糊匹配（SCUT-Fuzzy-Matching）的宣传短片。一台正在工作的比对引擎：数据注入 → 扫描 → 急停 → 命中 → 收束。

---

## 1. 成片

| | |
|---|---|
| 文件 | `MATCH_ENGINE_15s_1080p60.mp4` |
| 时长 | **15.65 秒**（939 帧） |
| 画面 | 1920 × 1080，60 fps，H.264 |
| 声音 | AAC 192 kbps，立体声 48 kHz |
| 电平 | 平均 −16.0 dB，峰值 −1.7 dB |

---

## 2. 时间轴

| 时间 | 段落 | 画面 | 声音 |
|---|---|---|---|
| 0.0–2.0 | 开机 | 四层环阵逐层画出，记录卡装入，HUD 亮起 | 低频床进入，启动气声（196/294/392Hz 上行） |
| 2.0–3.4 | 注入 | 粒子沿贝塞尔弧线切入核心 | 气声收尾 |
| 3.4–7.4 | 扫描 | 16 段进度弧逐段点亮；记录卡被扫过时**翻 90°**；外圈**数据带**旋转、光束附近提亮；计数按 16 格**机械进位** | 16 记机械 tick（0.062–0.092）+ 每 4 格一记棘轮闷响 |
| 7.4–7.8 | 急停 | 转速归零，画面静止半拍 | 音乐留白 |
| 7.8–9.2 | 命中 | 白闪 → 机械回弹（0.90→1.0）→ 光向中心收拢 → 核心转琥珀 | 52→80Hz 掉音闷响 + 金属微光 |
| 9.2–10.4 | 收束 | 外圈先停再收 → 中圈反向拧 → 内圈最后熄灭 | — |
| 10.4–11.4 | 结果 | 方框 + 勾，`1 / 100,000` | A 小调和弦 |
| 11.4–15.65 | 标题 | `PRIVATE FUZZY MATCH` 由 0.78 倍**撑开到 1.0 倍**后停住 2.55 秒 | 标题气浪 + 55Hz 落定 |

---

## 3. 视觉规范

**配色**

| 用途 | 色值 |
|---|---|
| 背景 | `#03060B` |
| 结构（环、刻度） | `#8DEBFF` |
| 高光 / 计算命中 | `#EAFBFF` |
| 进度 / 结果 | `#FFB020` |
| 辅助文字 | `#526B7B` |

**三层运动**：机械层（环阵，不同转速与方向）／数据层（刻度、记录卡、数据带）／能量层（光束、脉冲、轨迹）。三者速度不同，片子才不会“整盘一起转”。

**文字大小（1080p 下）**：MATCH ENGINE 34｜100,000 33｜数据带 token 26/18｜底部计数 52｜结果 44｜结尾标题 54。

---

## 4. 声音设计

- 底乐：A 小调，四段和声（Am → F → C → Em），每 1.8 秒一记 41–82Hz 次低频，结尾收在单音 A。
- 音效：开机气声、16 记机械 tick、棘轮、命中闷响、结果和弦、标题气浪。
- **全轨不含任何噪声层**（早前版本用过噪声做“空气感”，听起来是沙沙声，已全部换成正弦调制）。

---

## 5. 文件说明

```
MATCH_ENGINE_15s_1080p60.mp4   成片（可直接投放）
audio/
  final_mix.wav                最终混音（音效 + 底乐），与成片一一对应
  sfx.wav                      纯音效轨
  music.wav                    纯底乐轨
src/
  engine_final.py              ManimGL 场景源码（唯一源文件）
  make_track.py                音效生成脚本 → sfx.wav
  make_music.py                底乐生成脚本 → music.wav
  custom_config.yml            渲染配置（1080p60，深色底）
stills/
  01_scan.png                  扫描中的一帧
  02_100000.png                计数到 100,000
  03_end.png                   结尾定版
```

---

## 6. 重新渲染

环境：**ManimGL 1.7.2**（3b1b 版，不是 Manim Community），Windows + Anaconda。

```bash
# 渲染画面（约 4 分钟，无音轨）
manimgl src/engine_final.py EngineTeaser --hd -w

# 生成两条音轨
python src/make_track.py     # -> sfx.wav
python src/make_music.py     # -> music.wav

# 混合（音效 0.95 / 底乐 0.80，峰值归一化到 0.82）
python - <<'PY'
import numpy as np
from scipy.io import wavfile
sr, a = wavfile.read("sfx.wav");   _, b = wavfile.read("music.wav")
n = min(len(a), len(b))
m = a[:n].astype(np.float32)/32768.0*0.95 + b[:n].astype(np.float32)/32768.0*0.80
m = m/np.max(np.abs(m))*0.82
wavfile.write("final_mix.wav", sr, (m*32767).astype(np.int16))
PY

# 合成（注意：必须重编码视频，用 -c:v copy 在本机会卡死）
ffmpeg -i videos/EngineTeaser.mp4 -i final_mix.wav \
       -c:v libx264 -crf 18 -pix_fmt yuv420p -r 60 -c:a aac -b:a 192k \
       -shortest MATCH_ENGINE_15s_1080p60.mp4
```

---

## 7. 已知限制

1. **声音是合成的**：底乐与音效都由脚本生成（正弦和声 + 调制），不是真实乐器或采样库。如果想要更“厚”的质感，建议换一条授权音乐，替换 `audio/music.wav` 即可，节奏点已对齐 120 BPM。
2. **结尾标题 54 号已占画面宽 82%**，再放大需要改成两行。
3. **数据带底部有 ±15° 的缺口**，是为了给 `RECORDS 50,000` 让位，不是漏画。
4. **技术表述**：片中只有 `1 / 100,000` 与 `RECORDS` 两个数字，未声称“十万次密文比较”；`CKKS` 参数没有出现在画面里。
