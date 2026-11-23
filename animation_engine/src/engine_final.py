"""
比对引擎 · 20 秒技术预告片
ManimGL 1.7.x（不是 Manim Community：导入、命令行、渲染器都不一样）

预览:   manimgl engine_teaser.py EngineTeaser -l -w
中等:   manimgl engine_teaser.py EngineTeaser -m -w
出片:   manimgl engine_teaser.py EngineTeaser --hd -w

全片只有一个主体：一台"比对引擎"。
它接收一个名字（密文），在看不见内容的情况下对十万条记录做比对，
最后只吐出一个结果。不解释原理，只让机器转起来。

时间轴（120 BPM，每拍 0.5 秒）
    0.00 – 3.00   开机      环阵浮现、HUD 上电、扫描光掠过
    3.00 – 6.00   注入      一个密文从画面外切进核心
    6.00 – 11.00  加速      十拍，转速与计数一起冲
    11.00 – 11.50 急停      全静止半拍（给冲击波留白）
    11.50 – 13.50 命中      白闪、冲击波、弹簧回弹、100,000
    13.50 – 16.00 收束      塌成一点，弹成结果框，打勾
    16.00 – 20.00 收尾      拉远，标题
"""

import numpy as np
from manimlib import *

# ---------------------------------------------------------------- 色板 / 字体
BG    = "#05070A"   # 近黑，偏冷蓝
CYAN  = "#9BE7FF"   # 冷青白，主体线条
ICE   = "#EAF6FF"   # 纸白，关键强调
AMBER = "#FFB020"   # 琥珀，只给"关键数值 / 结果"
STEEL = "#5D7382"   # 钢灰，辅助标注

MONO = "Consolas"           # HUD、数字
CJK  = "Microsoft YaHei"    # 中文字幕

# 结尾那句话。面向中国的外企：英文字面 + 中文一句，最稳。
# 想整片纯英文就换下面两行（字体必须跟着换成 MONO，否则中文缺字）。
TAGLINE = "算完了十万条，却什么也没看见"
TAGLINE_FONT = CJK
# TAGLINE = "One hundred thousand comparisons. Nothing revealed."
# TAGLINE_FONT = MONO

BEAT = 0.5  # 120 BPM


def spring(t, freq=15.0, damping=6.2):
    """阻尼回弹：冲过头一点再定住，机械感就来自这里。"""
    norm = 1 - np.exp(-damping) * np.cos(freq)
    return (1 - np.exp(-damping * t) * np.cos(freq * t)) / norm


class EngineTeaser(Scene):
    # ============================================================ 装配
    def setup(self):
        super().setup()
        self.rng = np.random.default_rng(1120)

        # 全局旋钮：转速 / 镜头高度 / 比较次数 / 呼吸相位 / 两条透明度总线
        self.speed = ValueTracker(1.0)
        self.zoom = ValueTracker(8.0)
        self.count = ValueTracker(0.0)
        self.phase = ValueTracker(0.0)
        self.streak_fade = ValueTracker(0.0)
        self.dust_fade = ValueTracker(0.0)
        self.progress = ValueTracker(0.0)   # 扫描进度 0→1，驱动进度环和记录点亮
        self.add(self.speed, self.zoom, self.count, self.phase,
                 self.streak_fade, self.dust_fade, self.progress)
        self.phase.add_updater(lambda m, dt: m.increment_value(dt))
        # 镜头高度统一由 zoom 驱动，"节拍脉冲"只是让 zoom 弹一下
        self.camera.frame.add_updater(lambda m: m.set_height(self.zoom.get_value()))

        self.build_rings()
        self.build_records()
        self.build_band()
        self.build_core()
        self.build_hud()
        self.build_readout()
        self.build_streaks()
        self.build_dust()
        # 这三件靠透明度总线控制显隐，必须先入场景它们的 updater 才会跑
        self.add(self.readout, self.streaks, self.dust)

    # ============================================================ 时间轴
    def construct(self):
        self.boot()        # 0.0 – 3.0
        self.inject()      # 3.0 – 6.0
        self.accelerate()  # 6.0 – 11.0
        self.tension()     # 11.0 – 11.5
        self.impact()      # 11.5 – 13.5
        self.collapse()    # 13.5 – 16.0
        self.outro()       # 16.0 – 20.0

    # ---------------------------------------------------------- 刻度环
    def tick_ring(self, radius, count, length, width, color, opacity,
                  major_every=0, major_length=0.0, major_width=0.0):
        """一圈细刻度 + 若干加长加粗的大刻度，读起来才像仪表盘而不是虚线。"""
        fine = VGroup()
        for i in range(count):
            angle = TAU * i / count
            d = np.array([np.cos(angle), np.sin(angle), 0.0])
            fine.add(Line(radius * d, (radius + length) * d,
                          stroke_width=width, stroke_color=color,
                          stroke_opacity=opacity))
        majors = VGroup()
        if major_every:
            for i in range(0, count, major_every):
                angle = TAU * i / count
                d = np.array([np.cos(angle), np.sin(angle), 0.0])
                majors.add(Line(radius * d, (radius + major_length) * d,
                                stroke_width=major_width, stroke_color=color,
                                stroke_opacity=min(1.0, opacity + 0.30)))
        guide = Circle(
            radius=radius,
            stroke_width=0.8,
            stroke_color=color,
            stroke_opacity=0.13,
            fill_opacity=0.0,
        )
        return VGroup(guide, fine, majors)

    def build_rings(self):
        # (半径, 细刻度数, 细长, 细宽, 颜色, 透明度, 大刻度间隔, 大长, 大宽, 角速度, 方向)
        specs = [
            (1.16, 60, 0.065, 1.0, CYAN, 0.45, 5, 0.20, 1.8, 26.0, +1),
            (1.70, 80, 0.050, 0.9, CYAN, 0.32, 10, 0.17, 1.6, 17.0, -1),
            (2.18, 44, 0.100, 1.1, CYAN, 0.45, 4, 0.26, 1.9, 11.0, +1),
        ]
        self.rings = []
        for radius, n, ln, sw, color, op, every, mlen, mw, omega, sign in specs:
            ring = self.tick_ring(radius, n, ln, sw, color, op, every, mlen, mw)
            ring.add_updater(
                lambda m, dt, w=omega, s=sign: m.rotate(
                    s * w * self.speed.get_value() * dt * DEGREES,
                    about_point=ORIGIN,
                )
            )
            self.rings.append(ring)
        self.ring_group = VGroup(*self.rings)

    # ---------------------------------------------------------- 记录环 + 进度环
    def build_records(self):
        """一圈 120 条"记录"：不转，只被逐条点亮；外圈 36 段光弧显示扫描进度。"""
        self.records_count = 120
        self.record_ticks = []
        records = VGroup()
        for i in range(self.records_count):
            # 从 12 点开始顺时针排，索引顺序 = 点亮顺序
            angle = PI / 2 - TAU * i / self.records_count
            d = np.array([np.cos(angle), np.sin(angle), 0.0])
            tick = Line(2.36 * d, 2.50 * d,
                        stroke_width=1.3, stroke_color=STEEL, stroke_opacity=0.20)
            records.add(tick)
            self.record_ticks.append(tick)
        self.records_ring = records

        # 16 段：每段更大，进度读起来是"格子"，不是一圈噪点
        self.seg_count = 16
        self.rim = VGroup()
        for i in range(self.seg_count):
            angle = PI / 2 - TAU * i / self.seg_count
            self.rim.add(Arc(
                radius=2.60, start_angle=angle - 0.05, angle=0.29,
                stroke_width=4.2, stroke_color=AMBER, stroke_opacity=0.0,
            ))

        # 中圈分段结构：随扫描依次激活、轻微错位
        self.mid_count = 36
        self.mid_segs = VGroup()
        for i in range(self.mid_count):
            a = PI / 2 - TAU * i / self.mid_count
            self.mid_segs.add(Arc(radius=1.94, start_angle=a - 0.10, angle=0.145,
                                  stroke_width=2.2, stroke_color=CYAN,
                                  stroke_opacity=0.0))
        # 细密运算纹理：一圈很短的刻线，随节拍闪
        self.tex = VGroup()
        for i in range(64):
            a = TAU * i / 64
            d = np.array([np.cos(a), np.sin(a), 0.0])
            self.tex.add(Line(2.42 * d, 2.45 * d, stroke_width=1.0,
                              stroke_color=CYAN, stroke_opacity=0.0))

        self.scan_head = GlowDot(center=UP * 2.60, radius=0.06,
                                 color=AMBER, glow_factor=3.4).set_opacity(0.0)
        self.scan_beam = Line(ORIGIN, UP * 2.72, stroke_width=1.0,
                              stroke_color=AMBER, stroke_opacity=0.0)
        self.scan_group = Group(self.rim, self.mid_segs, self.tex, self.scan_head, self.scan_beam)

        self._lit_segs = 0
        self._lit_ticks = 0
        self._lit_mid = 0
        self._lit_tex = 0
        # 扫描那一层自己带 updater，进场景才开始跑
        self.rim.add_updater(self.update_scan)

    def update_scan(self, m, dt):
        """进度到哪儿：点亮到哪儿。只增量更新，别每帧全量刷。"""
        p = float(self.progress.get_value())

        n = int(p * self.seg_count + 1e-9)
        for i in range(self._lit_segs, min(n, self.seg_count)):
            self.rim[i].set_stroke(color=AMBER, width=4.2, opacity=0.95)
        self._lit_segs = max(self._lit_segs, n)

        # 中圈分段：扫到就亮，亮度略高于外圈
        m = int(p * self.mid_count + 1e-9)
        for i in range(self._lit_mid, min(m, self.mid_count)):
            self.mid_segs[i].set_stroke(color=ICE, width=2.4, opacity=0.55)
        self._lit_mid = max(self._lit_mid, m)

        # 细密纹理：跟着扫描走一段
        t = int(p * 64 + 1e-9)
        for i in range(self._lit_tex, min(t, 64)):
            self.tex[i].set_stroke(color=CYAN, width=1.0, opacity=0.35)
        self._lit_tex = max(self._lit_tex, t)

        # 记录卡：被扫过的那一瞬间翻 90°（像翻牌），翻完点亮
        if not hasattr(self, "_flip_e"):
            self._flip_e = [0.0] * self.records_count
        for i in range(self.records_count):
            cross = 3.4 + 4.0 * i / self.records_count
            raw = (self.time - cross) / 0.22
            if raw <= 0.0:
                continue
            e = min(1.0, raw)
            e = e * e * (3 - 2 * e)
            delta = e - self._flip_e[i]
            if delta > 1e-4:
                chip = self.record_ticks[i]
                chip.rotate(delta * PI / 2, about_point=chip.get_center())
                self._flip_e[i] = e
            if e >= 0.999 and self._flip_e[i] > 0 and not getattr(self.record_ticks[i], "_lit", False):
                chip = self.record_ticks[i]
                chip.set_fill(color=ICE, opacity=0.85)
                chip.set_stroke(color=ICE, width=0.9, opacity=0.55)
                chip._lit = True
        self._lit_ticks = int(p * self.records_count + 1e-9)

        angle = PI / 2 - TAU * p
        d = np.array([np.cos(angle), np.sin(angle), 0.0])
        self.scan_head.move_to(2.60 * d)
        self.scan_beam.set_points_as_corners([ORIGIN, 2.74 * d])

    # ---------------------------------------------------------- 环绕数据带
    def build_band(self):
        def make(radius, count, size, seed, op):
            r_ = np.random.default_rng(seed)
            g = VGroup()
            tk = []
            for i in range(count):
                a = TAU * i / count
                txt = "%02X%02X" % (r_.integers(0, 256), r_.integers(0, 256))
                t = Text(txt, font=MONO, font_size=size, color=STEEL)
                t.set_fill(STEEL, op).set_stroke(width=0)
                t.rotate(a - PI / 2)
                t.move_to(radius * np.array([np.cos(a), np.sin(a), 0.0]))
                g.add(t)
                tk.append(t)
            return g, tk

        self.band_in, self.tok_in = make(3.34, 22, 26, 11, 0.46)
        self.band_out, self.tok_out = make(3.90, 26, 18, 23, 0.18)
        self.band = Group(self.band_in, self.band_out)

    def band_glow(self, m, dt):
        """光束附近亮，往两边连续衰减（稍微渐变）。"""
        p = float(self.progress.get_value())
        a = PI / 2 - TAU * p
        for toks, base_op, gain in ((self.tok_in, 0.46, 0.54),
                                    (self.tok_out, 0.18, 0.34)):
            for t in toks:
                # 底部读数区留空，别和 RECORDS 叠在一起
                if t.get_y() < -2.35 and abs(t.get_x()) < 2.15:
                    t.set_fill(STEEL, 0.0)
                    continue
                cur = np.arctan2(t.get_y(), t.get_x())
                d = abs(((cur - a + PI) % TAU) - PI)
                k = max(0.0, 1.0 - d / 0.95)
                k = k * k
                t.set_fill(ICE if k > 0.45 else CYAN if k > 0.08 else STEEL,
                           base_op + gain * k)

    def start_band(self):
        self.add(self.band)
        self.band_in.add_updater(lambda m, dt: m.rotate(9.0 * dt, about_point=ORIGIN))
        self.band_out.add_updater(lambda m, dt: m.rotate(-4.5 * dt, about_point=ORIGIN))
        self.band_in.add_updater(self.band_glow)

    def start_scan(self):
        """6.0 秒：进度环亮起，开始扫。进度由场景时钟驱动，不需要额外动画。"""
        self.scan_head.set_opacity(1.0)
        self.scan_beam.set_stroke(opacity=0.5)
        self.add(self.scan_group)
        self.progress.add_updater(
            lambda m: m.set_value(np.clip((self.time - 3.4) / 4.0, 0.0, 1.0))
        )
        # 计数器：跟着 16 段一格一格跳（机械进位），跳格闪琥珀
        self.num.clear_updaters()

        def carry(m):
            p = float(np.clip((self.time - 3.4) / 4.0, 0.0, 1.0))
            n = int(p * 16 + 1e-9)
            m.set_value(int(round(100000 * n / 16)))
            t_step = 3.4 + 4.0 * n / 16
            f = np.exp(-max(0.0, self.time - t_step) / 0.10) if n > 0 else 0.0
            m.set_fill(AMBER if f > 0.35 else ICE, 1.0)

        self.num.add_updater(carry)

    def finish_scan(self):
        """进度锁死在 1，进度环整圈闭合。"""
        self.num.clear_updaters()
        self.num.set_value(100000)
        self.num.set_fill(ICE, 1.0)
        self.progress.clear_updaters()
        self.progress.set_value(1.0)
        self.count.clear_updaters()
        self.count.set_value(100000.0)

    # ---------------------------------------------------------- 核心
    def build_core(self):
        self.hex_out = RegularPolygon(
            n=6, radius=0.36, start_angle=0.0,
            stroke_width=1.6, stroke_color=CYAN, stroke_opacity=0.55, fill_opacity=0.0,
        )
        self.hex_in = RegularPolygon(
            n=6, radius=0.21, start_angle=PI / 6,
            stroke_width=1.2, stroke_color=CYAN, stroke_opacity=0.35, fill_opacity=0.0,
        )
        cross = VGroup(
            Line([-0.52, 0, 0], [-0.40, 0, 0], stroke_width=1.0, stroke_color=CYAN, stroke_opacity=0.30),
            Line([0.40, 0, 0], [0.52, 0, 0], stroke_width=1.0, stroke_color=CYAN, stroke_opacity=0.30),
            Line([0, -0.52, 0], [0, -0.40, 0], stroke_width=1.0, stroke_color=CYAN, stroke_opacity=0.30),
            Line([0, 0.40, 0], [0, 0.52, 0], stroke_width=1.0, stroke_color=CYAN, stroke_opacity=0.30),
        )
        self.core_dot = GlowDot(center=ORIGIN, radius=0.06, color=AMBER, glow_factor=3.0)
        self.core = Group(self.hex_out, self.hex_in, cross, self.core_dot)

        self.hex_out.add_updater(
            lambda m, dt: m.rotate(0.42 * self.speed.get_value() * dt, about_point=ORIGIN)
        )
        self.hex_in.add_updater(
            lambda m, dt: m.rotate(-0.66 * self.speed.get_value() * dt, about_point=ORIGIN)
        )
        self.core_dot.add_updater(
            lambda m: m.set_radius(
                0.055 + 0.014 * (0.5 + 0.5 * np.sin(self.phase.get_value() * 5.2))
            )
        )

    # ---------------------------------------------------------- HUD 角标
    def hud_text(self, string, size, color=STEEL, opacity=1.0, weight=NORMAL):
        return Text(string, font=MONO, font_size=size, color=color,
                    weight=weight).set_opacity(opacity)

    def corner_bracket(self, sx, sy, arm=0.42, x=5.95, y=3.40):
        corner = np.array([sx * x, sy * y, 0.0])
        return VGroup(
            Line(corner, corner - np.array([sx * arm, 0, 0])),
            Line(corner, corner - np.array([0, sy * arm, 0])),
        ).set_stroke(width=1.3, color=CYAN, opacity=0.42)

    def build_hud(self):
        brackets = VGroup(*[
            self.corner_bracket(sx, sy)
            for sx, sy in [(-1, 1), (1, 1), (-1, -1), (1, -1)]
        ])

        tl = VGroup(
            self.hud_text("MATCH ENGINE", 34, ICE, 0.94, BOLD),
            self.hud_text("// PRIVATE QUERY", 21, STEEL, 0.9),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        tl.move_to([-5.78, 3.30, 0], aligned_edge=UL)

        tr = VGroup(
            self.hud_text("RECORDS", 21, STEEL, 0.9),
            self.hud_text("100,000", 33, ICE, 0.94, BOLD),
        ).arrange(DOWN, aligned_edge=RIGHT, buff=0.16)
        tr.move_to([5.78, 3.30, 0], aligned_edge=UR)

        bl = self.hud_text("INPUT  ·  CIPHERTEXT", 22, STEEL, 0.88)
        bl.move_to([-5.78, -3.18, 0], aligned_edge=DL)

        self.hud = VGroup(brackets, tl, tr, bl)

        # 状态行：IDLE → INJECT → SCAN → RESOLVE → MATCH
        self.status_pos = np.array([-5.78, 2.14, 0.0])
        self.status_mob = None

    def set_status(self, text, color=ICE, opacity=0.62):
        """HUD 状态行直接硬切，不做过场——仪表的味道就在这个"硬"上。"""
        new = Text("STATUS  " + text, font=MONO, font_size=21, color=color)
        new.set_opacity(opacity)
        new.move_to(self.status_pos, aligned_edge=UL)
        if self.status_mob is not None:
            self.remove(self.status_mob)
        self.status_mob = new
        self.add(new)

    # ---------------------------------------------------------- 计数器
    def build_readout(self):
        self.num = DecimalNumber(
            0, num_decimal_places=0, group_with_commas=True,
            font_size=52, color=ICE, edge_to_fix=ORIGIN,
            text_config={"font": MONO, "weight": BOLD},
        )
        self.num.move_to([0, -3.05, 0])
        self.num.add_updater(lambda m: m.set_value(int(round(self.count.get_value()))))

        self.num_label = self.hud_text("RECORDS", 20, STEEL, 0.85)
        self.num_label.move_to([0, -2.45, 0])

        self.readout = VGroup(self.num_label, self.num)

    # ---------------------------------------------------------- 速度线
    def streak_updater(self, m, dt):
        m.dist += m.v * self.speed.get_value() * dt
        if m.dist > 7.4:
            m.dist = 1.15 + float(self.rng.uniform(0, 0.5))
        d = np.array([np.cos(m.ang), np.sin(m.ang), 0.0])
        m.set_points_as_corners([d * m.dist, d * (m.dist + m.ln)])
        m.set_stroke(opacity=m.base_op * self.streak_fade.get_value())

    def build_streaks(self):
        self.streaks = VGroup()
        for _ in range(56):
            ln = float(self.rng.uniform(0.22, 0.75))
            line = Line(
                ORIGIN, RIGHT * ln,
                stroke_width=float(self.rng.choice([1.0, 1.0, 1.6])),
                stroke_color=CYAN,
                stroke_opacity=0.0,
            )
            line.ang = float(self.rng.uniform(0, TAU))
            line.dist = float(self.rng.uniform(1.3, 3.4))
            line.ln = ln
            line.v = float(self.rng.uniform(1.4, 3.2))
            line.base_op = float(self.rng.uniform(0.07, 0.30))
            line.add_updater(self.streak_updater)
            self.streaks.add(line)

    # ---------------------------------------------------------- 空气里的碎片
    def dust_updater(self, m, dt):
        m.shift(RIGHT * m.v * self.speed.get_value() * dt)
        if m.get_center()[0] > 6.4:
            m.shift(LEFT * 12.8)
        m.set_opacity(m.base_op * self.dust_fade.get_value())

    def build_dust(self):
        tokens = ["0x3F", "A7", "9E", "CKKS", "0x1B", "FF", "4096", "0x7D", "E2", "B8"]
        self.dust = VGroup()
        for _ in range(22):
            t = Text(
                str(self.rng.choice(tokens)),
                font=MONO,
                font_size=int(self.rng.integers(11, 15)),
                color=CYAN,
                opacity=0.0,
            )
            while True:
                x = float(self.rng.uniform(-6.0, 6.0))
                y = float(self.rng.uniform(-3.0, 3.0))
                if x * x + y * y > 2.95 ** 2:
                    break
            t.move_to([x, y, 0])
            t.v = float(self.rng.uniform(0.18, 0.55))
            t.base_op = float(self.rng.uniform(0.12, 0.26))
            t.add_updater(self.dust_updater)
            self.dust.add(t)

    # ============================================================ 工具
    def cue(self, t):
        """把时间轴对齐到绝对秒 t：脚本里所有时间点都以这张表为准。"""
        remaining = t - self.time
        if remaining > 1e-4:
            self.wait(remaining)

    def wave(self, radius0=0.3, growth=3.4, color=ICE,
             width=2.4, opacity=0.8, fade_power=1.7, run_time=0.8):
        """冲击波：一个圆一边胀开、一边变细变淡。"""
        circle = Circle(radius=radius0, stroke_width=width, stroke_color=color,
                        stroke_opacity=opacity, fill_opacity=0.0)

        def update(m, alpha):
            m.become(Circle(
                radius=radius0 + growth * alpha,
                stroke_width=width * (1 - alpha) + 0.3,
                stroke_color=color,
                stroke_opacity=opacity * (1 - alpha) ** fade_power,
                fill_opacity=0.0,
            ))

        return UpdateFromAlphaFunc(circle, update, run_time=run_time)

    # ============================================================ 0.0 – 2.0 开机
    def boot(self):
        self.cue(0.0)
        self.play(AnimationGroup(
            *[ShowCreation(r, run_time=0.55) for r in self.rings],
            lag_ratio=0.06,
        ))
        self.add(*self.rings, self.ring_group)

        self.play(
            AnimationGroup(*[ShowCreation(b, run_time=0.28) for b in self.hud[0]], lag_ratio=0.08),
            FadeIn(self.hud[1:], run_time=0.40),
            FadeIn(self.core, run_time=0.40),
        )
        self.set_status("READY", STEEL, 0.7)
        self.play(AnimationGroup(
            *[ShowCreation(t, run_time=0.14) for t in self.record_ticks], lag_ratio=0.002))
        self.cue(1.55)
        self.play(ShowPassingFlash(
            Circle(radius=2.26, stroke_width=2.4, stroke_color=AMBER,
                   stroke_opacity=0.95, fill_opacity=0.0),
            time_width=0.35, run_time=0.45))
        self.cue(2.0)

    # ============================================================ 2.0 – 3.4 注入
    def inject(self):
        self.cue(2.0)
        self.set_status("INJECT", ICE, 0.9)
        path = CubicBezier(
            np.array([-8.0, -1.9, 0.0]), np.array([-4.6, -1.0, 0.0]),
            np.array([-1.8, 1.2, 0.0]), np.array([0.0, 0.0, 0.0]),
        )
        particle = GlowDot(center=path.get_start(), radius=0.085, color=ICE, glow_factor=3.2)
        trail = TracedPath(particle.get_center, time_traced=0.28,
                           stroke_color=CYAN, stroke_width=2.6, stroke_opacity=1.0)
        self.add(trail, particle)
        self.play(MoveAlongPath(particle, path, run_time=0.75, rate_func=rush_into))

        self.play(
            Flash(ORIGIN, color=AMBER, num_lines=16, flash_radius=0.38,
                  line_length=0.34, run_time=0.40),
            self.wave(radius0=0.16, growth=1.1, color=AMBER, width=2.4, opacity=0.9, run_time=0.40),
            self.speed.animate(run_time=0.40, rate_func=rush_from).set_value(2.4),
            FadeOut(particle, run_time=0.16),
        )
        self.remove(trail)

        # 轨道粒子 + 速度线快速进场
        orb_a = GlowDot(center=RIGHT, radius=0.035, color=CYAN, glow_factor=2.6)
        orb_b = GlowDot(center=UP, radius=0.028, color=ICE, glow_factor=2.6)
        ang_a, ang_b = ValueTracker(0.0), ValueTracker(PI)
        orb_a.add_updater(lambda m: m.move_to(1.02 * np.array(
            [np.cos(ang_a.get_value()), np.sin(ang_a.get_value()), 0.0])))
        orb_b.add_updater(lambda m: m.move_to(0.74 * np.array(
            [np.cos(-ang_b.get_value()), np.sin(-ang_b.get_value()), 0.0])))
        ang_a.add_updater(lambda m, dt: m.increment_value(2.2 * self.speed.get_value() * dt))
        ang_b.add_updater(lambda m, dt: m.increment_value(3.4 * self.speed.get_value() * dt))
        self.orbiters = Group(orb_a, orb_b)
        self.add(ang_a, ang_b, self.orbiters)
        self.play(self.streak_fade.animate(run_time=0.25).set_value(1.0),
                  self.dust_fade.animate(run_time=0.25).set_value(1.0))
        self.cue(3.4)

    # ============================================================ 3.4 – 7.4 扫描
    def accelerate(self):
        self.cue(3.4)
        self.set_status("SCAN", ICE, 0.95)
        self.start_band()
        self.play(FadeIn(self.band, run_time=0.45))
        self.start_scan()
        # 投影更亮：扫描手加粗提亮
        self.scan_beam.set_stroke(width=2.2, opacity=0.9)

        for i in range(1, 9):
            height = 8.0 + (7.0 - 8.0) * (i / 8.0)
            animations = [
                # 幅度加倍：1.028 → 1.056
                self.zoom.animate(run_time=0.18, rate_func=rush_from).set_value(height * 1.056),
                self.speed.animate(run_time=0.30, rate_func=linear).set_value(2.6 + 4.4 * i / 8.0),
            ]
            # 每拍一道白脉冲
            ang = float(self.rng.uniform(0, TAU))
            d = np.array([np.cos(ang), np.sin(ang), 0.0])
            animations.append(ShowPassingFlash(
                Line(-0.95 * d, 0.95 * d, stroke_width=2.6, stroke_color=ICE),
                time_width=1.0, run_time=0.22))
            # 只在第 2、6 拍做一次"错位对位"，幅度 0.12（是原来的两倍）
            if i in (2, 6):
                j = int(self.rng.integers(0, self.mid_count))
                animations.append(
                    self.mid_segs[j].animate(run_time=0.30, rate_func=there_and_back)
                    .rotate(0.12, about_point=ORIGIN))
            self.play(*animations)
            self.play(self.zoom.animate(run_time=0.10, rate_func=smooth).set_value(height))
            self.cue(3.4 + i * BEAT)

    # ============================================================ 7.4 – 7.8 急停
    def tension(self):
        self.cue(7.4)
        self.finish_scan()
        self.play(self.speed.animate(run_time=0.12, rate_func=rush_into).set_value(0.0))
        self.play(self.zoom.animate(run_time=0.26, rate_func=linear).set_value(7.0))
        self.cue(7.8)

    # ============================================================ 7.8 – 9.2 命中
    def impact(self):
        self.cue(7.8)
        self.set_status("RESOLVE", AMBER, 0.9)
        self.rim.clear_updaters()

        flash = Rectangle(width=36, height=22, fill_color="#FFFFFF",
                          fill_opacity=0.0, stroke_width=0.0)
        self.add(flash)
        self.play(flash.animate(run_time=0.04).set_fill(opacity=0.9))

        self.assembly = Group(*self.rings, self.records_ring, self.core)
        self.assembly.scale(0.90, about_point=ORIGIN)   # 回弹幅度也加倍

        self.play(
            flash.animate(run_time=0.18).set_fill(opacity=0.0),
            self.assembly.animate(run_time=0.62, rate_func=spring).scale(1 / 0.90, about_point=ORIGIN),
            self.wave(radius0=0.35, growth=6.4, color=ICE, width=2.8, opacity=0.9, run_time=0.62),
            self.streak_fade.animate(run_time=0.18).set_value(0.0),
            FadeOut(self.scan_group, run_time=0.20),
            FadeOut(self.band, run_time=0.30),
        )
        self.remove(flash, self.streaks, self.rim, self.mid_segs, self.tex, self.band)
        self.cue(8.55)

        self.play(
            self.num.animate(run_time=0.30, rate_func=spring).scale(1.34),
            self.num_label.animate(run_time=0.30).set_opacity(0.95),
            self.wave(radius0=0.55, growth=4.6, color=CYAN, width=1.6, opacity=0.55, run_time=0.30),
        )
        self.num.clear_updaters()

        # 光向中心收（幅度更大：2.62 → 0.10）
        gather = Circle(radius=2.62, stroke_width=3.0, stroke_color=AMBER, stroke_opacity=1.0)
        self.add(gather)
        self.play(
            gather.animate(run_time=0.35, rate_func=rush_into).scale(0.06).set_stroke(opacity=0.0),
            *[r.animate(run_time=0.35).set_stroke(color=ICE, opacity=0.10) for r in self.rings],
            self.records_ring.animate(run_time=0.35).set_stroke(opacity=0.06),
            self.hex_out.animate(run_time=0.35).set_stroke(color=AMBER, opacity=0.95),
            self.hex_in.animate(run_time=0.35).set_stroke(color=AMBER, opacity=0.7),
            self.core_dot.animate(run_time=0.35).scale(2.4),
            self.speed.animate(run_time=0.35, rate_func=smooth).set_value(0.2),
            self.zoom.animate(run_time=0.35, rate_func=smooth).set_value(6.9),
        )
        for dx, dy in [(0.05, 0.03), (-0.045, -0.02)]:
            self.play(self.camera.frame.animate.shift(np.array([dx, dy, 0.0])), run_time=0.04)
        self.play(self.camera.frame.animate.move_to(ORIGIN), run_time=0.06)
        self.cue(9.2)

    # ============================================================ 9.2 – 10.4 收束
    def collapse(self):
        self.cue(9.2)
        self.remove(self.orbiters)
        outer, mid, inner = self.rings[2], self.rings[1], self.rings[0]

        outer.clear_updaters()
        self.play(
            outer.animate(run_time=0.24, rate_func=smooth).scale(0.68, about_point=ORIGIN)
            .set_stroke(opacity=0.0),
            self.records_ring.animate(run_time=0.24, rate_func=smooth)
            .scale(0.68, about_point=ORIGIN).set_stroke(opacity=0.0),
            self.dust_fade.animate(run_time=0.24).set_value(0.0),
        )
        mid.clear_updaters()
        self.play(mid.animate(run_time=0.12, rate_func=smooth).rotate(-0.40, about_point=ORIGIN))
        self.play(mid.animate(run_time=0.18, rate_func=rush_into)
                  .scale(0.34, about_point=ORIGIN).set_stroke(opacity=0.0))
        self.play(
            inner.animate(run_time=0.24, rate_func=smooth)
            .scale(0.12, about_point=ORIGIN).set_stroke(opacity=0.0),
            self.core.animate(run_time=0.24, rate_func=smooth).scale(0.30, about_point=ORIGIN),
        )
        self.remove(self.dust, *self.rings, self.ring_group,
                    self.records_ring, self.core, self.assembly)

        point = GlowDot(center=ORIGIN, radius=0.10, color=AMBER, glow_factor=3.6)
        self.play(FadeIn(point, run_time=0.10),
                  Flash(ORIGIN, color=AMBER, num_lines=12, flash_radius=0.24,
                        line_length=0.26, run_time=0.14))
        self.cue(10.35)

        self.box = RoundedRectangle(width=1.05, height=1.05, corner_radius=0.17,
                                    stroke_width=2.6, stroke_color=AMBER,
                                    fill_color=AMBER, fill_opacity=0.04)
        self.box.scale(0.06, about_point=ORIGIN)
        self.play(FadeOut(point, run_time=0.08),
                  self.box.animate(run_time=0.34, rate_func=spring).scale(1 / 0.06, about_point=ORIGIN))
        self.check = VGroup(
            Line([-0.30, 0.02, 0], [-0.06, -0.23, 0], stroke_width=3.6, stroke_color=ICE),
            Line([-0.06, -0.23, 0], [0.31, 0.21, 0], stroke_width=3.6, stroke_color=ICE),
        )
        self.play(ShowCreation(self.check[0], run_time=0.10, rate_func=rush_from))
        self.play(ShowCreation(self.check[1], run_time=0.12, rate_func=rush_from))
        self.set_status("MATCH", AMBER, 1.0)
        self.cue(11.05)

        self.label = Text("MATCH", font=MONO, font_size=23, color=AMBER, weight=BOLD)
        self.label.move_to([0, -0.84, 0])
        self.result = Text("1 / 100,000", font=MONO, font_size=44, color=AMBER, weight=BOLD)
        self.result.move_to(self.num.get_center())
        self.result_label = Text("MATCHED", font=MONO, font_size=19, color=AMBER)
        self.result_label.move_to(self.num_label.get_center())
        self.play(FadeIn(self.label, shift=UP * 0.06, run_time=0.14),
                  FadeOut(self.num, run_time=0.12), FadeOut(self.num_label, run_time=0.12))
        self.play(FadeIn(self.result, run_time=0.14), FadeIn(self.result_label, run_time=0.14))
        self.readout.add(self.result, self.result_label)
        self.cue(11.35)

    # ============================================================ 11.35 – 12.0 收尾
    def outro(self):
        self.cue(11.35)
        final = VGroup(self.box, self.check, self.label)
        self.play(
            self.zoom.animate(run_time=0.28, rate_func=rush_from).set_value(8.4),
            final.animate(run_time=0.28, rate_func=rush_from).scale(0.80).move_to(UP * 1.45),
            FadeOut(self.hud, run_time=0.22),
            FadeOut(self.readout, run_time=0.22),
            FadeOut(self.status_mob, run_time=0.22),
        )
        # 标题：由小到大撑开，然后定住（不再推近）
        title = Text("P R I V A T E   F U Z Z Y   M A T C H", font=MONO,
                     font_size=54, color=ICE, weight=BOLD).move_to([0, -1.45, 0])
        title.scale(0.78, about_point=[0, -1.45, 0])
        rule = Line(LEFT * 1.32, RIGHT * 1.32, stroke_width=2.2,
                    color=AMBER)
        rule.move_to([0, -2.02, 0]).scale(0.78)

        self.play(FadeIn(title, run_time=0.45),
                  title.animate(run_time=1.30, rate_func=rush_from).scale(1 / 0.78),
                  rule.animate(run_time=1.30, rate_func=rush_from).scale(1 / 0.78))
        self.cue(12.75)
        self.play(rule.animate(run_time=0.35).set_stroke(opacity=0.55))
        # 定住：给足时间读完（约 2.5 秒）
        self.cue(15.30)
        self.play(FadeOut(VGroup(final, title, rule), run_time=0.35))
        self.cue(15.65)
