#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
novel_classify.py —— 本地批量小说风格归类（纯标准库，零依赖，不联网）

原理：只从每个文件抽样 500 KB 左右（开头 + 三处散布位置），用它估计整本书的
特征分布，再打分归类。几十 MB 的全篇不需要读完，更不需要上传给任何 LLM。

两条判定路径：
  * 零样本：内置题材词表的加权命中密度，并在整批语料上做 z 分数自校准
            （抵消「都市词在所有小说里都常见」这类base-rate偏差）。开箱即用。
  * 有标注：每类 15~30 本 → 字符 bigram TF-IDF 质心余弦（Rocchio）。更准，
            且自动贴合你自己的分类体系（子类型、你的叫法）。

用法：
    # 1) 零样本直接归类
    python3 novel_classify.py /volume1/novels --out 归类.csv --jobs 4

    # 2) 有标注样本时先训模型，再用模型归类
    python3 novel_classify.py --train ./seeds --model-out model.json
    python3 novel_classify.py /volume1/novels --model model.json --out 归类.csv

    # 3) 先小规模试跑，看低置信度比例，再上全量
    python3 novel_classify.py /volume1/novels --limit 300 --out 试跑.csv

    # 4) 直接落盘：把每本移进 目标目录/<类别>/（没有就建）
    python3 novel_classify.py /volume1/novels --move-to /volume1/分类 --dry-run  # 先看计划
    python3 novel_classify.py /volume1/novels --move-to /volume1/分类            # 真的搬
    python3 novel_classify.py --undo /volume1/分类/.move-manifest.csv            # 后悔了搬回来

输出 CSV：路径、大小、编码、抽样汉字数、类别、置信度、次选、低置信标记、
判定依据（命中的特征词）、各题材原始得分（可选近似重复指纹）。
低置信度的行再拿去人工/LLM 复核——每本只需 2 KB 摘要，不是全篇。
"""

import argparse
import csv
import errno
import json
import math
import os
import re
import shutil
import sys
import time
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

# ---------------------------------------------------------------- 题材词表
# 权重：3 = 高度专指，2 = 较专指，1 = 常见但偏弱。这是软先验，不是硬规则。
LEXICON = {
    "武侠": """
        江湖 内力:3 轻功:3 掌门:3 门派:2 剑法:2 刀法:2 侠客:3 镖局:3 武林:3
        帮主:3 暗器:3 真气:3 经脉:2 点穴:3 掌风:2 内功:2 少林:2 峨眉:2 武当:2
        丐帮:2 走镖:2 结拜:2 剑客:3 剑鞘:3 招数:2 比武:2 江湖人:3 快意恩仇:3
        掌力:3 剑光:2 刀锋:2 山门:2 香主:3 舵主:2 客栈:1 弟子:2 血债:1 义气:2
    """,
    "仙侠修真": """
        修真:3 元婴:3 金丹:3 筑基:3 灵石:3 飞升:3 渡劫:3 仙尊:3 神识:3 灵根:3
        洞府:3 天劫:3 化神:3 大乘:3 符箓:3 灵器:3 仙门:3 剑修:3 丹炉:3 炼化:3
        法力:3 灵气:3 丹药:2 阵法:2 功法:2 仙剑:2 妖兽:2 丹田:2 顿悟:2 神魂:2
        灵力:3 真元:3 闭关:2 淬体:2 雷劫:3 道基:3 长老:1 结界:1
    """,
    "玄幻奇幻": """
        魔法:3 斗气:3 魔兽:3 精灵:3 巨龙:3 法师:3 咒语:3 圣光:3 异界:3 兽人:3
        矮人:3 亡灵:3 神殿:3 巫妖:3 恶魔:3 魔晶:3 魔法阵:3 幻兽:3 精灵族:3
        骑士:2 佣兵:2 领主:2 城堡:2 血脉:2 神明:2 诅咒:2 牧师:2 符文:2
        秘境:2 天赋:1 觉醒:1 大陆:1 帝国:1
    """,
    "都市": """
        公司:2 总裁:3 董事长:3 老板:2 上班:2 手机:2 打车:2 地铁:2 相亲:3
        娱乐圈:3 写字楼:3 合同:2 客户:2 工资:2 房子:2 股票:2 面试:2 加班:2
        微信:2 直播:2 明星:2 粉丝:2 别墅:2 集团:2 股份:2 股东:2 创业:2 升职:2
        前台:2 出租车:2 律师:1 医院:1 咖啡:1 酒吧:1 城市:1 同学:1 项目:1
        会议:1 上市:2 商圈:2
    """,
    "科幻": """
        飞船:3 战舰:3 星球:3 宇宙:3 量子:3 人工智能:3 机器人:3 空间站:3 星际:3
        虫族:3 机甲:3 曲率:3 殖民:3 舰队:3 星系:3 陨石:3 宇航员:3 全息:3
        纳米:3 基因:2 实验室:2 辐射:2 数据流:2 能源核:3
    """,
    "历史": """
        皇帝:3 太子:3 朝廷:3 陛下:3 粮草:3 圣旨:3 知府:3 科举:3 边关:3
        藩王:3 朝堂:3 太监:3 宰相:3 御史:3 封地:3 六部:3 钦差:3 上朝:3
        龙椅:3 大臣:2 将军:2 王朝:2 后宫:2 战马:2 御笔:2 年号:2 兵马:1
    """,
    "言情": """
        男友:3 女友:3 表白:3 婚纱:3 分手:3 情书:3 约会:3 结婚证:3 老公:3
        老婆:3 婚约:3 初恋:3 暗恋:3 恋爱:2 心动:2 暧昧:2 撒娇:2 结婚:2
        单身:2 亲了一下:2 心跳:1 温柔:1 眼泪:1 拥抱:1
    """,
    "游戏竞技": """
        副本:3 玩家:3 公会:3 刷怪:3 掉率:3 任务奖励:3 组队:3 属性面板:3
        经验值:3 开黑:3 电竞:3 战队:3 上分:3 排位:3 血条:2 装备:2 服务器:2
        技能:2 等级:1
    """,
    "末世": """
        丧尸:3 末日:3 幸存者:3 避难所:3 尸群:3 变异体:3 安全区:3 辐射尘:3
        病毒:2 感染:2 物资:2 废墟:2 幸存:2 断壁残垣:2 缺粮:2 基地:1 围墙:1
    """,
    "灵异恐怖": """
        棺材:3 阴气:3 凶宅:3 怨灵:3 符咒:3 僵尸:3 冤魂:3 阳气:3 招魂:3
        鬼村:3 煞气:3 纸钱:3 灵堂:3 邪祟:3 道士:2 烧香:2 血光:2 结界:1
    """,
}

# 文件名/目录名里出现这些字样时给一点先验加成（很多书堆命名就带品类）
FILENAME_HINTS = {
    "武侠": ["武侠", "江湖", "刀剑", "剑客"],
    "仙侠修真": ["修仙", "修真", "仙侠", "仙尊", "剑仙", "洪荒"],
    "玄幻奇幻": ["玄幻", "奇幻", "异界", "魔法", "斗气", "西幻"],
    "都市": ["都市", "职场", "商战", "娱乐", "校园", "兵王", "赘婿", "神医"],
    "科幻": ["科幻", "星际", "机甲", "赛博", "未来"],
    "历史": ["历史", "穿越", "三国", "大明", "大唐", "宋朝", "皇朝"],
    "言情": ["言情", "甜宠", "总裁", "虐恋", "婚恋"],
    "游戏竞技": ["网游", "游戏", "电竞", "竞技"],
    "末世": ["末世", "丧尸", "废土", "末日"],
    "灵异恐怖": ["灵异", "恐怖", "悬疑", "盗墓", "诡"],
}

HINT_BOOST = 0.25      # 文件名命中时的相对加成
MIN_CHARS = 300        # 少于这个汉字数认为不是正文
MIN_HITS = 12.0        # 命中权重合计低于此值认为证据不足（不硬判）
MIN_TERMS = 2          # 只命中一个词种太薄，不足以定题材
MIN_CALIB_N = 30       # 语料少于这么多本时不做 z 分数自校准
DEFAULT_GENRE = "未分类"
NEG = re.compile(r"[^\u3400-\u4dbf\u4e00-\u9fff]")


def parse_lexicon(spec):
    out = {}
    for token in spec.split():
        term, _, w = token.partition(":")
        if term.strip():
            out[term.strip()] = float(w) if w else 1.0
    return out


LEX = {g: parse_lexicon(s) for g, s in LEXICON.items()}

# ---------------------------------------------------------------- 抽样与解码


def read_sample(path, budget):
    """读开头 + 三处散布位置，各取 1/4 预算，返回 (字节段列表, 文件大小)。

    只抽样是全流程能扛住上万文件的关键。分成多段而不是拼成一段，是为了
    按段打分再取中位数：超长的前言、目录、盗版广告、作者的话通常只污染
    其中一两段，中位数能把它们顶掉（只在开头取样会被前言直接带偏）。
    """
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if size <= budget:
            return [f.read()], size
        chunk = max(8 * 1024, budget // 4)
        segs = [f.read(chunk)]
        for frac in (0.30, 0.60, 0.85):
            f.seek(int(size * frac))
            segs.append(f.read(chunk))
    return segs, size


def _median(vals):
    s = sorted(vals)
    n = len(s)
    if not n:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _strict_decode(data, enc):
    """严格解码；末尾被截断的多字节字符允许回退 1~3 个字节。

    这一步很关键：按字节切样本时约有 2/3 的概率切在多字节汉字中间，
    不处理就会让「本可 UTF-8 解码」的样本失败、被误判成 GB18030。
    """
    for cut in range(4):
        chunk = data[:len(data) - cut] if cut else data
        if not chunk:
            break
        try:
            return chunk.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def _cjk_ratio(text):
    if not text:
        return 0.0
    n = min(len(text), 20000)
    sample = text[:n]
    return sum(1 for ch in sample if "\u3400" <= ch <= "\u9fff") / n


def sniff_encoding(raw, head_len=32 * 1024):
    """中文 txt 绝大多数是 UTF-8 或 GBK/GB18030；按严格解码能力判断。"""
    head = raw[:head_len]
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
        return "utf-16"
    if not head:
        return "utf-8"
    for enc in ("utf-8", "gb18030", "big5"):
        txt = _strict_decode(head, enc)
        if txt is None:
            continue
        # UTF-8 严格通过即采信；GB18030 会「成功」解码任意字节，
        # 因此要求解出来确实是中文（CJK 占比够高）
        if enc == "utf-8" or _cjk_ratio(txt) > 0.15:
            return enc
    return "gb18030"


def decode_segments(segs, head_len=32 * 1024):
    """用第一段（起点无跨界截断）定编码，再逐段解码并校验乱码率。

    乱码校验是第二道闸：万一编码判断错了，解码结果会出现成片 U+FFFD，
    此时换另一种编码重试，取乱码更少的那种。
    """
    enc = sniff_encoding(segs[0][:head_len]) if segs else "utf-8"

    def dec(e):
        return [s.decode(e, errors="replace") for s in segs]

    try:
        texts = dec(enc)
    except LookupError:
        enc = "gb18030"
        texts = dec(enc)
    bad = sum(t.count("\ufffd") for t in texts)
    n = max(1, sum(len(t) for t in texts))
    if bad > max(8, n * 0.01):
        for alt in ("gb18030", "utf-8-sig", "utf-8"):
            if alt == enc:
                continue
            try:
                t2 = dec(alt)
            except LookupError:
                continue
            if sum(t.count("\ufffd") for t in t2) < bad:
                return t2, alt
    return texts, enc


# ---------------------------------------------------------------- 特征


def clean_text(text):
    """只留汉字：去掉空白、标点、英文数字与乱码，避免无意义 bigram。"""
    return NEG.sub("", text)


def bigram_counts(ctext):
    d = {}
    get = d.get
    for a, b in zip(ctext, ctext[1:]):
        k = a + b
        d[k] = get(k, 0) + 1
    return d


def top_grams(grams, k):
    """只保留权重最高/最高频的 k 个 bigram：够用，且快一个数量级。"""
    if len(grams) <= k:
        return grams
    return dict(sorted(grams.items(), key=lambda kv: -kv[1])[:k])


def simhash(ctext, bits=64, keep=2000):
    """近似重复指纹（用 crc32 而非会随机化的 hash()，跨进程/跨次稳定）。"""
    vec = [0] * bits
    for g, c in top_grams(bigram_counts(ctext), keep).items():
        h = zlib.crc32(g.encode("utf-8"))
        for i in range(bits):
            vec[i] += c if (h >> (i % 32)) & 1 else -c
    out = 0
    for i in range(bits):
        if vec[i] > 0:
            out |= 1 << i
    return out


def lexicon_scores(ctext):
    """加权命中密度（每万字）+ 证据量（总权重、命中词种数）+ 命中词。

    密度会被短文本放大（几百字的文件命中一个词就能算出很高的密度），
    所以另外返回绝对证据量，用来判断「这些分到底有没有依据」。
    """
    scale = 10000.0 / max(1, len(ctext))
    scores, evidence = {}, {}
    mass = terms = 0.0
    for genre, words in LEX.items():
        total = 0.0
        hits = []
        for term, w in words.items():
            c = ctext.count(term)
            if c:
                total += c * w
                mass += c * w
                terms += 1
                hits.append((c * w, term, c))
        scores[genre] = total * scale
        hits.sort(reverse=True)
        evidence[genre] = " ".join(f"{t}×{c}" for _, t, c in hits[:6])
    return scores, evidence, (mass, terms)


def _vec(grams, total, idf, topk=5000):
    """相对词频 × idf，保留 topk 后 L2 归一化。

    用相对词频而不是计数，长书短书才可比；乘 idf 让「所有题材都高频的
    通用 bigram」权重趋零，这是质心能不能分开题材的关键。
    """
    if not total:
        return {}
    vec = {}
    for g, c in grams.items():
        w = idf.get(g)
        if w:
            vec[g] = c / total * w
    if len(vec) > topk:
        vec = dict(sorted(vec.items(), key=lambda kv: -kv[1])[:topk])
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if not norm:
        return {}
    return {g: v / norm for g, v in vec.items()}


def centroid_scores(ctext, model):
    """与模型质心做余弦相似度（Rocchio 分类器）。"""
    grams = bigram_counts(ctext)
    if not grams:
        return {}, {}
    vec = _vec(grams, len(ctext), model["idf"])
    if not vec:
        return {}, {}
    scores, ev = {}, {}
    for genre, cen in model["centroids"].items():
        s, hits = 0.0, []
        for g, v in vec.items():
            cv = cen.get(g)
            if cv:
                s += v * cv
                hits.append((v * cv, g))
        scores[genre] = s
        hits.sort(reverse=True)
        ev[genre] = " ".join(g for _, g in hits[:6])
    return scores, ev


# ---------------------------------------------------------------- 判定

_W = {}   # 每个 worker 进程内的共享配置（避免逐任务 pickle 模型）


def _init_worker(model, want_simhash):
    _W.update(model=model, simhash=want_simhash)


def classify_file(job):
    """只做特征与打分，不作最终判定（判定要等全批语料的自校准）。"""
    path, budget = job
    cfg = _W
    rec = {"path": path, "size_mb": 0.0, "read_mb": 0.0, "encoding": "", "chars": 0,
           "label": DEFAULT_GENRE, "confidence": 0.0, "margin": 0.0,
           "second": "", "low_conf": "", "hint": "", "evidence": "",
           "simhash": "", "_ok": False}
    try:
        segs, size = read_sample(path, budget)
        rec["size_mb"] = round(size / 1048576.0, 2)
        rec["read_mb"] = round(sum(len(x) for x in segs) / 1048576.0, 3)
        texts, enc = decode_segments(segs)
        rec["encoding"] = enc
        cts = [c for c in (clean_text(t) for t in texts) if len(c) >= 100]
        if not cts:
            cts = [clean_text(t) for t in texts]
        rec["chars"] = sum(len(c) for c in cts)
        if rec["chars"] < MIN_CHARS:
            rec.update(label="过短/非正文", low_conf="yes", _ok=False,
                       evidence=f"仅 {rec['chars']} 个汉字")
            return rec

        # 每段各自打分，再按题材取中位数：局部污染（前言/目录/广告）不会翻盘
        lex_pairs = [lexicon_scores(c) for c in cts]
        lex_segs = [p[0] for p in lex_pairs]
        rec["_lex"] = {g: _median([d[g] for d in lex_segs]) for g in LEX}
        totals = [sum(d.values()) for d in lex_segs]
        mi = sorted(range(len(totals)), key=lambda i: totals[i])[len(totals) // 2]
        rec["_ev"] = lex_pairs[mi][1]
        rec["_mass"] = lex_pairs[mi][2]

        base = os.path.basename(path)
        rec["_hint"] = next((g for g, ks in FILENAME_HINTS.items()
                             if any(k in base for k in ks)), "")
        model = cfg.get("model")
        if model:
            cen_pairs = [centroid_scores(c, model) for c in cts]
            cen_segs = [p[0] for p in cen_pairs]
            ct_all = list(model["centroids"])
            rec["_cen"] = {g: _median([d.get(g, 0.0) for d in cen_segs])
                           for g in ct_all}
            rec["_ev_cen"] = cen_pairs[mi][1]
        if cfg.get("simhash"):
            rec["simhash"] = "%016x" % simhash("".join(cts))
        rec["_ok"] = True
        return rec
    except OSError as e:
        rec.update(label="读取失败", low_conf="yes",
                   evidence=str(e)[:100])
        return rec


def finalize(rows, threshold=0.45, calibrate="auto", model=None):
    """全批打分 → （可选）z 分数自校准 → 定标签、置信度、判定依据。

    自校准解决的问题：某些题材的词（都市的公司/手机、武侠的弟子）本身在
    所有小说里都有基础出现率，原始密度会把它们判成「谁都得高分」。整批
    一起看时，用各题材在本批语料里的均值方差归一，比较的才是「相对突出」。
    """
    usable = [r for r in rows if r.get("_ok")]
    for r in rows:
        if not r.get("_ok"):
            continue
        base = r.get("_cen") or r["_lex"]
        r["_base"] = base

    want_cal = calibrate == "on" or (calibrate == "auto"
                                    and len(usable) >= MIN_CALIB_N)
    stats = {}
    if want_cal:
        genres = {g for r in usable for g in r["_base"]}
        for g in genres:
            vals = [r["_base"].get(g, 0.0) for r in usable]
            n = len(vals)
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / n
            stats[g] = (mean, math.sqrt(var) or 1e-9)

    for r in rows:
        if not r.get("_ok"):
            continue
        base = r["_base"]
        if want_cal:
            sc = {g: (v - stats[g][0]) / stats[g][1] for g, v in base.items()}
        else:
            m = max(base.values()) if base else 0.0
            sc = {g: (v / m if m > 0 else 0.0) for g, v in base.items()}

        # 文件名先验：只放大本来就有分的题材，不硬判
        hint = r.get("_hint", "")
        if hint and sc.get(hint, 0) > 0:
            sc[hint] *= 1.0 + HINT_BOOST
            r["hint"] = hint

        ranked = sorted(sc.items(), key=lambda kv: -kv[1])
        top, top_s = ranked[0]
        second, sec_s = ranked[1] if len(ranked) > 1 else ("", 0.0)
        for g, v in ranked:
            r["s_" + g] = round(base.get(g, 0.0), 2)

        # 证据量门槛只对「内置词表」路径生效：训练模型时词表与自定义类别
        # 未必对应（比如你自己分的子类型），不能拿词表命中数去否掉它
        mass, nterms = r.get("_mass", (0.0, 0))
        weak = (model is None and not hint
                and (mass < MIN_HITS or nterms < MIN_TERMS))
        if top_s <= 0 or weak:
            r.update(label=hint or DEFAULT_GENRE, low_conf="yes",
                     evidence=(f"仅文件名提示" if hint else
                               f"证据不足：命中 {mass:.0f} 权重 / {nterms} 词种"))
            continue

        r["label"] = top
        if want_cal:
            # 用 z 差值过 logistic 得到 0~1 的把握度，且要求绝对 z 也够高
            d = top_s - sec_s
            r["confidence"] = round(1 / (1 + math.exp(-d)), 3)
            r["margin"] = round(d, 2)
            if top_s < 0.5 or d < 0.3:
                r["low_conf"] = "yes"
        else:
            total = sum(v for _, v in ranked) or 1.0
            r["confidence"] = round(top_s / total, 3)
            r["margin"] = round((top_s - sec_s) / top_s, 3)
            if r["confidence"] < threshold or r["margin"] < 0.15:
                r["low_conf"] = "yes"

        r["second"] = second
        ev = (r.get("_ev") or {}).get(top, "")
        if r.get("_ev_cen"):
            ev = (ev + " | " + r["_ev_cen"].get(top, "")).strip(" |")
        r["evidence"] = ev[:160]

    for r in rows:
        for k in [k for k in r if k.startswith("_")]:
            del r[k]
    return rows


def iter_txt(paths, exts, limit, exclude=()):
    """遍历 txt。exclude 是要避开的目录（落盘目标），否则会把自己的输出再扫一遍。"""
    n, seen = 0, set()
    ex = tuple(os.path.realpath(e) for e in exclude)

    def skip(p):
        return any(p == e or p.startswith(e + os.sep) for e in ex)

    for root_path in paths:
        if os.path.isfile(root_path):
            candidates = [root_path]
        else:
            candidates = []
            for dp, dns, fns in os.walk(root_path):
                rdp = os.path.realpath(dp)
                dns[:] = [d for d in dns if not skip(os.path.join(rdp, d))]
                candidates.extend(os.path.join(dp, f) for f in fns)
        for p in candidates:
            if os.path.splitext(p)[1].lower() not in exts:
                continue
            rp = os.path.realpath(p)
            if rp in seen or skip(rp):    # 硬链接/重复挂载去重
                continue
            seen.add(rp)
            yield p
            n += 1
            if limit and n >= limit:
                return


# ---------------------------------------------------------------- 落盘（移动/复制）

MANIFEST_NAME = ".move-manifest.csv"
FS_BAD = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
DEFAULT_SKIP = "过短/非正文,读取失败"


def safe_label(label):
    """类别名当目录名时换掉非法字符（Windows/群晖/威联通都不接受这些）。"""
    s = FS_BAD.sub("_", label).strip().strip(".")
    return s or DEFAULT_GENRE


def unique_path(dst):
    """目标重名就加 (2)(3)…，绝不覆盖已有文件。"""
    if not os.path.exists(dst):
        return dst
    root, ext = os.path.splitext(dst)
    i = 2
    while True:
        cand = f"{root} ({i}){ext}"
        if not os.path.exists(cand):
            return cand
        i += 1


def transfer(src, dst, copy_mode):
    """同盘直接 rename（瞬时）；跨盘/跨网盘 rename 会 EXDEV，退回 copy2+删除。"""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if copy_mode:
        shutil.copy2(src, dst)
        return
    try:
        os.rename(src, dst)
    except OSError as e:
        if e.errno != errno.EXDEV:
            raise
        shutil.move(src, dst)


def do_move(rows, dest, copy_mode=False, dry_run=False, skip_labels=(),
            manifest=None):
    """按判定结果把文件搬进 dest/<类别>/。逐行写台账，中断也能回滚。"""
    dest = os.path.abspath(dest)
    if not dry_run:
        os.makedirs(dest, exist_ok=True)
    manifest = os.path.abspath(manifest or os.path.join(dest, MANIFEST_NAME))
    stats = Counter()
    fh = None
    if not dry_run:
        fresh = not os.path.exists(manifest) or os.path.getsize(manifest) == 0
        fh = open(manifest, "a", encoding="utf-8-sig", newline="")
        w = csv.writer(fh)
        if fresh:
            w.writerow(["ts", "action", "src", "dst", "label",
                        "confidence", "ok", "error"])
    try:
        for i, r in enumerate(rows, 1):
            src = r["path"]
            label = r.get("label", DEFAULT_GENRE)
            if label in skip_labels:
                stats["跳过(类别不搬)"] += 1
                continue
            if not os.path.exists(src):
                stats["跳过(已不存在)"] += 1
                continue
            target_dir = os.path.join(dest, safe_label(label))
            want = os.path.join(target_dir, os.path.basename(src))
            dst = unique_path(want)
            if dst != want:
                stats["重名改名"] += 1
            action = "copy" if copy_mode else "move"
            if dry_run:
                print(f"[计划] {label:<8} {src}  ->  {dst}")
                stats["计划"] += 1
                continue
            try:
                transfer(src, dst, copy_mode)
                stats[action] += 1
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), action, src,
                            dst, label, r.get("confidence", ""), 1, ""])
            except OSError as e:
                stats["失败"] += 1
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), action, src,
                            dst, label, r.get("confidence", ""), 0,
                            str(e)[:200]])
                print(f"[失败] {src}: {e}", file=sys.stderr)
            fh.flush()          # 关键：逐行落盘，跑到一半断电也能按台账回滚
            if i % 500 == 0:
                print(f"  已处理 {i}/{len(rows)}", flush=True)
    finally:
        if fh:
            fh.close()
    return stats, manifest


def do_undo(manifest, dry_run=False):
    """按台账把文件搬回原位（逆序处理，原位被占则加后缀，不覆盖）。"""
    if not os.path.exists(manifest):
        sys.exit(f"台账不存在：{manifest}")
    with open(manifest, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    stats = Counter()
    for row in reversed(rows):
        if row.get("action") != "move" or row.get("ok") != "1":
            stats["跳过(记录)"] += 1
            continue
        src, dst = row["src"], row["dst"]
        if not os.path.exists(dst):
            stats["跳过(已不在目标处)"] += 1
            continue
        back = unique_path(src)
        if dry_run:
            print(f"[计划回滚] {dst}  ->  {back}")
            stats["计划"] += 1
            continue
        try:
            transfer(dst, back, False)
            stats["已回滚"] += 1
        except OSError as e:
            stats["失败"] += 1
            print(f"[失败] {dst}: {e}", file=sys.stderr)
    return stats, len(rows)


# ---------------------------------------------------------------- 训练


def do_train(seed_dir, out_model, budget):
    """seeds/<题材>/*.txt → 字符 bigram TF-IDF 质心模型（每本书各自归一后平均）。"""
    classes = [d for d in sorted(os.listdir(seed_dir))
               if os.path.isdir(os.path.join(seed_dir, d))]
    if not classes:
        sys.exit(f"{seed_dir} 下没有题材子目录")
    book_vecs, df = {}, Counter()
    for c in classes:
        vecs = []
        for p in iter_txt([os.path.join(seed_dir, c)], {".txt"}, 0):
            segs, _ = read_sample(p, budget)
            texts, _ = decode_segments(segs)
            ct = clean_text("".join(texts))
            grams = bigram_counts(ct)
            if len(grams) < 50:
                continue
            vecs.append((grams, len(ct)))
            for g in grams:
                df[g] += 1
        if not vecs:
            print(f"  ! {c} 无可用 txt，跳过", file=sys.stderr)
            continue
        book_vecs[c] = vecs
        print(f"  {c}: {len(vecs)} 本, {len(df)} 个候选 bigram")
    n_cls = len(book_vecs)
    # log(n_cls/df)：df=n_cls 的通用 bigram 权重变 0，直接消失
    idf = {g: math.log(n_cls / d) for g, d in df.items() if d < n_cls}
    centroids = {}
    for c, vecs in book_vecs.items():
        acc, cnt = {}, 0
        for grams, total in vecs:
            for g, v in _vec(grams, total, idf).items():
                acc[g] = acc.get(g, 0.0) + v
            cnt += 1
        norm = math.sqrt(sum(v * v for v in acc.values())) or 1.0
        centroids[c] = {g: v / (norm * cnt) for g, v in acc.items()}
    with open(out_model, "w", encoding="utf-8") as f:
        json.dump({"classes": list(centroids), "idf": idf,
                   "centroids": centroids, "budget": budget}, f,
                  ensure_ascii=False)
    print(f"模型已写入 {out_model}（{n_cls} 类 / {len(idf)} 维）")


# ---------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="本地批量小说风格归类（纯标准库，不联网、不上传全文）")
    ap.add_argument("paths", nargs="*", help="小说目录或 txt 文件")
    ap.add_argument("--out", default="novel_labels.csv", help="输出 CSV")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2,
                    help="并行进程数")
    ap.add_argument("--sample-bytes", type=int, default=500_000,
                    help="每本抽样字节数（默认 500KB）")
    ap.add_argument("--ext", default=".txt", help="扩展名，逗号分隔")
    ap.add_argument("--limit", type=int, default=0, help="最多处理几个文件")
    ap.add_argument("--model", help="已训练的质心模型 json")
    ap.add_argument("--train", metavar="SEED_DIR",
                    help="用 SEED_DIR/<题材>/*.txt 训练模型后退出")
    ap.add_argument("--model-out", default="model.json")
    ap.add_argument("--simhash", action="store_true",
                    help="额外输出近似重复指纹")
    ap.add_argument("--calibrate", choices=["auto", "on", "off"], default="auto",
                    help="整批 z 分数自校准（默认 auto：≥30 本时开启）")
    ap.add_argument("--threshold", type=float, default=0.45,
                    help="未校准时置信度低于此值标为低置信（默认 0.45）")
    ap.add_argument("--csv-encoding", default="utf-8-sig",
                    help="CSV 编码，默认带 BOM 便于 Excel 打开")
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--move-to", metavar="DIR",
                    help="真正落盘：把每个文件移动/复制到 DIR/<类别>/（不存在则创建）")
    ap.add_argument("--copy", action="store_true",
                    help="配合 --move-to：复制而不是移动（原文件保留）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印将要做什么，不改动任何文件")
    ap.add_argument("--skip-labels", default=DEFAULT_SKIP,
                    help=f"这些类别不搬，逗号分隔（默认 {DEFAULT_SKIP}）")
    ap.add_argument("--manifest",
                    help=f"落盘台账路径（默认 <DIR>/{MANIFEST_NAME}），--undo 用它回滚")
    ap.add_argument("--undo", metavar="MANIFEST",
                    help="按台账把文件搬回原位")
    a = ap.parse_args(argv)

    if a.undo:
        stats, total = do_undo(a.undo, a.dry_run)
        print(f"台账 {total} 行：")
        for k, v in stats.most_common():
            print(f"  {k}: {v}")
        return 0

    if a.train:
        do_train(a.train, a.model_out, a.sample_bytes)
        return 0
    if not a.paths:
        ap.error("需要给出待归类目录/文件（或用 --train 训练模型）")

    model = None
    if a.model:
        with open(a.model, encoding="utf-8") as f:
            model = json.load(f)

    exts = {e if e.startswith(".") else "." + e
            for e in a.ext.lower().split(",")}
    files = list(iter_txt(a.paths, exts, a.limit,
                          exclude=[a.move_to] if a.move_to else []))
    if not files:
        sys.exit("没有找到符合条件的文件")
    print(f"文件 {len(files)} 个 | 抽样 {a.sample_bytes/1024:.0f} KB/本 | "
          f"{a.jobs} 进程 | 判定：{'质心模型' if model else '内置词表'}"
          f"+{'z校准' if a.calibrate != 'off' else '原始分'}")

    t0 = time.time()
    rows, every = [], max(1, a.progress_every)
    jobs = [(p, a.sample_bytes) for p in files]
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=_init_worker,
                             initargs=(model, a.simhash)) as ex:
        for rec in ex.map(classify_file, jobs, chunksize=8):
            rows.append(rec)
            if len(rows) % every == 0:
                el = time.time() - t0
                print(f"  {len(rows)}/{len(files)}  {len(rows)/el:.1f} 本/秒  "
                      f"{el/60:.1f} 分钟", flush=True)
    scan_el = time.time() - t0

    rows = finalize(rows, a.threshold, a.calibrate, model)
    el = time.time() - t0

    genres = list(LEX) + [g for g in (model or {}).get("classes", [])
                          if g not in LEX]
    cols = ["path", "size_mb", "read_mb", "encoding", "chars", "label", "confidence",
            "margin", "second", "low_conf", "hint", "evidence"] + \
           ["s_" + g for g in genres] + (["simhash"] if a.simhash else [])
    with open(a.out, "w", encoding=a.csv_encoding, newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["label"] not in ("过短/非正文", "读取失败")]
    low = sum(1 for r in ok if r["low_conf"])
    corpus_mb = sum(r["size_mb"] for r in rows)
    sampled = sum(r.get("read_mb", 0.0) for r in rows)
    print("\n== 汇总 ==")
    for g, c in Counter(r["label"] for r in rows).most_common():
        print(f"  {g:<10} {c:>6}  {c/len(rows)*100:5.1f}%")
    print(f"  低置信 {low}/{len(ok)}（{low/max(1,len(ok))*100:.1f}%）需复核")
    print(f"  扫描 {scan_el/60:.1f} 分钟 | 合计 {el/60:.1f} 分钟 | "
          f"{len(rows)/scan_el:.1f} 本/秒")
    corpus_s = (f"{corpus_mb/1024:.1f} GB" if corpus_mb >= 1024
                else f"{corpus_mb:.0f} MB")
    read_s = f"{sampled:.1f} MB" if sampled < 10 else f"{sampled:.0f} MB"
    pct_s = (f"（{min(100.0, sampled/max(corpus_mb,1e-9)*100):.1f}%）"
             if corpus_mb >= 1 else "")
    print(f"  语料 {corpus_s}，实读抽样 {read_s}{pct_s}，不上传任何内容")
    print(f"  结果：{os.path.abspath(a.out)}")

    if a.move_to:
        skip = {x.strip() for x in a.skip_labels.split(",") if x.strip()}
        print(f"\n== {'复制' if a.copy else '移动'}到 {os.path.abspath(a.move_to)} "
              f"{'（演练，不动文件）' if a.dry_run else ''} ==")
        stats, mf = do_move(rows, a.move_to, a.copy, a.dry_run, skip, a.manifest)
        for k, v in stats.most_common():
            print(f"  {k}: {v}")
        if not a.dry_run:
            print(f"  台账（可 --undo 回滚）：{mf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
