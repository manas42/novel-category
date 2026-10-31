#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
selftest.py —— 用合成语料验证 novel_classify.py 的管道、速度、编码鲁棒性，
并做一张消融表（词表/质心 × 是否 z 校准）。

注意：合成语料只能证明「管道跑得通、慢不慢、GBK 能不能读、各模块有没有
接错」，不能证明真实准确率——真实准确率要在你自己的书上量。要量准确率，
把 20~30 本已确认题材的书按 seeds/<题材>/ 放好，跑 --train 后用留出集评估。

    python3 selftest.py
"""

import csv
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "novel_classify.py")

# 各题材主题词（刻意与 LEXICON 不完全重合，避免自测虚高）
BANK = {
    "武侠": {
        "who": ["少年", "刀客", "老镖头", "师叔", "香主", "女侠", "舵主"],
        "obj": ["长剑", "单刀", "铁扇", "软鞭", "判官笔", "飞蝗石"],
        "verb": ["递出一招", "欺身而上", "沉肩卸力", "以掌缘切下", "错步闪开"],
        "place": ["破庙", "长街", "渡口", "山寨聚义厅", "荒村野店"],
        "kw": ["江湖", "内息", "轻身功夫", "刀锋", "气血翻涌", "拳脚", "山门",
               "师兄", "斗笠", "骏马", "酒碗", "镖旗", "恩怨", "过招", "招式"],
    },
    "仙侠修真": {
        "who": ["少年", "散修", "白衣女修", "老怪物", "执事长老", "剑修"],
        "obj": ["飞剑", "玉简", "丹炉", "储物袋", "本命法宝", "阵旗"],
        "verb": ["掐诀念咒", "引动灵气", "吐纳周天", "祭出法宝", "踏空而行"],
        "place": ["灵脉洞府", "宗门广场", "试炼秘境", "幻阵", "药园"],
        "kw": ["修仙", "灵气", "灵力", "突破境界", "元婴", "筑基", "凝丹",
               "淬体", "闭关", "法宝", "雷劫", "仙缘", "道基", "真元"],
    },
    "玄幻奇幻": {
        "who": ["少年", "佣兵", "圣殿骑士", "大魔导师", "精灵射手", "兽人战士"],
        "obj": ["法杖", "魔晶", "符文长剑", "卷轴", "龙鳞盾"],
        "verb": ["吟唱魔咒", "挥动斗气", "召唤元素", "闪现突进", "斩出一道光刃"],
        "place": ["魔法学院", "荒原营地", "神殿废墟", "巨龙巢穴", "精灵森林"],
        "kw": ["魔法", "斗气", "魔晶", "精灵", "巨龙", "兽人", "亡灵", "神殿",
               "咒语", "巫妖", "异界", "血脉", "领主", "骑士团"],
    },
    "都市": {
        "who": ["林总", "陈经理", "实习生", "女明星", "出租车司机", "前台小姐"],
        "obj": ["手机", "合同", "公文包", "咖啡杯", "车钥匙", "名片"],
        "verb": ["打来电话", "签了字", "挤进地铁", "打开笔记本", "把车停进地库"],
        "place": ["写字楼", "会议室", "商场", "医院走廊", "一家小餐馆"],
        "kw": ["公司", "总裁", "项目", "客户", "工资", "股票", "加班", "面试",
               "微信", "直播", "粉丝", "别墅", "股东", "上市", "商圈"],
    },
    "科幻": {
        "who": ["舰长", "驾驶员", "工程师", "AI", "殖民者", "机师"],
        "obj": ["机甲", "能量炮", "数据核心", "终端", "维生舱", "曲率引擎"],
        "verb": ["启动引擎", "扫描星域", "校准坐标", "接入主脑", "弹出弹射舱"],
        "place": ["空间站", "舰桥", "殖民星", "陨石带", "实验室"],
        "kw": ["飞船", "战舰", "星球", "宇宙", "量子", "人工智能", "机器人",
               "星际", "虫族", "机甲", "曲率", "舰队", "基因", "全息"],
    },
    "历史": {
        "who": ["皇帝", "太子", "首辅", "边关将军", "御史", "藩王"],
        "obj": ["圣旨", "奏折", "虎符", "玉玺", "长枪", "粮草车"],
        "verb": ["下旨", "跪伏在地", "上疏弹劾", "点卯升帐", "催动兵马"],
        "place": ["朝堂", "御书房", "边关城楼", "府衙", "军营"],
        "kw": ["朝廷", "陛下", "科举", "知府", "藩王", "太监", "宰相", "六部",
               "钦差", "上朝", "龙椅", "年号", "封地", "粮草"],
    },
}

FILLER = ["于是", "只是", "忽然", "不过", "半晌", "众人", "话音未落", "与此同时",
          "他抬起头", "她笑了一下", "空气安静下来", "风从窗缝里灌进来",
          "没有人说话", "远处传来一声闷响"]

TEMPLATES = [
    "{who}握紧{obj}，{verb}。",
    "「{kw}的事，你不必再问。」{who}低声道。",
    "{place}里，{who}看着{obj}，心里盘算着{kw}。",
    "{filler}，{who}{verb}，{obj}在光下发暗。",
    "关于{kw}，{who}只字未提，转身走进{place}。",
    "{filler}。{kw}、{kw}、{obj}，全压在这一句话里。",
    "{who}想起{place}那个下午，{kw}的味道还在鼻端。",
]


def para(rng, genre):
    b = BANK[genre]
    out = []
    for _ in range(rng.randint(3, 7)):
        out.append(rng.choice(TEMPLATES).format(
            who=rng.choice(b["who"]), obj=rng.choice(b["obj"]),
            verb=rng.choice(b["verb"]), place=rng.choice(b["place"]),
            kw=rng.choice(b["kw"]), filler=rng.choice(FILLER)))
    return "".join(out)


def make_book(rng, genre, target_mb, contaminate=0.15):
    """85% 主题段落 + 15% 串味段落，模拟混合题材/引文。"""
    target = int(target_mb * 1024 * 1024)
    other = [g for g in BANK if g != genre]
    head = f"《测试{genre}样本》\n作者：某人\n\n"
    parts, size = [], 0
    while size < target:
        g = genre if rng.random() > contaminate else rng.choice(other)
        p = f"第{len(parts)}章 {rng.choice(BANK[g]['kw'])}\n\n{para(rng, g)}\n\n"
        parts.append(p)
        size += len(p.encode("utf-8"))
    return head + "".join(parts)


def write_book(path, text, enc):
    with open(path, "wb") as f:
        f.write(text.encode(enc, errors="replace"))


def build_corpus(root, per_genre=10, seed=7):
    rng = random.Random(seed)
    meta = {}
    for genre in BANK:
        d = os.path.join(root, genre)
        os.makedirs(d, exist_ok=True)
        for i in range(per_genre):
            mb = round(rng.uniform(0.4, 3.0), 2)
            enc = rng.choice(["utf-8", "utf-8", "gb18030", "utf-8-sig"])
            name = f"{genre}_{i:02d}_{rng.choice(BANK[genre]['kw'])}.txt"
            p = os.path.join(d, name)
            write_book(p, make_book(rng, genre, mb), enc)
            meta[p] = (genre, enc)
    return meta


def run(cmd):
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(f"命令失败：{' '.join(cmd)}")
    return r.stdout, time.time() - t


def load(csv_path):
    with open(csv_path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def score(rows, meta):
    total = correct = low = enc_ok = 0
    per, wrong = {}, []
    for r in rows:
        if r["path"] not in meta:
            continue
        truth, enc = meta[r["path"]]
        total += 1
        ok = r["label"] == truth
        correct += ok
        low += r["low_conf"] == "yes"
        enc_ok += r["encoding"] == enc
        t = per.setdefault(truth, [0, 0])
        t[0] += 1
        t[1] += ok
        if not ok:
            wrong.append((os.path.basename(r["path"]), truth, r["label"],
                          r["confidence"], r["evidence"][:34]))
    return {"acc": correct / total, "low": low / total, "enc": enc_ok,
            "n": total, "per": per, "wrong": wrong}


def report(tag, s, elapsed, n):
    print(f"    {tag:<24} 准确率 {s['acc']*100:5.1f}%  低置信 "
          f"{s['low']*100:5.1f}%  编码 {s['enc']}/{s['n']}  "
          f"{n/elapsed:.1f} 本/秒")
    return s["acc"]


def txts(d):
    out = []
    for dp, _, fs in os.walk(d):
        out.extend(os.path.join(dp, f) for f in fs if f.lower().endswith(".txt"))
    return out


def check(cond, msg):
    if not cond:
        raise SystemExit(f"    ✗ 失败：{msg}")
    print(f"      ✓ {msg}")


def check_moves(root):
    """落盘测试：dry-run 不动手、重名不覆盖、重跑不复发、台账能回滚。"""
    print("[5/5] 落盘：移动 / 重名 / 幂等 / 回滚")
    mv = os.path.join(root, "mv_src")
    dest = os.path.join(root, "sorted")
    for sub in ("a", "b", "c"):
        os.makedirs(os.path.join(mv, sub))
    rng = random.Random(11)
    for g in ("武侠", "都市", "科幻"):
        for i in range(3):
            write_book(os.path.join(mv, f"{g}_{i}.txt"), make_book(rng, g, 0.4), "utf-8")
    # a 与 c 都是武侠且同名 → 必须自动改名而不是覆盖；b 是都市同样叫同名 → 各归各类
    write_book(os.path.join(mv, "a", "同名.txt"), make_book(rng, "武侠", 0.4), "utf-8")
    write_book(os.path.join(mv, "b", "同名.txt"), make_book(rng, "都市", 0.4), "utf-8")
    write_book(os.path.join(mv, "c", "同名.txt"), make_book(rng, "武侠", 0.4), "utf-8")
    with open(os.path.join(mv, "太短.txt"), "w", encoding="utf-8") as f:
        f.write("他走进客栈。" * 3)
    before = len(txts(mv))

    # 1) dry-run：什么都不该改
    run([sys.executable, SCRIPT, mv, "--move-to", dest, "--dry-run",
         "--calibrate", "off"])
    check(not os.path.isdir(dest), "dry-run 没有创建目标目录、没有搬任何文件")
    check(len(txts(mv)) == before, f"dry-run 后原目录仍是 {before} 个文件")

    # 2) 真搬
    _, el = run([sys.executable, SCRIPT, mv, "--move-to", dest,
                 "--calibrate", "off"])
    in_dest = txts(dest)
    check(len(in_dest) == before - 1, f"搬走 {before - 1} 本，太短的文件留下")
    check(len(txts(mv)) == 1, "原目录只剩那个过短文件")
    names = sorted(os.path.basename(p) for p in in_dest
                   if os.path.dirname(p).endswith("武侠"))
    check("同名.txt" in names and "同名 (2).txt" in names,
          f"同名武侠文件自动改名不覆盖：{names}")
    check(len([p for p in in_dest if p.endswith("同名.txt")]) == 2,
          "都市与武侠各自的同名文件分别落袋")
    check(all(os.path.basename(os.path.dirname(p)) in
              ("武侠", "都市", "科幻") for p in in_dest),
          "全部落在 类别/ 子目录下")

    # 3) 幂等：再跑一次不该重复搬
    r = subprocess.run([sys.executable, SCRIPT, mv, "--move-to", dest,
                        "--calibrate", "off"], capture_output=True, text=True)
    out = r.stdout + r.stderr
    check(len(txts(dest)) == len(in_dest),
          "重跑后目标目录文件数不变（已搬走的不会重复搬）")
    check("跳过(类别不搬): 1" in out and "  move:" not in out,
          "重跑时把仅剩的过短文件按类别跳过，没有误搬")

    # 4) 回滚
    mf = os.path.join(dest, ".move-manifest.csv")
    check(os.path.exists(mf), "生成了台账 .move-manifest.csv")
    run([sys.executable, SCRIPT, "--undo", mf])
    check(len(txts(mv)) == before, f"回滚后原目录恢复 {before} 个文件")
    check(len(txts(dest)) == 0, "目标目录已清空")
    print(f"      搬运用时 {el:.1f}s")


def main():
    root = tempfile.mkdtemp(prefix="nc_selftest_")
    try:
        print(f"[1/4] 生成合成语料于 {root}")
        meta = build_corpus(root)
        sizes = [os.path.getsize(p) / 1048576 for p in meta]
        print(f"      {len(meta)} 本 / {sum(sizes):.0f} MB，单本 "
              f"{min(sizes):.1f}~{max(sizes):.1f} MB，编码混合 "
              f"utf-8/gb18030/utf-8-sig，每本 15% 串味段落")

        print("[2/4] 零样本（内置词表）消融")
        lex = {}
        for cal in ("off", "on"):
            out = os.path.join(root, f"lex_{cal}.csv")
            _, el = run([sys.executable, SCRIPT, root, "--out", out,
                         "--jobs", "4", "--calibrate", cal,
                         "--progress-every", "99999"])
            s = score(load(out), meta)
            lex[cal] = s
            report(f"词表 cal={cal}", s, el, len(meta))
        print("      每题材命中（cal=on）：")
        for g, (n, c) in lex["on"]["per"].items():
            print(f"        {g:<8} {c}/{n}")
        if lex["on"]["wrong"]:
            print("      错例：")
            for w in lex["on"]["wrong"][:5]:
                print(f"        {w[0]:<22} 真={w[1]} 判={w[2]} "
                      f"conf={w[3]} {w[4]}")

        print("[3/4] 训练质心模型（每题材留 3 本作留出集）")
        seeds = os.path.join(root, "seeds")
        test_dir = os.path.join(root, "holdout")
        os.makedirs(seeds)
        os.makedirs(test_dir)
        meta2 = {}
        for g in BANK:
            files = sorted(os.listdir(os.path.join(root, g)))
            os.makedirs(os.path.join(seeds, g))
            for f in files[:-3]:
                os.rename(os.path.join(root, g, f), os.path.join(seeds, g, f))
            for f in files[-3:]:
                dst = os.path.join(test_dir, f)
                os.rename(os.path.join(root, g, f), dst)
                meta2[dst] = (g, meta[os.path.join(root, g, f)][1])
        model = os.path.join(root, "model.json")
        out_t, _ = run([sys.executable, SCRIPT, "--train", seeds,
                        "--model-out", model])
        print("      " + out_t.strip().replace("\n", "\n      "))
        for cal in ("off", "on"):
            out = os.path.join(root, f"cen_{cal}.csv")
            _, el = run([sys.executable, SCRIPT, test_dir, "--model", model,
                         "--out", out, "--jobs", "4", "--calibrate", cal])
            report(f"质心 cal={cal}", score(load(out), meta2), el, len(meta2))
        print(f"      留出集仅 {len(meta2)} 本，z 校准样本量偏小；"
              f"真实语料上万本时更稳")

        print("[4/4] 边界用例")
        edge = os.path.join(root, "edge")
        os.makedirs(edge)
        open(os.path.join(edge, "empty.txt"), "w").close()
        with open(os.path.join(edge, "english.txt"), "w") as f:
            f.write("This is not a novel. " * 5000)
        with open(os.path.join(edge, "tiny.txt"), "w") as f:
            f.write("他走进客栈。" * 34)
        rng = random.Random(3)
        front = "".join(rng.choice(BANK["都市"]["kw"]) for _ in range(200000))
        write_book(os.path.join(edge, "武侠_前言污染.txt"),
                   front + "\n\n" + make_book(rng, "武侠", 3.0), "utf-8")
        write_book(os.path.join(edge, "ok.txt"),
                   "江湖恩怨未了，剑客独行。" * 5000, "gb18030")
        out3 = os.path.join(edge, "r.csv")
        run([sys.executable, SCRIPT, edge, "--out", out3, "--jobs", "1",
             "--calibrate", "off"])
        for r in load(out3):
            print(f"        {os.path.basename(r['path']):<16}"
                  f" → {r['label']:<8} conf={r['confidence']:<5} "
                  f"enc={r['encoding']:<10} {r['evidence'][:32]}")

        check_moves(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("\n自测完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
