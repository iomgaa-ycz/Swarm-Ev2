"""V5 实验 Phase 1/Phase 2 切换失败诊断分析脚本。

分析每个竞赛的 journal.json，诊断为什么 21/22 竞赛无法从 Phase 1 切换到 Phase 2。
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# === 配置 ===
WORKSPACE = "/home/yuchengzhang/Code/Swarm-Ev2/workspace"
GRADING_REPORT = os.path.join(WORKSPACE, "2026-02-24T07-32-44-GMT_grading_report.json")

# 4 个必需基因块（V6，与 gene_parser.py 一致）
REQUIRED_GENES = [
    "DATA", "MODEL", "TRAIN", "POSTPROCESS",
]


def parse_solution_genes(code: str) -> dict:
    """从代码中解析基因 section。"""
    genes = {}
    pattern = re.compile(r"^#\s*\[SECTION:\s*(\w+)\]", re.MULTILINE)
    matches = list(pattern.finditer(code))
    for i, match in enumerate(matches):
        section_name = match.group(1)
        start_idx = match.end()
        if i + 1 < len(matches):
            end_idx = matches[i + 1].start()
        else:
            end_idx = len(code)
        content = code[start_idx:end_idx].strip()
        genes[section_name] = content
    return genes


def validate_genes(genes: dict) -> bool:
    """验证是否包含全部 7 个必需基因块。"""
    return all(g in genes for g in REQUIRED_GENES)


def get_competition_dirs():
    """获取所有竞赛目录。"""
    dirs = []
    for item in sorted(os.listdir(WORKSPACE)):
        full_path = os.path.join(WORKSPACE, item)
        if os.path.isdir(full_path) and "_" in item:
            journal_path = os.path.join(full_path, "logs", "journal.json")
            if os.path.exists(journal_path):
                dirs.append((item, full_path, journal_path))
    return dirs


def analyze_journal(journal_path: str) -> dict:
    """分析单个竞赛的 journal.json。"""
    with open(journal_path) as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    if not nodes:
        return None

    # 基本统计
    total = len(nodes)
    task_types = Counter(n.get("task_type", "unknown") for n in nodes)

    good_nodes = [n for n in nodes if not n.get("is_buggy", True) and not n.get("dead", False)]
    dead_nodes = [n for n in nodes if n.get("dead", False)]
    buggy_nodes = [n for n in nodes if n.get("is_buggy", False)]

    # 基因分析
    gene_section_counts = []  # 每个节点的基因 section 数量
    valid_gene_nodes = []  # 通过 7 段验证的节点
    good_but_incomplete = []  # good 但基因不完整的节点
    section_presence = Counter()  # 每个 section 出现次数

    # 按 code 字段重新解析（因为 journal 中存储的 genes 可能已经过滤）
    for n in nodes:
        # 优先使用 journal 中存储的 genes
        genes = n.get("genes", {})
        if not genes and n.get("code"):
            genes = parse_solution_genes(n["code"])

        if isinstance(genes, dict):
            gene_count = len(genes)
            gene_section_counts.append(gene_count)

            for section in genes:
                section_presence[section] += 1

            is_valid = validate_genes(genes)
            is_good = not n.get("is_buggy", True) and not n.get("dead", False)

            if is_valid:
                valid_gene_nodes.append(n)

            if is_good and not is_valid:
                missing = [g for g in REQUIRED_GENES if g not in genes]
                good_but_incomplete.append({
                    "id": n.get("id", "?")[:8],
                    "task_type": n.get("task_type", "?"),
                    "gene_count": gene_count,
                    "has_sections": sorted(genes.keys()),
                    "missing": missing,
                    "metric": n.get("metric_value"),
                })
        else:
            gene_section_counts.append(0)

    # 基因数量分布
    gene_count_dist = Counter(gene_section_counts)

    # valid_pool 按 main.py 逻辑计算
    valid_pool = [
        n for n in nodes
        if not n.get("is_buggy", True) and not n.get("dead", False)
        and validate_genes(n.get("genes", {}))
    ]

    # 时间分析
    ctimes = [n.get("ctime", 0) for n in nodes if n.get("ctime")]
    exec_times = [n.get("exec_time", 0) for n in nodes if n.get("exec_time")]

    first_time = min(ctimes) if ctimes else 0
    last_time = max(ctimes) if ctimes else 0
    duration = last_time - first_time

    # valid_pool 增长时间线
    valid_timeline = []
    running_valid = 0
    for i, n in enumerate(nodes):
        genes = n.get("genes", {})
        is_good = not n.get("is_buggy", True) and not n.get("dead", False)
        is_gene_valid = validate_genes(genes) if isinstance(genes, dict) else False
        if is_good and is_gene_valid:
            running_valid += 1
            if running_valid <= 10:  # 只记录前10个
                valid_timeline.append({
                    "node_idx": i,
                    "valid_count": running_valid,
                    "ctime": n.get("ctime", 0),
                    "elapsed": n.get("ctime", 0) - first_time if first_time else 0,
                })

    # Phase 2 检测
    has_phase2 = task_types.get("merge", 0) > 0 or task_types.get("mutate", 0) > 0

    # 第一个 merge/mutate 节点的时间
    phase2_start = None
    for n in nodes:
        if n.get("task_type") in ("merge", "mutate"):
            phase2_start = {
                "node_idx": nodes.index(n),
                "ctime": n.get("ctime", 0),
                "elapsed": n.get("ctime", 0) - first_time if first_time else 0,
            }
            break

    # best metric
    metrics = [n.get("metric_value") for n in good_nodes if n.get("metric_value") is not None]

    return {
        "total_nodes": total,
        "task_types": dict(task_types),
        "good_count": len(good_nodes),
        "dead_count": len(dead_nodes),
        "buggy_count": len(buggy_nodes),
        "valid_gene_nodes": len(valid_gene_nodes),
        "valid_pool_count": len(valid_pool),
        "valid_pool_rate": len(valid_pool) / total if total > 0 else 0,
        "good_but_incomplete_count": len(good_but_incomplete),
        "good_but_incomplete_samples": good_but_incomplete[:5],
        "gene_count_distribution": dict(sorted(gene_count_dist.items())),
        "section_presence": dict(section_presence.most_common()),
        "has_phase2": has_phase2,
        "phase2_start": phase2_start,
        "duration_seconds": duration,
        "avg_exec_time": sum(exec_times) / len(exec_times) if exec_times else 0,
        "valid_timeline": valid_timeline,
        "best_metric": min(metrics) if metrics else None,  # 假设 lower is better
        "metrics_count": len(metrics),
    }


def analyze_system_log(comp_dir: str) -> dict:
    """分析 system.log 中的时间和 Phase 信息。"""
    log_path = os.path.join(comp_dir, "logs", "system.log")
    if not os.path.exists(log_path):
        return {"exists": False}

    result = {
        "exists": True,
        "phase1_valid_logs": [],
        "phase2_logs": [],
        "timeout_errors": 0,
        "total_lines": 0,
    }

    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                result["total_lines"] += 1

                # 查找 Phase 1 valid_pool 日志
                if "Phase 1 |" in line and "valid=" in line:
                    result["phase1_valid_logs"].append(line.strip()[:200])

                # 查找 Phase 2 日志
                if "Phase 2" in line:
                    result["phase2_logs"].append(line.strip()[:200])

                # 查找 TimeoutError
                if "TimeoutError" in line or "timeout" in line.lower():
                    result["timeout_errors"] += 1
    except Exception as e:
        result["error"] = str(e)

    return result


def load_grading_report():
    """加载 grading report。"""
    if not os.path.exists(GRADING_REPORT):
        return {}
    with open(GRADING_REPORT) as f:
        data = json.load(f)

    results = {}
    reports = data.get("competition_reports", [])
    for item in reports:
        comp_id = item.get("competition_id", "")
        # 确定奖牌
        if item.get("gold_medal"):
            medal = "gold"
        elif item.get("silver_medal"):
            medal = "silver"
        elif item.get("bronze_medal"):
            medal = "bronze"
        else:
            medal = "-"
        results[comp_id] = {
            "score": item.get("score"),
            "medal": medal,
            "gold_threshold": item.get("gold_threshold"),
            "silver_threshold": item.get("silver_threshold"),
            "bronze_threshold": item.get("bronze_threshold"),
        }
    return results


def main():
    print("=" * 100)
    print("V5 实验 Phase 1 → Phase 2 切换失败诊断报告")
    print("=" * 100)
    print()

    # 加载 grading report
    grading = load_grading_report()

    # 获取所有竞赛
    competitions = get_competition_dirs()
    print(f"共找到 {len(competitions)} 个竞赛目录\n")

    all_results = {}

    for comp_name, comp_dir, journal_path in competitions:
        # 提取竞赛简名
        short_name = comp_name.split("_")[0]
        if len(short_name) > 40:
            short_name = short_name[:37] + "..."

        result = analyze_journal(journal_path)
        if result is None:
            continue

        log_result = analyze_system_log(comp_dir)
        result["system_log"] = log_result

        # 匹配 grading
        for gk, gv in grading.items():
            if gk in comp_name or comp_name.startswith(gk):
                result["grading"] = gv
                break

        all_results[short_name] = result

    # ===================================================================
    # 1. 总览表格
    # ===================================================================
    print("=" * 100)
    print("1. 各竞赛 valid_pool 统计总览")
    print("=" * 100)
    print()

    header = f"{'竞赛':<45} {'总节点':>6} {'good':>5} {'dead':>5} {'valid_pool':>10} {'通过率':>7} {'Phase2':>7} {'奖牌':>6}"
    print(header)
    print("-" * 100)

    phase2_count = 0
    total_valid_sum = 0
    total_nodes_sum = 0

    for name, r in sorted(all_results.items()):
        total_nodes_sum += r["total_nodes"]
        total_valid_sum += r["valid_pool_count"]

        phase2_str = "YES" if r["has_phase2"] else "NO"
        if r["has_phase2"]:
            phase2_count += 1

        medal = r.get("grading", {}).get("medal", "-")
        rate = f"{r['valid_pool_rate']*100:.1f}%"

        print(f"{name:<45} {r['total_nodes']:>6} {r['good_count']:>5} {r['dead_count']:>5} "
              f"{r['valid_pool_count']:>10} {rate:>7} {phase2_str:>7} {str(medal):>6}")

    print("-" * 100)
    overall_rate = total_valid_sum / total_nodes_sum * 100 if total_nodes_sum > 0 else 0
    print(f"{'总计/平均':<45} {total_nodes_sum:>6} {'':>5} {'':>5} {total_valid_sum:>10} {overall_rate:.1f}%   {phase2_count}/{len(all_results)}")
    print()

    # ===================================================================
    # 2. 基因数量分布分析
    # ===================================================================
    print("=" * 100)
    print("2. 基因 Section 数量分布（全局）")
    print("=" * 100)
    print()

    global_gene_dist = Counter()
    for name, r in all_results.items():
        for k, v in r["gene_count_distribution"].items():
            global_gene_dist[int(k)] += v

    print(f"{'基因数量':>10} {'节点数':>10} {'占比':>10}")
    print("-" * 35)
    total_all = sum(global_gene_dist.values())
    for k in sorted(global_gene_dist.keys()):
        pct = global_gene_dist[k] / total_all * 100
        bar = "#" * int(pct / 2)
        print(f"{k:>10} {global_gene_dist[k]:>10} {pct:>9.1f}% {bar}")

    print()
    print(f"需要 7 个 section 才能通过验证 (REQUIRED_GENES)")
    genes_7_or_more = sum(v for k, v in global_gene_dist.items() if k >= 7)
    print(f"基因数 >= 7 的节点: {genes_7_or_more}/{total_all} ({genes_7_or_more/total_all*100:.1f}%)")
    print()

    # ===================================================================
    # 3. 缺失基因分析
    # ===================================================================
    print("=" * 100)
    print("3. good 但基因不完整的节点分析（被 valid_pool 拒绝的潜力节点）")
    print("=" * 100)
    print()

    # 统计全局缺失的 section
    missing_counter = Counter()
    total_good_incomplete = 0
    total_good_complete = 0

    for name, r in all_results.items():
        total_good_incomplete += r["good_but_incomplete_count"]
        total_good_complete += r["valid_pool_count"]

        for sample in r.get("good_but_incomplete_samples", []):
            for m in sample.get("missing", []):
                missing_counter[m] += 1

    # 更精确地统计：遍历所有节点
    full_missing_counter = Counter()
    good_incomplete_per_comp = {}

    for comp_name, comp_dir, journal_path in competitions:
        short_name = comp_name.split("_")[0]
        with open(journal_path) as f:
            data = json.load(f)

        nodes = data.get("nodes", [])
        incomplete = 0
        for n in nodes:
            is_good = not n.get("is_buggy", True) and not n.get("dead", False)
            genes = n.get("genes", {})
            if is_good and isinstance(genes, dict) and not validate_genes(genes):
                incomplete += 1
                for g in REQUIRED_GENES:
                    if g not in genes:
                        full_missing_counter[g] += 1
        good_incomplete_per_comp[short_name] = incomplete

    print(f"全局统计: good 且基因完整 (valid_pool) = {total_good_complete}")
    print(f"全局统计: good 但基因不完整 = {sum(good_incomplete_per_comp.values())}")
    print(f"全局统计: 基因不完整占所有 good 的比例 = {sum(good_incomplete_per_comp.values()) / (total_good_complete + sum(good_incomplete_per_comp.values())) * 100:.1f}%")
    print()

    print("各 REQUIRED_GENE 的缺失频率:")
    print(f"{'基因 Section':<20} {'缺失次数':>10} {'缺失率':>10}")
    print("-" * 45)
    total_incomplete_all = sum(good_incomplete_per_comp.values())
    for gene in REQUIRED_GENES:
        cnt = full_missing_counter[gene]
        rate = cnt / total_incomplete_all * 100 if total_incomplete_all > 0 else 0
        print(f"{gene:<20} {cnt:>10} {rate:>9.1f}%")
    print()

    # ===================================================================
    # 4. 竞赛级别详细基因分布
    # ===================================================================
    print("=" * 100)
    print("4. 每个竞赛的基因完整度详情")
    print("=" * 100)
    print()

    header = f"{'竞赛':<45} {'good':>5} {'valid':>6} {'incomplete':>10} {'gap_to_8':>8}"
    print(header)
    print("-" * 80)

    for name, r in sorted(all_results.items()):
        gap = max(0, 8 - r["valid_pool_count"])
        incomplete = good_incomplete_per_comp.get(name, 0)
        print(f"{name:<45} {r['good_count']:>5} {r['valid_pool_count']:>6} {incomplete:>10} {gap:>8}")

    print()

    # ===================================================================
    # 5. 时间分析
    # ===================================================================
    print("=" * 100)
    print("5. 时间分析")
    print("=" * 100)
    print()

    print(f"{'竞赛':<45} {'耗时(h)':>8} {'节点数':>6} {'平均耗时/节点(s)':>16} {'Phase1日志条':>12}")
    print("-" * 95)

    for name, r in sorted(all_results.items()):
        duration_h = r["duration_seconds"] / 3600
        avg_exec = r["avg_exec_time"]
        log_count = len(r.get("system_log", {}).get("phase1_valid_logs", []))
        print(f"{name:<45} {duration_h:>8.2f} {r['total_nodes']:>6} {avg_exec:>16.1f} {log_count:>12}")

    print()

    # ===================================================================
    # 6. nomad2018 成功案例分析
    # ===================================================================
    print("=" * 100)
    print("6. nomad2018 成功案例深入分析")
    print("=" * 100)
    print()

    nomad = None
    for name, r in all_results.items():
        if "nomad2018" in name:
            nomad = r
            nomad_name = name
            break

    if nomad:
        print(f"竞赛: {nomad_name}")
        print(f"总节点: {nomad['total_nodes']}")
        print(f"task_type 分布: {nomad['task_types']}")
        print(f"valid_pool 数量: {nomad['valid_pool_count']}")
        print(f"good 但不完整: {nomad['good_but_incomplete_count']}")
        print(f"总耗时: {nomad['duration_seconds']/3600:.2f} 小时")
        print()

        print("valid_pool 增长时间线:")
        for vt in nomad["valid_timeline"]:
            elapsed_min = vt["elapsed"] / 60
            print(f"  第 {vt['node_idx']:>3} 个节点 → valid_pool={vt['valid_count']}, 已过 {elapsed_min:.1f} 分钟")

        if nomad["phase2_start"]:
            p2 = nomad["phase2_start"]
            print(f"\nPhase 2 开始:")
            print(f"  节点索引: {p2['node_idx']}")
            print(f"  已过时间: {p2['elapsed']/60:.1f} 分钟 ({p2['elapsed']/3600:.2f} 小时)")

        print()
        print("基因数量分布:")
        for k, v in sorted(nomad["gene_count_distribution"].items()):
            print(f"  {k} 个 section: {v} 个节点")

        print()
        print("Section 出现频率:")
        for section, count in sorted(nomad["section_presence"].items(), key=lambda x: -x[1]):
            print(f"  {section:<20}: {count} 次")
    else:
        print("未找到 nomad2018 数据")

    print()

    # ===================================================================
    # 7. 对比分析：nomad2018 vs 其他竞赛
    # ===================================================================
    print("=" * 100)
    print("7. 对比分析：nomad2018 vs 其余竞赛")
    print("=" * 100)
    print()

    if nomad:
        others = {k: v for k, v in all_results.items() if "nomad2018" not in k}

        avg_total = sum(v["total_nodes"] for v in others.values()) / len(others) if others else 0
        avg_good = sum(v["good_count"] for v in others.values()) / len(others) if others else 0
        avg_valid = sum(v["valid_pool_count"] for v in others.values()) / len(others) if others else 0
        avg_rate = sum(v["valid_pool_rate"] for v in others.values()) / len(others) if others else 0

        print(f"{'指标':<30} {'nomad2018':>15} {'其他平均':>15}")
        print("-" * 65)
        print(f"{'总节点数':<30} {nomad['total_nodes']:>15} {avg_total:>15.1f}")
        print(f"{'good 节点数':<30} {nomad['good_count']:>15} {avg_good:>15.1f}")
        print(f"{'valid_pool 数量':<30} {nomad['valid_pool_count']:>15} {avg_valid:>15.1f}")
        print(f"{'valid_pool 通过率':<30} {nomad['valid_pool_rate']*100:>14.1f}% {avg_rate*100:>14.1f}%")
        print(f"{'平均执行时间(s)':<30} {nomad['avg_exec_time']:>15.1f} {sum(v['avg_exec_time'] for v in others.values())/len(others):>15.1f}")

    print()

    # ===================================================================
    # 8. 根因分类
    # ===================================================================
    print("=" * 100)
    print("8. Phase 1→2 切换失败根因分类")
    print("=" * 100)
    print()

    # 分类逻辑：
    # A: valid_pool >= 8 但时间耗尽 → 时间不足（不太可能，因为 valid>=8 就会进入 Phase2）
    # B: valid_pool 在 1-7 之间 → 基因验证过严
    # C: valid_pool == 0 → good 节点太少或全部基因不完整
    # D: good 节点充足但 valid=0 → 纯基因验证问题

    categories = defaultdict(list)

    for name, r in all_results.items():
        if r["has_phase2"]:
            categories["SUCCESS"].append(name)
        elif r["valid_pool_count"] >= 8:
            categories["A_TIME_EXHAUSTED"].append(name)  # 理论上不应该出现
        elif r["valid_pool_count"] >= 1:
            categories["B_GENE_STRICT"].append(name)
        elif r["good_count"] >= 8 and r["valid_pool_count"] == 0:
            categories["D_PURE_GENE_ISSUE"].append(name)
        elif r["good_count"] > 0 and r["valid_pool_count"] == 0:
            categories["C_FEW_GOOD_AND_GENE"].append(name)
        else:
            categories["E_NO_GOOD"].append(name)

    category_labels = {
        "SUCCESS": "成功进入 Phase 2",
        "A_TIME_EXHAUSTED": "valid_pool>=8 但时间耗尽",
        "B_GENE_STRICT": "有部分 valid (1-7) 但未达 K=8",
        "C_FEW_GOOD_AND_GENE": "good 不足 + 基因不完整",
        "D_PURE_GENE_ISSUE": "good>=8 但 valid=0（纯基因问题）",
        "E_NO_GOOD": "无 good 节点",
    }

    for cat, label in category_labels.items():
        comps = categories.get(cat, [])
        print(f"[{cat}] {label}: {len(comps)} 个竞赛")
        for c in comps:
            r = all_results[c]
            print(f"  - {c}: total={r['total_nodes']}, good={r['good_count']}, valid_pool={r['valid_pool_count']}")
        print()

    # ===================================================================
    # 9. 假设分析：如果降低基因要求
    # ===================================================================
    print("=" * 100)
    print("9. 假设分析：降低基因 Section 要求后的 valid_pool")
    print("=" * 100)
    print()

    # 模拟不同的基因要求
    thresholds = [7, 6, 5, 4, 3, 2, 1]

    print(f"{'竞赛':<45}", end="")
    for t in thresholds:
        print(f" {f'≥{t}段':>6}", end="")
    print(f" {'good':>6}")
    print("-" * (45 + 7 * len(thresholds) + 7))

    would_phase2 = defaultdict(int)  # threshold → 能进入 Phase 2 的竞赛数

    for comp_name, comp_dir, journal_path in sorted(competitions, key=lambda x: x[0]):
        short_name = comp_name.split("_")[0]
        if short_name not in all_results:
            continue

        with open(journal_path) as f:
            data = json.load(f)
        nodes = data.get("nodes", [])

        print(f"{short_name:<45}", end="")

        for t in thresholds:
            count = 0
            for n in nodes:
                is_good = not n.get("is_buggy", True) and not n.get("dead", False)
                genes = n.get("genes", {})
                if is_good and isinstance(genes, dict) and len(genes) >= t:
                    count += 1

            if count >= 8:
                would_phase2[t] += 1
                print(f" {count:>5}*", end="")  # * 表示能进入 Phase 2
            else:
                print(f" {count:>6}", end="")

        good_count = all_results[short_name]["good_count"]
        print(f" {good_count:>6}")

    print("-" * (45 + 7 * len(thresholds) + 7))
    print(f"{'能进入 Phase 2 的竞赛数':<45}", end="")
    for t in thresholds:
        print(f" {would_phase2[t]:>5}*", end="")
    print()
    print("(*号表示 valid_pool >= 8，可触发 Phase 2)")
    print()

    # ===================================================================
    # 10. 假设分析：如果降低 K 阈值
    # ===================================================================
    print("=" * 100)
    print("10. 假设分析：降低 Phase 2 触发阈值 K")
    print("=" * 100)
    print()

    k_values = [8, 7, 6, 5, 4, 3, 2, 1]

    print(f"{'竞赛':<45} {'valid_pool':>10}", end="")
    for k in k_values:
        print(f" {'K='+str(k):>5}", end="")
    print()
    print("-" * (45 + 10 + 6 * len(k_values)))

    k_phase2_count = defaultdict(int)

    for name, r in sorted(all_results.items()):
        vp = r["valid_pool_count"]
        print(f"{name:<45} {vp:>10}", end="")
        for k in k_values:
            if vp >= k:
                k_phase2_count[k] += 1
                print(f" {'YES':>5}", end="")
            else:
                print(f" {'NO':>5}", end="")
        print()

    print("-" * (45 + 10 + 6 * len(k_values)))
    print(f"{'进入 Phase 2 竞赛数':<45} {'':>10}", end="")
    for k in k_values:
        print(f" {k_phase2_count[k]:>5}", end="")
    print()
    print()

    # ===================================================================
    # 11. 结论与建议
    # ===================================================================
    print("=" * 100)
    print("11. 结论与建议")
    print("=" * 100)
    print()

    print("【核心发现】")
    print(f"  1. 22 个竞赛中仅 {phase2_count} 个进入 Phase 2")
    print(f"  2. 全局 valid_pool 通过率: {overall_rate:.1f}%")
    print(f"  3. 基因数 >= 7 的节点仅占 {genes_7_or_more/total_all*100:.1f}%")
    print()

    print("【根因分析】")
    print("  根因 1: 7 段基因验证过严")
    print(f"    - 全局仅 {genes_7_or_more}/{total_all} 个节点有 >= 7 个基因 section")
    most_missing = full_missing_counter.most_common(3)
    if most_missing:
        print(f"    - 最常缺失的 section: {', '.join(f'{g[0]}({g[1]}次)' for g in most_missing)}")
    print()

    print("  根因 2: K=8 阈值过高")
    for k in [4, 3, 2]:
        print(f"    - 若 K={k}: {k_phase2_count[k]}/{len(all_results)} 竞赛可进入 Phase 2")
    print()

    print("  根因 3: 双重瓶颈叠加（基因验证严 + K 值高）")
    print("    - 即使 good 节点足够，也因基因不完整而被拒绝")
    print("    - 被拒绝后无法积累 valid_pool，K=8 永远达不到")
    print()

    print("【建议修复方案】")
    print("  方案 A（推荐）: 降低基因 section 要求从 7 段到 4 段")
    for t in [4, 5]:
        print(f"    - 若要求 >= {t} 段: {would_phase2[t]} 个竞赛可进入 Phase 2 (K=8)")
    print()
    print("  方案 B: 降低 K 值从 8 到 3-4")
    for k in [3, 4]:
        print(f"    - 若 K={k} (保持 7 段要求): {k_phase2_count[k]} 个竞赛可进入 Phase 2")
    print()
    print("  方案 C（最佳）: 同时降低基因要求到 4 段 + K 值降到 4")
    print("    - 预期: 大部分竞赛均可进入 Phase 2")
    print()
    print("  方案 D: 取消 Phase 1/Phase 2 时间分离，GA 全程可用")
    print("    - 当 valid_pool >= K 时即触发 GA，不等待 Phase 1 完成")
    print("    - 避免 Phase 1 消耗全部时间预算")


if __name__ == "__main__":
    main()
