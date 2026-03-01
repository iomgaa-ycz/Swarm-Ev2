#!/usr/bin/env python3
"""V6 实验结果自动化分析脚本 - 解析所有竞赛的 journal.json"""
import json
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

WORKSPACE = Path("/home/yuchengzhang/Code/Swarm-Ev2/workspace")

def parse_journal(journal_path):
    """解析单个竞赛的 journal.json"""
    with open(journal_path) as f:
        data = json.load(f)
    
    nodes = data.get("nodes", [])
    stats = {
        "total_nodes": len(nodes),
        "task_types": Counter(),
        "buggy_by_type": defaultdict(lambda: {"total": 0, "buggy": 0}),
        "exc_types": Counter(),
        "metrics": [],
        "best_metric": None,
        "best_node_id": None,
        "lower_is_better": None,
        "exec_times": [],
        "dead_nodes": 0,
        "debug_attempts_dist": Counter(),
        "gene_counts": [],  # 每个节点有多少基因
        "has_merge": False,
        "has_mutate": False,
        "merge_count": 0,
        "mutate_count": 0,
        "draft_count": 0,
    }
    
    for node in nodes:
        task_type = node.get("task_type", "unknown")
        stats["task_types"][task_type] += 1
        
        if task_type == "draft":
            stats["draft_count"] += 1
        elif task_type == "merge":
            stats["merge_count"] += 1
            stats["has_merge"] = True
        elif task_type == "mutate":
            stats["mutate_count"] += 1
            stats["has_mutate"] = True
        
        is_buggy = node.get("is_buggy", None)
        stats["buggy_by_type"][task_type]["total"] += 1
        if is_buggy:
            stats["buggy_by_type"][task_type]["buggy"] += 1
        
        if node.get("exc_type"):
            stats["exc_types"][node["exc_type"]] += 1
        
        metric = node.get("metric_value")
        if metric is not None:
            stats["metrics"].append(metric)
        
        lower = node.get("lower_is_better")
        if lower is not None:
            stats["lower_is_better"] = lower
        
        exec_time = node.get("exec_time")
        if exec_time is not None:
            stats["exec_times"].append(exec_time)
        
        if node.get("dead"):
            stats["dead_nodes"] += 1
        
        debug_att = node.get("debug_attempts", 0)
        stats["debug_attempts_dist"][debug_att] += 1
        
        genes = node.get("genes", {})
        if isinstance(genes, dict):
            stats["gene_counts"].append(len(genes))
    
    # 计算最佳metric
    if stats["metrics"]:
        if stats["lower_is_better"]:
            stats["best_metric"] = min(stats["metrics"])
        else:
            stats["best_metric"] = max(stats["metrics"])
        
        # 找最佳节点
        for node in nodes:
            if node.get("metric_value") == stats["best_metric"]:
                stats["best_node_id"] = node.get("id", "?")
                break
    
    return stats


def parse_system_log(log_path):
    """从 system.log 提取关键信息"""
    info = {
        "phase2_triggered": False,
        "phase2_lines": [],
        "timeout_errors": 0,
        "total_errors": 0,
        "adaptive_timeout": None,
        "ga_trigger_threshold": None,
    }
    
    if not log_path.exists():
        return info
    
    with open(log_path) as f:
        for line in f:
            if "Phase 2" in line or "phase 2" in line or "phase2" in line.lower():
                info["phase2_triggered"] = True
                info["phase2_lines"].append(line.strip())
            if "TimeoutError" in line:
                info["timeout_errors"] += 1
            if "[ERROR]" in line:
                info["total_errors"] += 1
            if "自适应超时" in line:
                info["adaptive_timeout"] = line.strip()
            if "GA 触发阈值" in line or "GA触发" in line:
                info["ga_trigger_threshold"] = line.strip()
            if "混合模式" in line or "mixed_mode" in line:
                info["phase2_lines"].append(line.strip())
    
    return info


def main():
    # 加载 grading report
    grading_path = WORKSPACE / "2026-02-28T14-22-07-GMT_grading_report.json"
    with open(grading_path) as f:
        grading = json.load(f)
    
    # 建立竞赛ID映射
    comp_scores = {}
    for report in grading["competition_reports"]:
        comp_scores[report["competition_id"]] = report
    
    print("=" * 120)
    print("V6 MLE-Bench 实验结果自动化分析")
    print("=" * 120)
    
    # 统计汇总
    all_stats = {}
    
    for comp_dir in sorted(WORKSPACE.iterdir()):
        if not comp_dir.is_dir():
            continue
        
        comp_name = comp_dir.name.rsplit("_", 5)[0]  # 去掉UUID后缀
        # 更准确地提取竞赛名
        parts = comp_dir.name.split("_")
        # UUID 是最后5段（格式: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx）
        # 找到UUID开始位置
        uuid_start = None
        for i in range(len(parts)):
            remaining = "_".join(parts[i:])
            if len(remaining) == 36 and remaining.count("-") == 4:
                uuid_start = i
                break
        
        if uuid_start:
            comp_name = "-".join(parts[:uuid_start])  # 用-连接
        else:
            comp_name = comp_dir.name[:comp_dir.name.rfind("_")]
        
        # 尝试匹配 grading report 中的名称
        matched_comp_id = None
        for cid in comp_scores:
            if cid in comp_dir.name:
                matched_comp_id = cid
                break
        
        journal_path = comp_dir / "logs" / "journal.json"
        system_log_path = comp_dir / "logs" / "system.log"
        
        if not journal_path.exists():
            print(f"\n### {comp_name}: journal.json 不存在")
            all_stats[comp_name] = {"error": "no journal"}
            continue
        
        stats = parse_journal(journal_path)
        log_info = parse_system_log(system_log_path)
        
        # 合并 grading 数据
        if matched_comp_id:
            gr = comp_scores[matched_comp_id]
            stats["final_score"] = gr["score"]
            stats["gold_threshold"] = gr["gold_threshold"]
            stats["silver_threshold"] = gr["silver_threshold"]
            stats["bronze_threshold"] = gr["bronze_threshold"]
            stats["medal"] = "🥇" if gr["gold_medal"] else ("🥈" if gr["silver_medal"] else ("🥉" if gr["bronze_medal"] else "无"))
            stats["above_median"] = gr["above_median"]
            stats["valid_submission"] = gr["valid_submission"]
            stats["is_lower_better_grade"] = gr["is_lower_better"]
        
        stats["log_info"] = log_info
        all_stats[matched_comp_id or comp_name] = stats
    
    # === 输出汇总表 ===
    print("\n## 1. 竞赛级汇总表\n")
    print(f"{'竞赛':<50} {'分数':>10} {'奖牌':>4} {'铜牌线':>10} {'差距':>10} {'节点数':>6} {'Draft':>6} {'Merge':>6} {'Mutate':>6} {'GA?':>4}")
    print("-" * 130)
    
    medal_count = {"🥇": 0, "🥈": 0, "🥉": 0, "无": 0}
    ga_triggered = 0
    total_nodes = 0
    total_draft = 0
    total_merge = 0
    total_mutate = 0
    
    for comp_id, stats in sorted(all_stats.items()):
        if "error" in stats:
            print(f"{comp_id:<50} {'N/A':>10} {'':>4}")
            continue
        
        score = stats.get("final_score")
        medal = stats.get("medal", "?")
        bronze = stats.get("bronze_threshold")
        lower = stats.get("is_lower_better_grade", False)
        
        if score is not None and bronze is not None:
            if lower:
                gap = score - bronze  # positive = worse for lower_is_better
            else:
                gap = bronze - score  # positive = below threshold
            gap_str = f"{gap:+.5f}"
        else:
            gap_str = "N/A"
        
        score_str = f"{score:.5f}" if score is not None else "null"
        bronze_str = f"{bronze:.5f}" if bronze is not None else "N/A"
        
        has_ga = "✅" if stats["has_merge"] or stats["has_mutate"] else "❌"
        if stats["has_merge"] or stats["has_mutate"]:
            ga_triggered += 1
        
        medal_count[medal] = medal_count.get(medal, 0) + 1
        total_nodes += stats["total_nodes"]
        total_draft += stats["draft_count"]
        total_merge += stats["merge_count"]
        total_mutate += stats["mutate_count"]
        
        print(f"{comp_id:<50} {score_str:>10} {medal:>4} {bronze_str:>10} {gap_str:>10} {stats['total_nodes']:>6} {stats['draft_count']:>6} {stats['merge_count']:>6} {stats['mutate_count']:>6} {has_ga:>4}")
    
    print("-" * 130)
    print(f"{'合计':<50} {'':>10} {'':>4} {'':>10} {'':>10} {total_nodes:>6} {total_draft:>6} {total_merge:>6} {total_mutate:>6} {f'{ga_triggered}/22':>4}")
    
    print(f"\n奖牌分布: 🥇={medal_count.get('🥇', 0)} 🥈={medal_count.get('🥈', 0)} 🥉={medal_count.get('🥉', 0)} 无={medal_count.get('无', 0)}")
    print(f"GA 触发率: {ga_triggered}/22 = {ga_triggered/22*100:.1f}%")
    print(f"总节点: {total_nodes}, Draft: {total_draft} ({total_draft/total_nodes*100:.1f}%), Merge: {total_merge} ({total_merge/total_nodes*100:.1f}%), Mutate: {total_mutate} ({total_mutate/total_nodes*100:.1f}%)")
    
    # === 过线但没拿奖牌的竞赛 ===
    print("\n\n## 2. 过线(above_median)但没拿奖牌的竞赛（重点分析对象）\n")
    near_miss = []
    for comp_id, stats in sorted(all_stats.items()):
        if "error" in stats:
            continue
        if stats.get("above_median") and stats.get("medal") == "无":
            score = stats.get("final_score")
            bronze = stats.get("bronze_threshold")
            lower = stats.get("is_lower_better_grade", False)
            if score is not None and bronze is not None:
                if lower:
                    gap = score - bronze
                else:
                    gap = bronze - score
                near_miss.append((comp_id, score, bronze, gap, lower, stats))
    
    near_miss.sort(key=lambda x: abs(x[3]))
    
    for comp_id, score, bronze, gap, lower, stats in near_miss:
        dir_indicator = "↓(lower better)" if lower else "↑(higher better)"
        print(f"\n### {comp_id}")
        print(f"  分数: {score:.5f} | 铜牌线: {bronze:.5f} | 差距: {abs(gap):.5f} {dir_indicator}")
        print(f"  节点: {stats['total_nodes']} (draft={stats['draft_count']}, merge={stats['merge_count']}, mutate={stats['mutate_count']})")
        
        # buggy rate
        for tt in ["draft", "merge", "mutate"]:
            bt = stats["buggy_by_type"].get(tt, {"total": 0, "buggy": 0})
            if bt["total"] > 0:
                rate = bt["buggy"] / bt["total"] * 100
                print(f"  {tt} buggy率: {bt['buggy']}/{bt['total']} = {rate:.1f}%")
        
        # 错误类型
        if stats["exc_types"]:
            print(f"  错误分布: {dict(stats['exc_types'].most_common(5))}")
        
        # 最佳metric
        if stats["best_metric"] is not None:
            print(f"  最佳内部metric: {stats['best_metric']:.6f} (节点: {stats['best_node_id']})")
        
        # 执行时间
        if stats["exec_times"]:
            avg_time = sum(stats["exec_times"]) / len(stats["exec_times"])
            max_time = max(stats["exec_times"])
            timeout_count = sum(1 for t in stats["exec_times"] if t > 3500)
            print(f"  执行时间: 平均={avg_time:.0f}s, 最大={max_time:.0f}s, 超时(>3500s)={timeout_count}个")
        
        # GA 信息
        if stats["log_info"]["phase2_triggered"]:
            print(f"  Phase 2: ✅ 已触发")
            for line in stats["log_info"]["phase2_lines"][:3]:
                print(f"    {line}")
        else:
            print(f"  Phase 2: ❌ 未触发")
        
        if stats["log_info"]["adaptive_timeout"]:
            print(f"  超时配置: {stats['log_info']['adaptive_timeout']}")
        
        # dead nodes
        if stats["dead_nodes"] > 0:
            print(f"  死节点: {stats['dead_nodes']}")
    
    # === 退步竞赛分析 ===
    print("\n\n## 3. Task Type 分布详情\n")
    for comp_id, stats in sorted(all_stats.items()):
        if "error" in stats:
            continue
        tt = dict(stats["task_types"])
        if not tt:
            continue
        medal = stats.get("medal", "?")
        ga = "GA" if (stats["has_merge"] or stats["has_mutate"]) else "Draft-only"
        print(f"  {comp_id:<50} {medal} | {ga:<12} | {tt}")
    
    # === 错误类型汇总 ===
    print("\n\n## 4. 全局错误类型分布\n")
    global_exc = Counter()
    for comp_id, stats in all_stats.items():
        if "error" in stats:
            continue
        global_exc.update(stats["exc_types"])
    
    total_errors = sum(global_exc.values())
    for exc, count in global_exc.most_common(10):
        print(f"  {exc:<40} {count:>5} ({count/total_errors*100:.1f}%)")
    print(f"  {'总错误数':<40} {total_errors:>5}")
    
    # === Metric 分布 ===
    print("\n\n## 5. 各竞赛 Metric 分布\n")
    for comp_id, stats in sorted(all_stats.items()):
        if "error" in stats or not stats["metrics"]:
            continue
        metrics = sorted(stats["metrics"])
        medal = stats.get("medal", "?")
        lower = stats.get("is_lower_better_grade", False)
        best = stats["best_metric"]
        bronze = stats.get("bronze_threshold")
        n_good = len(metrics)
        n_total = stats["total_nodes"]
        print(f"  {comp_id:<50} {medal} | good={n_good}/{n_total} | best={best:.5f} | bronze={bronze if bronze else 'N/A'} | {'↓' if lower else '↑'}")

    # === Phase 2 详情 ===
    print("\n\n## 6. Phase 2 GA 触发详情\n")
    for comp_id, stats in sorted(all_stats.items()):
        if "error" in stats:
            continue
        log = stats["log_info"]
        if log["phase2_triggered"] or stats["has_merge"] or stats["has_mutate"]:
            print(f"\n  {comp_id}:")
            print(f"    merge={stats['merge_count']}, mutate={stats['mutate_count']}")
            for line in log["phase2_lines"][:5]:
                print(f"    LOG: {line}")


if __name__ == "__main__":
    main()
