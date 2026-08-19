#!/bin/bash
# 每日跑一次完整流水线。由 launchd 调用，也可以手动执行。
#
# launchd 给的环境几乎是空的（没有 PATH、没有 locale、不继承你 shell 里的任何东西），
# 所以这里所有路径都写死，不依赖 `cd` 之外的任何上下文。
# API key 不在这里读 —— pna 自己从项目根的 .env 加载，跟工作目录无关。

set -uo pipefail

ROOT="/Users/ray/Code/paper-news-agent"
PNA="$ROOT/.venv/bin/pna"
LOG_DIR="$ROOT/logs"
LOCK="$ROOT/.daily.lock"

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$(date +%Y-%m-%d).log"

# mkdir 是原子的，所以它同时是"上锁"和"检查有没有人在跑"。
# 需要这道锁是因为一次 run 可能跑 10 分钟以上，而睡眠唤醒后 launchd 会补跑漏掉的定时，
# 两个实例同时写同一个 JSONL 会互相覆盖。
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "[$(date '+%F %T')] 上一次还在跑（$LOCK 存在），这次跳过" >> "$LOG"
    exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

{
    echo "===================================================================="
    echo "[$(date '+%F %T')] 开始"

    # 等网络就绪：唤醒后 launchd 可能比 Wi-Fi 先起来，
    # 那样 OAI 抓取会直接失败，而这本来只需要多等几秒。
    for i in $(seq 1 30); do
        if /sbin/ping -c1 -t2 export.arxiv.org >/dev/null 2>&1; then break; fi
        [ "$i" -eq 30 ] && { echo "[$(date '+%F %T')] 等了 60s 还是没网，放弃"; exit 1; }
        sleep 2
    done

    "$PNA" run --date auto
    code=$?

    # 推上去，否则 GitHub Pages 永远停在上一次。
    # 只在流水线成功时推：失败那天的页面可能是半成品或空的，宁可线上留着昨天的。
    if [ $code -eq 0 ]; then
        cd "$ROOT" || exit 1
        git add docs data/feedback.jsonl data/runs
        if git diff --staged --quiet; then
            echo "[$(date '+%F %T')] 没有变化，不推送"
        else
            git -c user.name="paper-news-agent" \
                -c user.email="actions@users.noreply.github.com" \
                commit -q -m "digest: $(date +%F)"
            # push 失败不算整体失败 —— 论文已经抓到本地了，网络问题明天会自己补上。
            git push -q origin main 2>&1 || echo "[$(date '+%F %T')] push 失败，本地已提交"
        fi
    fi

    echo "[$(date '+%F %T')] 结束，退出码 $code"

    # 日志跟着页面快照一起滚动，别让它无限长。
    find "$LOG_DIR" -name '*.log' -type f -mtime +14 -delete
    exit $code
} >> "$LOG" 2>&1
