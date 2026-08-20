#!/bin/bash
# 每日跑一次完整流水线，失败则由 launchd 在后续时点重试。
#
# launchd 在 12:00 / 13:00 / 14:00 / 15:00 / 16:00 各触发一次（共 5 次机会）。
# 重试不是脚本里 sleep 一小时，而是让 launchd 重新触发：这样机器中途睡了
# 下一个时点还能接上，也不用让一个进程举着锁挂五小时。
#
# 每次触发都先判断今天是不是已经成功过了 —— 成功过就立刻退出，
# 所以正常情况下 13:00 之后那四次几乎不花时间。
#
# launchd 给的环境几乎是空的（没有 PATH、没有 locale、不继承你 shell 里的任何东西），
# 所以这里所有路径都写死。API key 不在这里读 —— pna 自己从项目根的 .env 加载。

set -uo pipefail

ROOT="/Users/ray/Code/paper-news-agent"
PNA="$ROOT/.venv/bin/pna"
PY="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/logs"
STATE="$ROOT/.state"
LOCK="$ROOT/.daily.lock"
MAX_ATTEMPTS=5

mkdir -p "$LOG_DIR" "$STATE"
LOG="$LOG_DIR/$(date +%Y-%m-%d).log"

# 一轮重试的身份用**本地**日期，它在 00:00-23:59 之间是稳定的。
# 不能用 UTC 日期：17:00 PDT 已经是 UTC 的第二天，那样最后几次重试会
# 被当成新的一轮，把计数清零。
CYCLE="$(date +%Y-%m-%d)"
STAMP="$STATE/success-$CYCLE"
TRIES="$STATE/attempts-$CYCLE"
TARGET="$STATE/target-$CYCLE"

[ -f "$STAMP" ] && exit 0   # 今天已经成功过

# mkdir 是原子的，所以它同时是"上锁"和"检查有没有人在跑"。
# 需要这道锁是因为一次 run 可能跑 20 分钟，跨过下一个触发时点。
if ! mkdir "$LOCK" 2>/dev/null; then
    # 强制关机会留下锁目录，那样之后每次触发都会静默跳过，永远不再更新。
    # 超过 90 分钟的锁一定是死的 —— 最长的一次真实运行是 17 分钟。
    if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +90 2>/dev/null)" ]; then
        echo "[$(date '+%F %T')] 清理残留的锁（>90 分钟）" >> "$LOG"
        rmdir "$LOCK" 2>/dev/null && mkdir "$LOCK" 2>/dev/null || exit 0
    else
        echo "[$(date '+%F %T')] 上一次还在跑，这次跳过" >> "$LOG"
        exit 0
    fi
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

{
    # 目标日期只在本轮第一次触发时算，之后固定复用。
    # 直接问 pna 而不是在 shell 里重算工作日逻辑 —— 两处实现同一规则迟早会漂。
    if [ ! -s "$TARGET" ]; then
        # 传本地日期而不是让它用 UTC：冬天 16:00 PST 就是 UTC 次日 00:00，
        # 那时 UTC 口径会指向一个 17:00 PST 才公告的日子。
        "$PY" -c "import datetime as dt; from pna.cli import resolve_date; print(resolve_date('auto', dt.date.today()))" > "$TARGET"
    fi
    DATE="$(cat "$TARGET")"

    n=$(( $(cat "$TRIES" 2>/dev/null || echo 0) + 1 ))
    echo "$n" > "$TRIES"

    echo "===================================================================="
    echo "[$(date '+%F %T')] 开始 第 $n/$MAX_ATTEMPTS 次尝试，目标日期 $DATE"

    # 等网络就绪：唤醒后 launchd 可能比 Wi-Fi 先起来，
    # 那样 OAI 抓取会直接失败，而这本来只需要多等几秒。
    for i in $(seq 1 30); do
        if /sbin/ping -c1 -t2 export.arxiv.org >/dev/null 2>&1; then break; fi
        [ "$i" -eq 30 ] && { echo "[$(date '+%F %T')] 等了 60s 还是没网，放弃本次"; exit 1; }
        sleep 2
    done

    # caffeinate -i 挡住空闲睡眠。曾有一次日志是 08:30 开始、17:19 才推送 ——
    # 机器中途睡了，进程被挂起九小时。-i 只挡空闲睡眠，合盖照样睡，别指望它。
    /usr/bin/caffeinate -i "$PNA" run --date "$DATE"
    code=$?

    if [ $code -eq 0 ]; then
        touch "$STAMP"

        # 推上去，否则 GitHub Pages 永远停在上一次。
        cd "$ROOT" || exit 1
        # 逐个 add。`git add a b c` 只要有一个路径不存在就整条 fatal 并且
        # **什么都不暂存**，接着脚本会打印"没有变化，不推送" —— 页面改好了却推不上去，
        # 而且看不出哪里错了。feedback.jsonl 只要哪天被删掉就会踩到。
        for p in docs data/feedback.jsonl data/runs; do
            [ -e "$p" ] && git add "$p"
        done
        if git diff --staged --quiet; then
            echo "[$(date '+%F %T')] 没有变化，不推送"
        else
            git -c user.name="paper-news-agent" \
                -c user.email="actions@users.noreply.github.com" \
                commit -q -m "digest: $DATE"
            # push 失败不算整体失败 —— 论文已经抓到本地了。
            git push -q origin main 2>&1 || echo "[$(date '+%F %T')] push 失败，本地已提交"
        fi

        [ "$n" -gt 1 ] && echo "[$(date '+%F %T')] 第 $n 次重试成功"
    else
        # 中间几次失败不打扰你 —— 瞬时网络问题下一个钟头自己就好了，
        # 每次都弹窗只会训练你忽略它。只在机会用完时说一声。
        if [ "$n" -ge "$MAX_ATTEMPTS" ]; then
            /usr/bin/osascript -e "display notification \"$DATE 连续 $n 次失败，见 logs/$CYCLE.log\" with title \"arXiv digest 失败\"" 2>/dev/null || true
            echo "[$(date '+%F %T')] 已用完 $MAX_ATTEMPTS 次机会，今天放弃"
        else
            echo "[$(date '+%F %T')] 第 $n 次失败，下个钟头再试"
        fi
    fi

    echo "[$(date '+%F %T')] 结束，退出码 $code"

    # 日志和状态戳跟着页面快照一起滚动，别让它们无限长。
    find "$LOG_DIR" -name '*.log' -type f -mtime +14 -delete
    find "$STATE" -type f -mtime +14 -delete
    exit $code
} >> "$LOG" 2>&1
