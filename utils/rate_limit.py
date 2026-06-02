"""
rate_limit.py
全局每日调用限流（保护作者自己的 API key 不被滥用）。

设计：
- 用一个 JSON 文件记录「日期 + 当日已用次数」，所有访客共享这一个计数器，
  从而对作者的 API 总开销形成上限（而不是按每个浏览器会话各算各的——那样太容易绕过）。
- 跨天自动归零。
- Streamlit Cloud 的文件系统是临时的：容器重启会清空计数（额度提前恢复），
  这对一个作品集 demo 完全可接受。真正的「硬上限」请同时在 Anthropic / Voyage
  控制台设置每月消费上限（见 README 部署说明）。
"""

import os
import json
from datetime import date
from typing import Tuple

LIMIT_FILE = "./rate_limit.json"


def _read() -> dict:
    today = date.today().isoformat()
    if os.path.exists(LIMIT_FILE):
        try:
            with open(LIMIT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("date") == today:
                return data
        except Exception:
            pass
    return {"date": today, "count": 0}


def get_count() -> int:
    """返回当日已用次数（只读，不自增）。"""
    return _read().get("count", 0)


def check_and_increment(daily_limit: int) -> Tuple[bool, int]:
    """
    尝试占用一次额度。

    Returns:
        (allowed, count):
            allowed - 是否还有额度（True 表示本次允许，并已 +1）
            count   - 本次操作后的当日累计次数
    """
    data = _read()
    if data["count"] >= daily_limit:
        return False, data["count"]

    data["count"] += 1
    try:
        with open(LIMIT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        # 写盘失败不阻断主流程（最多是少计一次数）
        pass
    return True, data["count"]
