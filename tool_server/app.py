"""AgentEval Mock 工具服务（FastAPI 入口）。

评测时编排器启动本服务，各 Agent 框架通过 HTTP 调用工具完成任务，每个
评测实例（instance_id）拥有独立状态空间与调用日志。

HTTP 接口契约：
- GET  /healthz                        -> {"ok": true}
- POST /instances/{id}/reset           body: {"domain", "initial_state"?}
- POST /instances/{id}/tools/{name}    body: 工具参数 dict
- GET  /instances/{id}/state           -> 完整状态 dict
- GET  /instances/{id}/log             -> 日志数组

返回格式：业务成功 {"ok": true, "result": ...}；业务拒绝（HTTP 200）
{"ok": false, "error": "<中文原因>", "code": "<错误码>"}；
instance/tool 不存在 -> HTTP 404；body 非法/缺参 -> HTTP 422。
"""
from __future__ import annotations

import argparse
import copy
import threading
from typing import Any, Literal

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .registry import required_params
from .state import InstanceState, ToolError
from .tools import DOMAIN_TOOLS

app = FastAPI(
    title="AgentEval Mock Tool Server",
    description="多框架 Agent 评测体系的有状态 mock 工具服务",
    version="1.0",
)

# 进程内实例注册表与互斥锁（并发安全：实例字典与单实例状态读写统一加锁）
_INSTANCES: dict[str, InstanceState] = {}
_LOCK = threading.RLock()


class ResetRequest(BaseModel):
    """POST /reset 请求体：domain 限定三域，initial_state 可缺省为空。"""

    model_config = ConfigDict(extra="ignore")

    domain: Literal["travel", "shop", "analytics"]
    initial_state: dict[str, Any] = Field(default_factory=dict)


def _get_instance(instance_id: str) -> InstanceState:
    """取实例，不存在抛 404（供 tool/state/log 三个端点共用）。"""
    instance = _INSTANCES.get(instance_id)
    if instance is None:
        raise HTTPException(status_code=404, detail=f"实例 {instance_id} 不存在")
    return instance


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    """健康检查。"""
    return {"ok": True}


@app.post("/instances/{instance_id}/reset")
def reset_instance(instance_id: str, body: ResetRequest) -> dict[str, Any]:
    """重置（或首次初始化）一个实例：建种子 -> initial_state 深合并 -> 置 session。"""
    with _LOCK:
        instance = _INSTANCES.get(instance_id)
        if instance is None:
            instance = InstanceState()
            _INSTANCES[instance_id] = instance
        instance.reset(body.domain, body.initial_state)
    return {
        "ok": True,
        "result": {"instance_id": instance_id, "domain": body.domain},
    }


@app.post("/instances/{instance_id}/tools/{name}")
def call_tool(
    request: Request,
    instance_id: str,
    name: str,
    args: dict[str, Any] = Body(..., description="工具参数对象"),
) -> Any:
    """执行一次工具调用：校验必填参数 -> 执行业务 -> 写日志。

    业务拒绝（ToolError）转为 HTTP 200 的 {"ok": false, ...}；
    工具不存在/实例不存在 -> 404；参数缺失/body 非法 -> 422。
    """
    caller = request.headers.get("X-Caller", "agent")
    with _LOCK:
        instance = _get_instance(instance_id)
        handler = DOMAIN_TOOLS.get(instance.domain, {}).get(name)
        if handler is None:
            raise HTTPException(
                status_code=404,
                detail=f"域 {instance.domain} 中不存在工具 {name}",
            )
        missing = [p for p in required_params(instance.domain, name) if p not in args]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"缺少必需参数: {', '.join(missing)}",
            )
        try:
            result = handler(instance, args)
        except ToolError as exc:
            instance.record_log(
                name, args, ok=False, error=exc.message, code=exc.code, caller=caller
            )
            return JSONResponse(
                status_code=200,
                content={"ok": False, "error": exc.message, "code": exc.code},
            )
        instance.record_log(name, args, ok=True, result=result, caller=caller)
    return {"ok": True, "result": result}

@app.get("/instances/{instance_id}/state")
def read_state(instance_id: str) -> Any:
    """读取实例完整状态（verifier 通过此接口做终态检查），深拷贝返回。"""
    with _LOCK:
        instance = _get_instance(instance_id)
        return copy.deepcopy(instance.state)


@app.get("/instances/{instance_id}/log")
def read_log(instance_id: str) -> list[dict[str, Any]]:
    """读取实例工具调用日志数组。"""
    with _LOCK:
        instance = _get_instance(instance_id)
        return copy.deepcopy(instance.log)


def main() -> None:
    """命令行入口：python -m tool_server.app [--host ..] [--port ..]。"""
    parser = argparse.ArgumentParser(description="启动 AgentEval mock 工具服务")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8200, help="监听端口")
    args = parser.parse_args()
    # 仅作为脚本执行时启动服务器；TestClient 导入 app 不受影响
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
