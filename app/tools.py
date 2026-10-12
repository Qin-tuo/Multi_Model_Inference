"""Finite tool handlers registered through Pipecat's FunctionSchema API.

Each ToolTurn binds one trusted user transcript and profile. It does not own a
conversation loop. Durable state remains in Mem0 and APScheduler plugins.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
import uuid
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (CancelFrame, FunctionCallResultProperties,
    FunctionCallsStartedFrame, InterruptionFrame, LLMFullResponseEndFrame,
    LLMFullResponseStartFrame, LLMTextFrame)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import FunctionCallParams

from app.memory import MemoryAdapterError


class ToolResponseGuard(FrameProcessor):
    """Buffer a bounded write confirmation until durable tool outcomes exist.

    Pipecat still owns all tool execution and continuation. This output policy
    prevents an unexecuted write from being announced or handed to TTS.
    """

    def __init__(self, turn):
        super().__init__()
        self.turn = turn
        self._parts = []
        self._bytes = 0
        self._tool_round = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, (CancelFrame, InterruptionFrame)):
            self.turn.cancel()
            self._parts.clear()
            self._bytes = 0
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._parts.clear()
            self._bytes = 0
            self._tool_round = False
        elif isinstance(frame, FunctionCallsStartedFrame):
            self._tool_round = True
        elif isinstance(frame, LLMTextFrame) and self.turn.required_operation():
            self._bytes += len(frame.text.encode("utf-8"))
            if self._bytes <= 16384:
                self._parts.append(frame.text)
            return
        elif isinstance(frame, LLMFullResponseEndFrame) and self.turn.required_operation():
            if not self._tool_round and not self.turn.cancelled:
                error = self.turn.response_error()
                content = error or ("回复超过长度限制，请查询实际状态。" if self._bytes > 16384 else "".join(self._parts))
                if content:
                    await self.push_frame(LLMTextFrame(content))
            self._parts.clear()
            self._bytes = 0
        await self.push_frame(frame, direction)


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RememberArgs(Args):
    key: str = Field(min_length=1,max_length=128,description="Stable short fact key")
    content: str = Field(min_length=1,max_length=2048,description="User's explicitly stated fact")


class CorrectArgs(Args):
    memory_id: str = Field(min_length=1,max_length=128)
    expected_version: int = Field(ge=1)
    content: str = Field(min_length=1,max_length=2048)


class ForgetArgs(Args):
    memory_id: str = Field(min_length=1,max_length=128)


class CreateReminderArgs(Args):
    text: str = Field(min_length=1,max_length=500)
    delay_seconds: int | None = Field(default=None,ge=1,le=31536000)
    due_at: str | None = Field(default=None,max_length=64,description="Explicit ISO8601 datetime including timezone")

    @model_validator(mode="after")
    def exactly_one_time(self):
        if (self.delay_seconds is None)==(self.due_at is None):
            raise ValueError("Provide exactly one of delay_seconds or due_at")
        return self


class CancelReminderArgs(Args):
    reminder_id: str = Field(min_length=1,max_length=128)


@dataclass
class ToolTurn:
    user_id: str
    user_text: str
    memory: object
    reminders: object
    timezone_name: str = "Asia/Shanghai"
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    accepted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cancelled: bool = False
    calls: list = field(default_factory=list)
    _executed: dict = field(default_factory=dict)
    _attempts: int = 0
    _gate: asyncio.Lock = field(default_factory=asyncio.Lock)

    def _write_allowed(self, name):
        text = self.user_text.strip()
        if re.match(r"^(不要|别|不用|不必)",text): return name=="cancel_reminder" and "提醒" in text
        if name=="remember": return bool(re.match(r"^(?:(?:请|麻烦)你?(?:帮我)?|帮我)?(记住|记下|保存)(?!了[吗么]|什么|了吗)",text))
        if name=="correct": return bool(re.search(r"更正|修改|改为|改成|更新|以后.*(详细|简短)",text))
        if name=="forget": return bool(re.search(r"忘记|删除|清除",text))
        if name=="create_reminder": return "提醒" in text and "取消" not in text and bool(re.search(r"秒|分钟|小时|明天|后天|今天|点|设置",text))
        if name=="cancel_reminder": return "提醒" in text and bool(re.search(r"取消|删除|撤销|别|不要",text))
        return True

    def cancel(self):
        # Already accepted durable writes may finish; old results must not
        # trigger another model response. No subsequent operation is admitted.
        self.cancelled = True

    def initial_tool_choice(self):
        # Use the framework/provider's structured tool mode for clear write
        # requests. Ordinary conversation remains automatic; lookup-based
        # changes may query IDs before applying the user's operation.
        for name in ("remember","create_reminder"):
            if self._write_allowed(name):
                return "required"
        if any(self._write_allowed(name) for name in ("correct","forget","cancel_reminder")):
            return "required"
        return "auto"

    def system_guidance(self):
        if self._write_allowed("remember"):
            return "本轮需要保存用户明确要求记住的事实。直接输出 <tool_call> 开始的 remember 工具调用，arguments 包括 key 和 content；不要先口头确认。例如用户说‘记住我喜欢喝茶’，调用 remember，key 为 drink_preference，content 为 用户喜欢喝茶。只有本轮工具返回成功后才简短确认。"
        if self._write_allowed("create_reminder"):
            return "本轮是提醒设置任务。先调用 create_reminder 工具；相对时间换算为秒。已有本轮成功的tool结果后，再简短确认。"
        return ""

    def required_operation(self):
        for name in ("remember", "correct", "forget", "create_reminder", "cancel_reminder"):
            if self._write_allowed(name):
                return name
        return None

    def response_error(self):
        """Trusted completion gate: model prose alone cannot confirm a write."""
        required = self.required_operation()
        if self.cancelled:
            return "本轮已停止，请查询实际状态后再继续。"
        if required and not any(call["tool"] == required and call["outcome"].get("ok") for call in self.calls):
            return "本轮没有获得该操作成功的结果，无法确认已完成。请重试或查询实际状态。"
        return None

    def _invoke(self, name, args, request_id):
        if name=="get_time":
            now=datetime.now(ZoneInfo(self.timezone_name))
            return {"time":now.isoformat(),"timezone":self.timezone_name,
                    "user_message":"现在是"+now.strftime("%Y年%m月%d日 %H:%M:%S")+"。"}
        if name=="get_device_status":
            from tools.project import meminfo
            return {"memory_bytes":meminfo(),"scope":"current host memory; model health is separate"}
        if name=="remember":
            result=self.memory.remember(self.user_id,"fact",args["key"],args["content"],source_id=self.turn_id,
                                        source_excerpt=self.user_text,request_id=request_id)
            return {**result,"user_message":"已记住："+args["content"]}
        if name=="list_memories":
            return {"items":self.memory.list_memories(self.user_id,limit=10)}
        if name=="correct":
            result=self.memory.correct(self.user_id,args["memory_id"],args["content"],expected_version=args["expected_version"],
                                       source_id=self.turn_id,source_excerpt=self.user_text,request_id=request_id)
            return {**result,"user_message":"已更正为："+args["content"]}
        if name=="forget":
            result=self.memory.forget(self.user_id,args["memory_id"],request_id=request_id)
            return {**result,"user_message":"已删除这条记忆及相关派生内容。"}
        if name=="create_reminder":
            due=(self.accepted_at+timedelta(seconds=args["delay_seconds"])) if args["delay_seconds"] is not None else datetime.fromisoformat(args["due_at"])
            if due.tzinfo is None or due <= self.accepted_at: raise ValueError("Reminder needs an unambiguous future time with timezone")
            result=self.reminders.create(self.user_id,args["text"],due,request_id=request_id,timezone_name=self.timezone_name)
            return {**result,"user_message":"已设置提醒："+args["text"]+"，时间为"+due.astimezone(ZoneInfo(self.timezone_name)).isoformat()}
        if name=="list_reminders": return {"items":self.reminders.list_reminders(self.user_id)}
        if name=="cancel_reminder":
            result=self.reminders.cancel(self.user_id,args["reminder_id"],request_id=request_id)
            return {**result,"user_message":"提醒取消请求已处理。"}
        raise ValueError("Unregistered tool")

    async def execute(self, name, model, arguments):
        async with self._gate:
            def record(answer):
                self.calls.append({"tool":name,"arguments":arguments,"outcome":answer})
                return answer
            if self.cancelled: return record({"ok":False,"code":"CANCELLED"})
            self._attempts += 1
            if self._attempts>3: return record({"ok":False,"code":"TOOL_LIMIT"})
            try:
                args=model.model_validate(arguments).model_dump()
                digest=hashlib.sha256(json.dumps([name,args],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
                if digest in self._executed: return record(self._executed[digest])
                if len(self._executed)>=2: return record({"ok":False,"code":"TOOL_LIMIT"})
                if not self._write_allowed(name): return record({"ok":False,"code":"EXPLICIT_REQUEST_REQUIRED"})
                # The caller has admitted this operation before cancellation.
                # Shield its actual result so cancellation never reports a
                # fictitious rollback of a durable SDK transaction.
                work=asyncio.create_task(asyncio.to_thread(self._invoke,name,args,self.turn_id+":"+digest))
                try:
                    result=await asyncio.shield(work)
                except asyncio.CancelledError:
                    self.cancelled=True
                    result=await work
                answer={"ok":True,"result":result}
                self._executed[digest]=answer
            except (ValidationError,ValueError,MemoryAdapterError,RuntimeError) as error:
                answer={"ok":False,"code":getattr(error,"code","INVALID_ARGUMENT"),"message":str(error)[:500]}
            except Exception as error:
                answer={"ok":False,"code":"OPERATION_FAILED","message":type(error).__name__}
            return record(answer)

    def schemas(self, *, names=None):
        # Advertise only operations admitted for this trusted user turn. The
        # small baseline model also has less irrelevant schema text to attend to.
        required = self.required_operation()
        related = {"correct": "list_memories", "forget": "list_memories", "cancel_reminder": "list_reminders"}
        allowed = {required} if required else {"get_time", "get_device_status", "list_memories", "list_reminders"}
        if required in related:
            allowed.add(related[required])
        names = allowed if names is None else allowed.intersection(names)
        definitions=[
            ("get_time","查询真实日期时间，禁止猜测。",Args),
            ("get_device_status","读取本机资源状态。",Args),
            ("remember","用户明确要求记住时保存事实。",RememberArgs),
            ("list_memories","列出记忆ID、内容和版本。",Args),
            ("correct","更正指定记忆，版本须来自查询结果。",CorrectArgs),
            ("forget","用户明确要求遗忘时删除指定记忆。",ForgetArgs),
            ("create_reminder","设置提醒，相对秒数或明确ISO时间二选一。",CreateReminderArgs),
            ("list_reminders","列出未完成提醒及ID。",Args),
            ("cancel_reminder","取消用户指定的提醒。",CancelReminderArgs),
        ]
        schemas=[]
        for name,description,model in definitions:
            if names is not None and name not in names: continue
            def make_handler(tool_name,arg_model):
                @tool_options(cancel_on_interruption=True,timeout_secs=10)
                async def handler(params: FunctionCallParams):
                    result=await self.execute(tool_name,arg_model,params.arguments)
                    params.context.set_tool_choice("auto")
                    if result.get("code")=="TOOL_LIMIT": params.context.set_tools([])
                    await params.result_callback(result,properties=FunctionCallResultProperties(run_llm=not self.cancelled))
                return handler
            schema=model.model_json_schema()
            schemas.append(FunctionSchema(name=name,description=description,properties=schema.get("properties",{}),
                                          required=schema.get("required",[]),handler=make_handler(name,model)))
        return ToolsSchema(standard_tools=schemas)
