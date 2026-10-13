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
from functools import cached_property
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
from app.temporal import resolve_reminder_time
from app.tool_policy import command_text, literal_fact, literal_note, literal_reminder, negated, withdrawn, vague_target


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
        elif isinstance(frame, LLMTextFrame):
            self._bytes += len(frame.text.encode("utf-8"))
            if self._bytes <= 16384:
                self._parts.append(frame.text)
            return
        elif isinstance(frame, LLMFullResponseEndFrame):
            if not self._tool_round and not self.turn.cancelled:
                content = self.turn.safe_response("回复超过长度限制，请查询实际状态。" if self._bytes > 16384 else "".join(self._parts))
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
    text: str = Field(min_length=1,max_length=500,description="Only the reminder content, e.g. 喝水; omit the delay/time and 提醒我 command")
    delay_seconds: int | None = Field(default=None,ge=1,le=31536000)
    due_at: str | None = Field(default=None,max_length=64,description="Explicit ISO8601 datetime including timezone")

    @model_validator(mode="after")
    def exactly_one_time(self):
        if (self.delay_seconds is None)==(self.due_at is None):
            raise ValueError("Provide exactly one of delay_seconds or due_at")
        return self


class CancelReminderArgs(Args):
    reminder_id: str = Field(min_length=1,max_length=128)


class CreateNoteArgs(Args):
    title: str = Field(min_length=1, max_length=120)
    body: str = Field(min_length=1, max_length=2048)


@dataclass
class ToolTurn:
    user_id: str
    user_text: str
    memory: object
    reminders: object
    records: object = None
    timezone_name: str = "Asia/Shanghai"
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    accepted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cancelled: bool = False
    calls: list = field(default_factory=list)
    memory_candidates: list | None = None
    time_override: object = None
    clarification_override: str | None = None
    target_bindings: dict = field(default_factory=dict)
    _executed: dict = field(default_factory=dict)
    _attempts: int = 0
    _gate: asyncio.Lock = field(default_factory=asyncio.Lock)

    @cached_property
    def resolved_time(self):
        return self.time_override or resolve_reminder_time(self.user_text, self.accepted_at, self.timezone_name)

    @cached_property
    def reminder_candidates(self):
        rows = self.reminders.list_reminders(self.user_id) if self.reminders else []
        clause = re.split(r"[，,。；;]", command_text(self.user_text))[0]
        matched = [r for r in rows if r["text"] and r["text"].rstrip("。") in clause]
        return matched or rows

    def known_memories(self):
        if self.memory_candidates is None:
            self.memory_candidates = self.memory.list_memories(self.user_id, limit=5) if self.memory else []
        return self.memory_candidates

    @property
    def clarification_prompt(self):
        if self.clarification_override: return self.clarification_override
        required = self.required_operation()
        if required == "create_reminder" and self.resolved_time.due is None:
            return "请告诉我具体的提醒时间，例如十分钟后或明天上午八点。"
        if required == "remember" and not literal_fact(self.user_text):
            return "你希望我记住什么信息？"
        if required in {"correct", "forget"}:
            rows = self.known_memories()
            if not rows or vague_target(self.user_text):
                return "你想更正哪条信息、改成什么内容？" if required == "correct" else "你想忘记哪条信息？请说出具体内容。"
        if required == "cancel_reminder":
            rows = self.reminder_candidates
            if not rows: return "当前没有找到可取消的提醒，请说明要取消的具体内容。"
            if len(rows) > 5 or (len(rows) > 1 and vague_target(self.user_text)):
                return "你想取消哪一条提醒？请说出具体内容。"
        return None

    @property
    def fixed_response(self):
        if withdrawn(self.user_text) or negated(self.user_text): return "好的，本轮不做修改。"
        return self.clarification_prompt

    def trusted_bindings(self, name):
        if self.target_bindings and name == self.required_operation(): return dict(self.target_bindings)
        if name == "create_reminder" and self.resolved_time.due is not None:
            bindings = self.resolved_time.arguments()
            if body := literal_reminder(self.user_text, self.resolved_time.phrase): bindings["text"] = body
            return bindings
        if name == "remember" and (content := literal_fact(self.user_text)): return {"content": content}
        if name == "create_note": return literal_note(self.user_text)
        return {}

    def safe_response(self, text):
        if self.fixed_response: return self.fixed_response
        if error := self.response_error(): return error
        required = self.required_operation()
        for call in reversed(self.calls):
            if call["tool"] == required and call["outcome"].get("ok"):
                if message := call["outcome"].get("result", {}).get("user_message"): return message
        # A model cannot turn a chat statement or a denied command into an
        # apparent mutation. Quoted explanations are not matched by this prefix.
        if re.match(r"^(?:(?:好的|好|没问题|收到|明白了)[，,:：。\s]*)?(?:我)?(?:已(?:经)?(?:记住|保存|删除|清除|取消|设置|创建|更正|更新)|提醒已设置)", text.strip()):
            if not any(c["outcome"].get("ok") and c["tool"] in {"remember","correct","forget","create_note","create_reminder","cancel_reminder"} for c in self.calls):
                return "明白了。"
        return text

    def _write_allowed(self, name):
        if name not in {"remember", "correct", "forget", "create_note", "create_reminder", "cancel_reminder"}:
            return True
        if withdrawn(self.user_text) or negated(self.user_text): return False
        text = command_text(self.user_text)
        if re.match(r"^(不用|不必|如何|怎么|解释|为什么|假如|如果|假设|你能解释|你知道|有人说|我忘记)", text): return False
        if re.match(r"^(不要|别)",text): return name=="cancel_reminder" and bool(re.match(r"^(不要|别)(再)?提醒", text))
        note_request = bool(re.match(r"^(?:(?:新建|创建|保存|记录|添加|写)(?:一[条篇份个])?.{0,6}笔记|记(?:一?[个条篇份])?笔记)", text))
        if name=="create_note": return note_request
        if note_request: return False
        if re.match(r"^(?:修改|更正|更新|删除|清除)(?:这条|那条|我的|该|所有|全部)?笔记", text): return False
        if name=="remember": return bool(re.match(r"^(记住|记下|保存)(?!了[吗么]|什么|了吗)",text))
        if name=="correct": return bool(re.match(r"^(?:更正|修改|更新|改为|改成|把.+?(?:更正|修改|更新|改为|改成))",text))
        if name=="forget": return bool(re.match(r"^(?:忘记|忘掉|删除|清除|把.+?(?:忘掉|删除|清除))",text)) and "提醒" not in text
        if name=="create_reminder":
            imperative = re.match(r"^(?:设置|设定|创建|新增|提醒我|今天|明天|后天|大后天|今晚|明早|在|过|等|[0-9零一二两三四五六七八九十百半几])", text)
            return bool(imperative) and "提醒" in text
        if name=="cancel_reminder": return "提醒" in text and bool(re.match(r"^(?:取消|删除|撤销|把.+?(?:取消|删除|撤销))",text))
        return True

    def cancel(self):
        # Already accepted durable writes may finish; old results must not
        # trigger another model response. No subsequent operation is admitted.
        self.cancelled = True

    def initial_tool_choice(self):
        if self.fixed_response: return "auto"
        # Use the framework/provider's structured tool mode for clear write
        # requests. Ordinary conversation remains automatic; lookup-based
        # changes may query IDs before applying the user's operation.
        for name in ("remember","create_reminder", "create_note"):
            if self._write_allowed(name):
                return "required"
        if any(self._write_allowed(name) for name in ("correct","forget","cancel_reminder")):
            return "required"
        return "required" if self.requested_lookup() else "auto"

    def requested_lookup(self):
        text = self.user_text
        if re.search(r"笔记(?!本)", text) and re.search(r"查询|查看|列出|读|我的笔记|笔记内容", text):
            return "list_notes"
        if "提醒" in text and re.search(r"查询|查看|列出|哪些|有什么", text):
            return "list_reminders"
        if re.search(r"记忆|记住", text) and re.search(r"查询|查看|列出|哪些|什么", text):
            return "list_memories"
        if re.search(r"现在几点|现在的?时间|今天.*日期|今天.*星期", text):
            return "get_time"
        return None

    def system_guidance(self):
        if self.fixed_response: return "本轮不能执行写操作。请只回答：" + self.fixed_response
        if self._write_allowed("create_note"):
            return "本轮需要保存本地笔记。调用 create_note。用户明确提供的标题和正文由应用原样传入，只填写schema中仍需的参数。成功后再确认。"
        if self._write_allowed("remember"):
            return "本轮需要保存用户明确要求记住的信息。调用 remember，提供稳定简短的key；完整内容由应用从用户原文传入，禁止自行摘要或省略。只有工具成功后才确认。"
        if self._write_allowed("create_reminder"):
            return "本轮是提醒设置任务。时间和明确事项由应用按用户原文传入。调用 create_reminder，只填写schema中仍需的参数，禁止猜测时间或改写原文。成功后简短确认。"
        if self._write_allowed("forget"):
            return "本轮必须执行遗忘。使用本轮证据中的记忆ID调用 forget，参数 memory_id；若不知道ID，先调用 list_memories，再调用 forget。输出 <tool_call> 开始的工具调用，只有 forget 成功后才确认。"
        if self._write_allowed("correct"):
            return "本轮必须更正已有记忆。使用本轮证据中的 memory_id 和 version 调用 correct，参数 memory_id、expected_version、content；若不知道ID和版本先调用 list_memories。只有 correct 成功后才确认。"
        if self._write_allowed("cancel_reminder"):
            rows = [{"id":r["id"],"text":r["text"],"due":r["due_utc"]} for r in self.reminder_candidates[:5]]
            return "本轮取消用户指定的提醒。以下是实际查询的候选，只能选择匹配的ID；若没有匹配项，选择NO_MATCH，不能猜测。" + json.dumps(rows,ensure_ascii=False)
        if lookup := self.requested_lookup():
            call = json.dumps({"name": lookup, "arguments": {}}, ensure_ascii=False)
            return f"本轮需要查询实际数据。先调用工具：<tool_call>\n{call}\n</tool_call>\n只有得到工具结果后才回答，历史偏好不能当成笔记或提醒。"
        return ""

    def required_operation(self):
        for name in ("remember", "correct", "forget", "create_reminder", "cancel_reminder", "create_note"):
            if self._write_allowed(name):
                return name
        return self.requested_lookup()

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
            message = f"现在是{now.hour}点{now.minute}分。"
            if re.search(r"日期|几号|几月|哪天", self.user_text): message = f"今天是{now.year}年{now.month}月{now.day}日。"
            if re.search(r"星期|周几", self.user_text): message = "今天是星期" + "一二三四五六日"[now.weekday()] + "。"
            return {"time":now.isoformat(),"timezone":self.timezone_name,
                    "weekday":now.weekday(), "user_message":message}
        if name=="get_device_status":
            from tools.project import meminfo
            return {"memory_bytes":meminfo(),"scope":"current host memory; model health is separate"}
        if name=="remember":
            result=self.memory.remember(self.user_id,"fact",args["key"],args["content"],source_id=self.turn_id,
                                        source_excerpt=self.user_text,request_id=request_id,
                                        source_date=self.accepted_at.astimezone(ZoneInfo(self.timezone_name)).date().isoformat(),source_speaker='user')
            if self.records:
                self.records.attach_memories(self.user_id, self.turn_id, [result["memory_id"]])
            return {**result,"user_message":"已记住。"}
        if name=="list_memories":
            items=self.memory.list_memories(self.user_id,limit=5)
            self.memory_candidates = items
            if self.records:
                items = self.records.filter_memories(self.user_id, items)
                self.records.attach_memories(self.user_id, self.turn_id, [x["memory_id"] for x in items])
            return {"items":[{"memory_id": x["memory_id"], "version": x["version"], "content": x["content"][:160]} for x in items]}
        if name=="correct":
            result=self.memory.correct(self.user_id,args["memory_id"],args["content"],expected_version=args["expected_version"],
                                       source_id=self.turn_id,source_excerpt=self.user_text,request_id=request_id,
                                       source_date=self.accepted_at.astimezone(ZoneInfo(self.timezone_name)).date().isoformat(),source_speaker='user')
            if self.records:
                self.records.purge_memory(self.user_id, args["memory_id"], deleted=False, exclude_turn=self.turn_id)
                self.records.attach_memories(self.user_id, self.turn_id, [args["memory_id"]])
            return {**result,"user_message":"已更正为："+args["content"]}
        if name=="forget":
            if self.records:
                self.records.attach_memories(self.user_id, self.turn_id, [args["memory_id"]])
                self.records.purge_memory(self.user_id, args["memory_id"])
            result=self.memory.forget(self.user_id,args["memory_id"],request_id=request_id)
            return {**result,"user_message":"已删除这条记忆及相关派生内容。"}
        if name=="create_note":
            if self.records is None: raise RuntimeError("Notes storage is unavailable")
            return self.records.create_note(self.user_id, **args, source_id=self.turn_id, request_id=request_id)
        if name=="list_notes":
            if self.records is None: raise RuntimeError("Notes storage is unavailable")
            rows = self.records.list_notes(self.user_id, limit=4)
            return {"items": [{"title": row["title"], "body": row["body"][:160] + ("…（内容节选）" if len(row["body"]) > 160 else ""),
                               "created_at": datetime.fromisoformat(row["created_at"]).astimezone(ZoneInfo(self.timezone_name)).isoformat()} for row in rows[:3]],
                    "display_scope": "最近三条笔记" if len(rows) > 3 else "当前笔记"}
        if name=="create_reminder":
            due=(self.accepted_at+timedelta(seconds=args["delay_seconds"])) if args["delay_seconds"] is not None else datetime.fromisoformat(args["due_at"])
            if due.tzinfo is None or due <= self.accepted_at: raise ValueError("Reminder needs an unambiguous future time with timezone")
            result=self.reminders.create(self.user_id,args["text"],due,request_id=request_id,timezone_name=self.timezone_name)
            local_due = due.astimezone(ZoneInfo(self.timezone_name))
            day = "今天" if local_due.date() == self.accepted_at.astimezone(ZoneInfo(self.timezone_name)).date() else f"{local_due.month}月{local_due.day}日"
            return {**result,"user_message":f"提醒已设置，会在{day}{local_due.hour}点{local_due.minute}分{local_due.second}秒提醒你。"}
        if name=="list_reminders":
            rows = self.reminders.list_reminders(self.user_id)
            return {"items":[{k:row[k] for k in ("id","text","due_utc","status")} for row in rows[:10]],
                    "total":len(rows), "display_scope":"最近十条以内"}
        if name=="cancel_reminder":
            result=self.reminders.cancel(self.user_id,args["reminder_id"],request_id=request_id)
            if result["status"] != "cancelled": raise ValueError("Reminder was not found in the selected profile")
            return {**result,"user_message":"提醒取消请求已处理。"}
        raise ValueError("Unregistered tool")

    async def execute(self, name, model, arguments, *, provider_arguments=None):
        async with self._gate:
            def record(answer):
                row = {"tool":name,"arguments":arguments,"outcome":answer}
                if provider_arguments is not None: row["provider_arguments"] = provider_arguments
                self.calls.append(row)
                del self.calls[:-100]
                return answer
            if self.cancelled: return record({"ok":False,"code":"CANCELLED"})
            self._attempts += 1
            try:
                arguments = {**arguments, **self.trusted_bindings(name)}
                try: args=model.model_validate(arguments).model_dump()
                except ValidationError:
                    if self._attempts > 3: return record({"ok":False,"code":"TOOL_LIMIT"})
                    raise
                digest=hashlib.sha256(json.dumps([name,args],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
                if digest in self._executed: return record(self._executed[digest])
                if self._attempts>3: return record({"ok":False,"code":"TOOL_LIMIT"})
                if len(self._executed)>=2: return record({"ok":False,"code":"TOOL_LIMIT"})
                if not self._write_allowed(name): return record({"ok":False,"code":"EXPLICIT_REQUEST_REQUIRED"})
                if name in {"remember","correct","forget","create_note","create_reminder","cancel_reminder"} and self.clarification_prompt:
                    return record({"ok":False,"code":"NEEDS_CLARIFICATION","question":self.clarification_prompt})
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
        if self.fixed_response: return ToolsSchema(standard_tools=[])
        related = {"correct": "list_memories", "forget": "list_memories", "cancel_reminder": "list_reminders"}
        allowed = {required} if required else {"get_time", "get_device_status", "list_memories", "list_reminders", "list_notes"}
        if self.records is None:
            allowed -= {"create_note", "list_notes"}
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
            ("create_reminder","设置提醒。时间由应用从用户原文解析，模型只填写提醒事项。",CreateReminderArgs),
            ("list_reminders","列出未完成提醒及ID。",Args),
            ("cancel_reminder","取消用户指定的提醒。",CancelReminderArgs),
            ("create_note", "保存用户明确要求创建的本地笔记。", CreateNoteArgs),
            ("list_notes", "查询当前用户已保存的本地笔记。", Args),
        ]
        schemas=[]
        for name,description,model in definitions:
            if names is not None and name not in names: continue
            bindings = self.trusted_bindings(name)
            def make_handler(tool_name,arg_model,bound):
                @tool_options(cancel_on_interruption=True,timeout_secs=10)
                async def handler(params: FunctionCallParams):
                    arguments = {**params.arguments, **bound}
                    if arguments.get("memory_id",arguments.get("reminder_id")) == "NO_MATCH":
                        self.clarification_override = "没有找到明确匹配的条目，请说出具体内容。"
                        result = {"ok":False,"code":"NEEDS_CLARIFICATION","question":self.clarification_override}
                        self.calls.append({"tool":tool_name,"arguments":arguments,"provider_arguments":params.arguments,"outcome":result})
                    else:
                        result=await self.execute(tool_name,arg_model,arguments,provider_arguments=params.arguments)
                    if result.get("ok") and tool_name in {"correct", "forget"}:
                        # Drop stale recalled text and old tool listings before
                        # Pipecat resumes generation with the real tool result.
                        messages = params.context.get_messages()
                        pending = []
                        for message in messages:
                            if message.get("role") == "assistant" and message.get("tool_calls"):
                                calls = [call for call in message["tool_calls"] if call.get("id") == params.tool_call_id]
                                if calls: pending.append({"role": "assistant", "tool_calls": calls})
                            elif message.get("role") == "tool" and message.get("tool_call_id") == params.tool_call_id:
                                pending.append(message)
                        params.context.set_messages([messages[0], {"role": "user", "content": "请根据本轮工具真实结果确认刚才的记忆维护操作。"}, *pending])
                    completed_write = result.get("ok") and tool_name in {"remember","correct","forget","create_note","create_reminder","cancel_reminder"}
                    params.context.set_tool_choice("auto")
                    if completed_write or self.clarification_prompt or self._attempts >= 3 or result.get("code")=="TOOL_LIMIT":
                        params.context.set_tools([])
                    elif tool_name in {"list_memories","list_reminders"}:
                        params.context.set_tools(self.schemas())
                    await params.result_callback(result,properties=FunctionCallResultProperties(run_llm=not self.cancelled))
                return handler
            schema=model.model_json_schema()
            properties = {key:value for key,value in schema.get("properties",{}).items() if key not in bindings}
            if name in {"correct","forget"} and "memory_id" in properties:
                properties["memory_id"]["enum"] = [row["memory_id"] for row in self.known_memories()] + ["NO_MATCH"]
            if name == "cancel_reminder" and "reminder_id" in properties:
                properties["reminder_id"]["enum"] = [row["id"] for row in self.reminder_candidates[:5]] + ["NO_MATCH"]
            schemas.append(FunctionSchema(name=name,description=description,properties=properties,
                                          required=[key for key in schema.get("required",[]) if key not in bindings],
                                          handler=make_handler(name,model,bindings)))
        return ToolsSchema(standard_tools=schemas)
