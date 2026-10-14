"""Trusted text boundaries for finite tool authorization and literal fields.

This policy does not run conversations or replace the model. It distinguishes
commands from quotations/withdrawals and preserves explicit payloads verbatim.
"""
import re


def command_text(text):
    text = text.strip()
    for _ in range(3):
        reduced = re.sub(r"^(?:(?:请|麻烦)你?(?:帮我)?|帮我|我想(?:请你)?|我要)[，,\s]*", "", text)
        if reduced == text: break
        text = reduced
    return text


def unquoted(text):
    return re.sub(r'“[^”]*”|‘[^’]*’|"[^"]*"|\x27[^\x27]*\x27', "〈引用〉", text)


def withdrawn(text):
    return bool(re.search(r"(?:^|[。；;，,])\s*(?:算了|取消(?:刚才|之前|这次|本次|这个|上述)(?:的)?(?:请求|操作|设置)|(?:不用|不要|别|不必)(?:再)?(?:记|保存|设置|创建|删除|取消)(?:了|啦))", unquoted(text)))


def negated(text):
    plain = command_text(unquoted(text))
    if re.match(r"^(不要|别)(?:再)?提醒", plain): return False  # Explicit cancellation.
    if re.match(r"^(?:不用|不必|无需|不需要|不要|别)", plain): return True
    prefix = plain.split("提醒", 1)[0]
    return "提醒" in plain and bool(re.search(r"不用|不必|无需|不需要|不要|别", prefix))


def memory_listing_request(text):
    """Recognize an inventory request,not discussion of human memory.

    Specific historical/factual questions already use semantic retrieval.
    Enumeration supplies arbitrary stored IDs and is appropriate only for a
    requested inventory or as a prerequisite of correction/forgetting.
    """
    if withdrawn(text) or negated(text):return False
    plain=re.sub(r'\s+','',command_text(unquoted(text))).rstrip('。？！?!')
    inventory=r'(?:(?:我的|你(?:为我)?(?:保存|存储|记住)的|已(?:经)?(?:保存|存储|记住)的|全部|所有|当前|现有))*记忆(?:列表|清单)?'
    if re.fullmatch(r'(?:查询|查看|列出|展示|显示|读出|回顾)(?:一下)?'+inventory,plain):return True
    if re.fullmatch(inventory+r'(?:里有什么|有哪些|是什么|包含什么)(?:内容|信息)?(?:吗|呢)?',plain):return True
    subject=r'(?:(?:查询|查看|告诉我)(?:一下)?)?(?:你)?(?:还|已经|都|曾经)?'
    remembered=r'(?:记住|记得|保存|存储)(?:了)?(?:什么|哪些)(?:关于我的)?(?:内容|信息|事情|事实)?'
    if re.fullmatch(subject+remembered+r'(?:吗|呢)?',plain):return True
    return re.fullmatch(r'我(?:有|还有)哪些(?:已(?:经)?(?:保存|存储|记住)的)?记忆(?:吗|呢)?',plain) is not None


def literal_fact(text):
    value = command_text(text)
    if re.fullmatch(r"(?:记住|记下|保存)(?:一下|这条偏好|这条信息|这个事实|这句话|这段话)?[，,:：。\s]*", value): return None
    match = re.match(r"^(?:记住|记下|保存)(?:一下|这条偏好|这条信息|这个事实|这句话|这段话|以下内容)?[，,:：。\s]*(.+)$", value, re.S)
    return match.group(1).strip() if match else None


def literal_note(text):
    result = {}
    title = re.search(r"标题(?:是|为|[:：])\s*(.+?)(?=[，,。；;]?\s*内容(?:是|为|[:：])|$)", text, re.S)
    body = re.search(r"内容(?:是|为|[:：])\s*(.+)$", text, re.S)
    if title: result["title"] = title.group(1).strip(" ，,。；;：:")
    if body:
        value = body.group(1).strip().rstrip("。")
        if len(value) >= 2 and (value[0],value[-1]) in {("“","”"),("‘","’"),("\"","\"")}:
            value = value[1:-1]
        result["body"] = value
    return result


def literal_reminder(text, time_phrase=""):
    # Clarification additions are control metadata, not part of the task text.
    text = text.split("，补充时间：", 1)[0]
    explicit = re.search(r"内容(?:是|为|[:：])\s*(.+)$", text, re.S)
    if explicit:
        value = explicit.group(1)
    else:
        after = re.search(r"提醒(?:我们|我)(?:一下)?\s*(.+)$", text, re.S)
        if after:
            value = after.group(1)
            if time_phrase and value.startswith(time_phrase): value = value[len(time_phrase):]
        else:
            named = re.match(r"^(?:设置|设定|创建|新增)(?:一[个条])?(.+?)(?:的)?提醒[。！!\s]*$", command_text(text))
            if not named: return None
            value = named.group(1)
            if time_phrase and time_phrase in value: return None
    value = value.strip(" ，,。；;：:!！\n")
    if len(value) >= 2 and (value[0],value[-1]) in {("“","”"),("‘","’"),("\"","\"")}:
        value = value[1:-1]
    return value or None


def vague_target(text):
    plain = command_text(text)
    plain = re.sub(r"^(?:请)?(?:把)?(?:忘记|忘掉|删除|清除|取消|撤销)?", "", plain)
    return bool(re.fullmatch(r"(?:它|那|这)(?:个|条|件)?(?:事|记忆|提醒|信息)?(?:更正|修改|更新)?(?:一下|掉)?[。！？?！\s]*", plain))
