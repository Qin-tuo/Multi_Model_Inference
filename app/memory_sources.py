"""Explicit provenance and narrow calendar constraints for dialogue recall.

Source dates describe when a message was said, not when every event mentioned
in that message occurred. Missing years stay unspecified; no year is invented.
"""
from datetime import date,datetime
import re

SPEAKERS={'user','assistant','tool','observation'}
DATE_KEYS=('source_date','source_month_day','source_speaker')

_NUMBER=r'(?:[-负]?(?:[0-9]+(?:\.[0-9]+)?|[零〇一二两三四五六七八九十百千万亿]+(?:点[零〇一二三四五六七八九]+)?))'
_OPERATOR=r'(?:加上?|减去?|乘以?|除以|[+＋\-−×*÷/])'
_LITERAL_ARITHMETIC=re.compile(r'(?:(?:请问|请计算|请算一下|算一下|计算|帮我算一下|帮我计算)[，,]?)?'
    +_NUMBER+r'(?:'+_OPERATOR+_NUMBER+r')+'
    +r'(?:等于(?:几|多少)|(?:结果)?是多少|得(?:几|多少)|=)?[？?。.!！]?')


def literal_arithmetic(text):
    """Recognize a complete numeric expression,not a calculator or intent LLM.

    Units,variables,personal facts,anaphora,quoted text and extra instructions
    deliberately fall back to ordinary retrieval. Active dialogue stays intact.
    """
    if not isinstance(text,str) or len(text)>256:return False
    compact=re.sub(r'\s+','',text)
    if re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|1\d{2}-\d{4}-\d{4}|0\d{2,3}-\d{7,8}',compact):return False
    return _LITERAL_ARITHMETIC.fullmatch(compact) is not None


def provenance(source_date=None,source_speaker=None):
    result={}
    if source_date is not None:
        if not isinstance(source_date,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',source_date):
            raise ValueError('source_date must be an explicit ISO calendar date')
        date.fromisoformat(source_date)
        result.update(source_date=source_date,source_month_day=source_date[5:])
    if source_speaker is not None:
        if source_speaker not in SPEAKERS:raise ValueError('Unsupported source speaker')
        result['source_speaker']=source_speaker
    return result


def constraints(source_date=None,source_month_day=None,source_speaker=None):
    result=provenance(source_date,source_speaker)
    if source_month_day is not None:
        if not isinstance(source_month_day,str) or not re.fullmatch(r'\d{2}-\d{2}',source_month_day):
            raise ValueError('source_month_day must be MM-DD')
        date.fromisoformat('2000-'+source_month_day)
        if source_date is not None and source_date[5:]!=source_month_day:raise ValueError('Conflicting source calendar constraints')
        result['source_month_day']=source_month_day
    return result


def dialogue_lookup(text):
    """Recognize one explicit source day in a historical dialogue question.

    Event-only questions, relative days, ambiguous ranges and statements are
    left to ordinary retrieval. This does not claim full temporal reasoning.
    """
    if not re.search(r'什么|哪|几|多少|如何|怎么|为什么|记得|[?？]',text):return {}
    if not re.search(r'聊|讨论|谈到|说过|说了|提到|提过|分享过|分享了|推荐过|推荐了|建议过|建议了|告诉过|告诉了',text):return {}
    pattern=r'\d{4}-\d{2}-\d{2}|(?:(?:\d{4}|[零〇一二三四五六七八九]{4})年)?[0-9一二三四五六七八九十两]{1,3}月[0-9一二三四五六七八九十]{1,3}(?:日|号)'
    found=list(re.finditer(pattern,text))
    if len(found)!=1:return {}
    if re.search(r'(?:今年|去年|前年|明年|后年|本年|上年)(?:的|在)?\s*$',text[:found[0].start()]):return {}
    if (re.match(r'[，,\s]*(?:之前|之后|以前|以后|以来|为止|到|至)',text[found[0].end():])
            or re.search(r'(?:早于|晚于|截至|自|从)\s*$',text[:found[0].start()])):return {}
    phrase=found[0].group()
    try:
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}',phrase):day=date.fromisoformat(phrase);year=True
        else:
            from app.temporal import _LOCK,_parser
            with _LOCK:parsed=_parser().parse_time(phrase,time_base=datetime(2000,1,1),time_type='time_point',strict=True)
            if parsed.get('type')!='time_point' or parsed.get('definition')!='accurate':return {}
            day=date.fromisoformat(parsed['time'][0][:10]);year='年' in phrase
    except (ValueError,OverflowError):
        # A malformed date in conversational input is not an application error.
        return {}
    result={'source_date':day.isoformat()} if year else {'source_month_day':day.strftime('%m-%d')}
    speakers=set()
    for match in re.finditer(r'(?<![和向给对为跟与同])([我你])(?:当时|之前|曾经|曾|又|也|还|再次)?(?:[和向给对为跟与同][我你])?(?:推荐过|推荐了|建议过|建议了|分享过|分享了|说过|说了|提到过|提到了|提过|告诉过|告诉了)',text):
        speakers.add('user' if match[1]=='我' else 'assistant')
    if len(speakers)==1:result['source_speaker']=speakers.pop()
    return result
