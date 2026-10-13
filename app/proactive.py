"""Gesture policy and durable deduplication on the existing SQLAlchemy outbox.

There is no additional scheduler, inference worker or audio queue here. The
existing NotificationPump polls this adapter before claiming its next output.
Only anonymous greetings are admitted, never personal memory or model prompts.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import time

from sqlalchemy import Column, Integer, BigInteger, MetaData, String, Table, delete, insert, select, update, func
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.reminders import _NOTIFICATIONS, _REMINDERS, _validate_id

_META = MetaData()
_POLICY = Table('assistant_visual_policy', _META,
    Column('user_id', String(256), primary_key=True), Column('muted', Integer, nullable=False),
    Column('boot_id', String(64)), Column('last_admitted_ns', BigInteger))
_LATCH = Table('assistant_visual_latches', _META,
    Column('id', String(64), primary_key=True), Column('user_id', String(256), nullable=False),
    Column('source_key', String(128), nullable=False), Column('track_id', String(64), nullable=False),
    Column('event_capture_ns', BigInteger, nullable=False))


class VisualFeedback:
    def __init__(self, service, rules, user_id, *, cooldown_seconds=2, clock=time.monotonic_ns):
        _validate_id(user_id, 'user_id')
        if not .1 <= cooldown_seconds <= 3600: raise ValueError('Invalid visual cooldown')
        if service.visual_guard is not None: raise RuntimeError('Visual feedback is already attached')
        self.service, self.rules, self.user_id, self.clock = service, rules, user_id, clock
        self.cooldown_ns = int(cooldown_seconds*1e9)
        self.boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        self.events, self.counts = {}, Counter()
        self.closed = False
        _META.create_all(service.engine)
        with service.engine.begin() as connection:
            connection.execute(sqlite_insert(_POLICY).values(user_id=user_id, muted=0).on_conflict_do_nothing())
            self.muted = bool(connection.execute(select(_POLICY.c.muted).where(_POLICY.c.user_id==user_id)).scalar_one())
        service.visual_guard = self.allowed

    def allowed(self, notification_id, *, started=False):
        event = self.events.get(notification_id)
        return bool(not self.closed and not self.muted and event and self.rules.valid(event, started=started))

    def set_muted(self, muted):
        if not isinstance(muted, bool): raise ValueError('muted must be boolean')
        with self.service._lock, self.service.engine.begin() as connection:
            connection.execute(update(_POLICY).where(_POLICY.c.user_id==self.user_id).values(muted=int(muted)))
            self.muted = muted
            # Enabling again cannot resurrect an attempt revoked by this mute.
            if muted: self.events.clear()

    def _key(self, source, track):
        return hashlib.sha256(f'{len(self.user_id)}:{self.user_id}:{source}:{track}'.encode()).hexdigest()

    def tick(self):
        if self.closed: return
        snapshot = self.rules.snapshot()
        events = self.rules.drain_events()
        now = self.clock()
        with self.service._lock, self.service.engine.begin() as connection:
            # Release requires observed neutral keypoints, not disappearance,
            # app restart or a new analysis epoch for the same publisher.
            if snapshot['source_key']:
                connection.execute(delete(_LATCH).where(_LATCH.c.user_id==self.user_id,
                    _LATCH.c.source_key!=snapshot['source_key']))
            for person in snapshot['people']:
                if person['released_at_ns'] is not None:
                    connection.execute(delete(_LATCH).where(
                        _LATCH.c.id==self._key(snapshot['source_key'],person['track_id']),
                        _LATCH.c.event_capture_ns<person['released_at_ns']))
            # Preserve a started attempt until the sink reports completion or
            # uncertainty. Pending stale/muted events are terminal and do not retry.
            rows = connection.execute(select(_NOTIFICATIONS).where(_NOTIFICATIONS.c.user_id==self.user_id,
                _NOTIFICATIONS.c.id.startswith('visual:'),
                _NOTIFICATIONS.c.status.in_(['pending','claimed','playing']))).mappings().all()
            live_ids = {row['id'] for row in rows}
            for key in list(self.events):
                if key not in live_ids: self.events.pop(key, None)
            for row in rows:
                if row['status']=='playing' or self.allowed(row['id']): continue
                connection.execute(update(_NOTIFICATIONS).where(_NOTIFICATIONS.c.id==row['id'])
                    .values(status='expired', delivery_evidence='visual_stale_or_muted'))
                connection.execute(update(_REMINDERS).where(_REMINDERS.c.id==row['reminder_id']).values(status='expired'))
                self.service._clear_attempt(row['id'],row['attempt_id'])
                self.events.pop(row['id'],None)
                self.counts['expired'] += 1
            for event in events:
                if not self.rules.valid(event):
                    self.counts['stale'] += 1; continue
                latch = self._key(event['source_key'],event['track_id'])
                if connection.execute(select(_LATCH.c.id).where(_LATCH.c.id==latch)).first():
                    self.counts['latched'] += 1; continue
                if connection.execute(select(func.count()).select_from(_LATCH).where(_LATCH.c.user_id==self.user_id)).scalar_one()>=128:
                    self.counts['latch_capacity'] += 1; continue
                # Suppressed events also latch; removing mute/busy must not
                # greet an old held pose. A newly observed gesture may retry.
                connection.execute(insert(_LATCH).values(id=latch,user_id=self.user_id,
                    source_key=event['source_key'],track_id=event['track_id'],event_capture_ns=event['capture_ns']))
                policy = connection.execute(select(_POLICY).where(_POLICY.c.user_id==self.user_id)).mappings().one()
                cooldown = (policy['boot_id']==self.boot_id and policy['last_admitted_ns'] is not None
                            and now-policy['last_admitted_ns']<self.cooldown_ns)
                if self.muted or self.service.is_muted(self.user_id) or cooldown:
                    self.counts['suppressed'] += 1; continue
                pending = connection.execute(select(func.count()).select_from(_NOTIFICATIONS).where(
                    _NOTIFICATIONS.c.user_id==self.user_id,
                    _NOTIFICATIONS.c.status.in_(['pending','claimed','playing']))).scalar_one()
                if pending>=self.service.max_pending:
                    self.counts['outbox_capacity'] += 1; continue
                key = 'visual:'+hashlib.sha256((self.user_id+':'+event['event_id']).encode()).hexdigest()[:56]
                if connection.execute(select(_NOTIFICATIONS.c.id).where(_NOTIFICATIONS.c.id==key)).first():
                    self.counts['duplicate'] += 1; continue
                wall = datetime.now(timezone.utc)
                expiry = wall+timedelta(seconds=max(0,(event['deadline_ns']-now)/1e9))
                text = '你好，有什么可以帮你？'
                connection.execute(insert(_REMINDERS).values(id=key,user_id=self.user_id,text=text,
                    due_utc=wall.isoformat(),timezone='UTC',status='fired',created_utc=wall.isoformat(),expires_utc=expiry.isoformat()))
                connection.execute(insert(_NOTIFICATIONS).values(id=key,reminder_id=key,user_id=self.user_id,
                    text=text,status='pending',created_utc=wall.isoformat(),delivery_attempts=0))
                connection.execute(update(_POLICY).where(_POLICY.c.user_id==self.user_id)
                    .values(boot_id=self.boot_id,last_admitted_ns=now))
                self.events[key]=event
                self.counts['enqueued'] += 1

    def status(self):
        return {'muted':self.muted,'cooldown_seconds':self.cooldown_ns/1e9,
                'live_events':len(self.events),'counts':dict(self.counts),'scene':self.rules.snapshot()}

    def close(self):
        self.closed = True
        self.events.clear()
        self.service.visual_guard = None
