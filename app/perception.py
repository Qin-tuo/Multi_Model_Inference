"""Bounded temporal rules over yolo_ros's tracked, 1-based COCO keypoints.

Detection/tracking/inference remain in yolo_ros. These rules describe anonymous
presence and hand gestures; they do not identify people or infer intent.
"""
from collections import Counter, deque
from dataclasses import dataclass, field
import hashlib
import math
import threading
import time
import uuid


@dataclass(frozen=True)
class Person:
    track_id: str
    score: float
    cx: float
    cy: float
    width: float
    height: float
    keypoints: dict  # yolo_ros IDs 6/7 shoulders, 8/9 elbows, 10/11 wrists.
    distance_m: float | None = None


@dataclass
class Track:
    person: Person
    first_seen_ns: int
    last_seen_ns: int
    observations: int = 1
    high_since_ns: int | None = None
    low_since_ns: int | None = None
    released_at_ns: int | None = None
    emitted: bool = False
    samples: deque = field(default_factory=lambda: deque(maxlen=32))


class GestureRules:
    def __init__(self, *, clock=time.monotonic_ns, max_tracks=16, max_event_queue=32,
                 max_age_ms=200, stale_seconds=.5, confirmation_seconds=.3,
                 raised_seconds=1.2, neutral_seconds=.5, wave_window_seconds=1.2,
                 keypoint_score=.6, wave_excursion=.18, event_ttl_seconds=2):
        if (not 1 <= max_tracks <= 64 or not 1 <= max_event_queue <= 100
                or not all(math.isfinite(x) and x > 0 for x in
                    (max_age_ms, stale_seconds, confirmation_seconds, raised_seconds,
                     neutral_seconds, wave_window_seconds, wave_excursion, event_ttl_seconds))
                or not 0 < keypoint_score <= 1):
            raise ValueError("Invalid gesture bounds")
        self.clock, self.lock = clock, threading.RLock()
        self.max_tracks, self.max_event_queue = max_tracks, max_event_queue
        self.max_age_ns, self.stale_ns = int(max_age_ms*1e6), int(stale_seconds*1e9)
        self.confirm_ns, self.raise_ns, self.neutral_ns = [int(x*1e9) for x in (confirmation_seconds,raised_seconds,neutral_seconds)]
        self.wave_window_ns, self.event_ttl_ns = int(wave_window_seconds*1e9), int(event_ttl_seconds*1e9)
        self.keypoint_score, self.wave_excursion = keypoint_score, wave_excursion
        self.epoch = uuid.uuid4().hex
        self.source_key = None
        self.last_arrival_ns = self.last_capture_ns = self.last_source_ns = None
        self.tracks, self.events = {}, deque()
        self.counts = Counter()

    def _reset(self, reason):
        self.epoch = uuid.uuid4().hex
        self.tracks.clear(); self.events.clear()
        self.last_arrival_ns = self.last_capture_ns = self.last_source_ns = None
        self.counts["reset:"+reason] += 1

    def invalidate(self, reason="input_closed"):
        with self.lock: self._reset(reason)

    def _hands(self, person):
        raised, valid_sides = [], 0
        for side, shoulder, elbow, wrist in (("left",6,8,10),("right",7,9,11)):
            points = [person.keypoints.get(index) for index in (shoulder,elbow,wrist)]
            if any(p is None or p[2] < self.keypoint_score for p in points): continue
            s,e,w = points
            valid_sides += 1
            if w[1] < s[1] - .12*person.height and e[1] < s[1] + .25*person.height:
                raised.append((side,(w[0]-s[0])/person.width))
        return raised, valid_sides == 2 and not raised

    def _wave(self, samples, side):
        points = [(t,x) for t,s,x in samples if s == side]
        if len(points) < 5 or points[-1][0]-points[0][0] < self.confirm_ns: return False
        pivot = points[0][1]
        direction = reversals = 0
        for _,value in points[1:]:
            if not direction:
                if abs(value-pivot) >= self.wave_excursion:
                    direction = 1 if value > pivot else -1
                    pivot = value
            elif direction*(value-pivot) >= 0:
                pivot = value
            elif abs(value-pivot) >= self.wave_excursion:
                direction *= -1
                reversals += 1
                pivot = value
        return reversals >= 2

    def update(self, people, *, source_key, source_stamp_ns, capture_ns, arrival_ns=None):
        now = self.clock() if arrival_ns is None else arrival_ns
        with self.lock:
            if not source_key or len(source_key)>128 or source_stamp_ns <= 0 or not 0 <= now-capture_ns <= self.max_age_ns:
                self.counts["stale_or_invalid_clock"] += 1; return
            if self.source_key != source_key:
                self._reset("publisher_changed"); self.source_key = source_key
            elif self.last_arrival_ns is not None and now-self.last_arrival_ns > self.stale_ns:
                self._reset("stream_gap")
            elif self.last_source_ns is not None and source_stamp_ns < self.last_source_ns-self.stale_ns:
                self._reset("source_clock_changed")
            if self.last_source_ns is not None and source_stamp_ns <= self.last_source_ns:
                self.counts["duplicate_or_out_of_order"] += 1; return
            self.last_arrival_ns, self.last_capture_ns, self.last_source_ns = now,capture_ns,source_stamp_ns
            for key in list(self.tracks):
                if now-self.tracks[key].last_seen_ns > 2_000_000_000: self.tracks.pop(key)
            selected = sorted(people,key=lambda p:p.score,reverse=True)
            if len(selected) > self.max_tracks: self.counts["person_limit"] += len(selected)-self.max_tracks
            seen = set()
            for person in selected[:self.max_tracks]:
                valid = (person.track_id and len(person.track_id)<=64 and person.track_id not in seen
                    and all(math.isfinite(x) for x in (person.score,person.cx,person.cy,person.width,person.height))
                    and .5 <= person.score <= 1 and 15 <= person.width <= 1280 and 30 <= person.height <= 960
                    and 0 <= person.cx <= 640 and 0 <= person.cy <= 480
                    and len(person.keypoints) <= 17
                    and all(len(p)==3 and all(math.isfinite(x) for x in p) and 0<=p[2]<=1
                            and -64<=p[0]<=704 and -48<=p[1]<=528 for p in person.keypoints.values())
                    and (person.distance_m is None or math.isfinite(person.distance_m) and 0<person.distance_m<=20))
                if not valid: self.counts["invalid_person"] += 1; continue
                seen.add(person.track_id)
                track = self.tracks.get(person.track_id)
                if track is not None and capture_ns-track.last_seen_ns > self.stale_ns:
                    # An occlusion cannot count as a sustained gesture. Keep
                    # the emitted latch until an observed neutral pose releases it.
                    track.first_seen_ns=capture_ns; track.observations=0
                    track.high_since_ns=track.low_since_ns=None; track.samples.clear()
                if track is None:
                    if len(self.tracks)>=self.max_tracks:
                        oldest=min(self.tracks,key=lambda k:self.tracks[k].last_seen_ns)
                        self.tracks.pop(oldest); self.counts["track_eviction"] += 1
                    track = self.tracks[person.track_id] = Track(person,capture_ns,capture_ns)
                else:
                    track.person = person
                    track.observations += 1
                    track.last_seen_ns = capture_ns
                raised, neutral = self._hands(person)
                if neutral:
                    track.high_since_ns=None; track.samples.clear()
                    if track.low_since_ns is None: track.low_since_ns=capture_ns
                    if (capture_ns-track.low_since_ns >= self.neutral_ns
                            and (track.released_at_ns is None or track.released_at_ns < track.low_since_ns)):
                        track.released_at_ns=capture_ns
                        track.high_since_ns=None; track.emitted=False; track.samples.clear()
                    continue
                track.low_since_ns=None
                if not raised:
                    track.high_since_ns=None; track.samples.clear()
                    continue
                if track.high_since_ns is None: track.high_since_ns=capture_ns
                for side,x in raised: track.samples.append((capture_ns,side,x))
                while track.samples and capture_ns-track.samples[0][0] > self.wave_window_ns: track.samples.popleft()
                if track.emitted or track.observations<3 or capture_ns-track.first_seen_ns<self.confirm_ns: continue
                kind = "wave" if any(self._wave(track.samples,side) for side,_ in raised) else None
                if kind is None and capture_ns-track.high_since_ns>=self.raise_ns: kind="hand_raise"
                if kind is None: continue
                track.emitted=True
                key=f"{source_key}:{person.track_id}:{track.high_since_ns}:{kind}"
                event={"event_id":hashlib.sha256(key.encode()).hexdigest(),"kind":kind,"track_id":person.track_id,
                    "source_key":source_key,"epoch":self.epoch,"capture_ns":capture_ns,
                    "source_stamp_ns":source_stamp_ns,"deadline_ns":capture_ns+self.event_ttl_ns,
                    "text":"你好，有什么可以帮你？"}
                if len(self.events)>=self.max_event_queue:
                    self.events.popleft(); self.counts["event_queue_eviction"] += 1
                self.events.append(event); self.counts["event:"+kind] += 1
            self.counts["frames"] += 1

    def valid(self, event, *, started=False):
        now=self.clock()
        with self.lock:
            track=self.tracks.get(event["track_id"])
            return bool(event["epoch"]==self.epoch and event["source_key"]==self.source_key
                and self.last_arrival_ns is not None and now-self.last_arrival_ns<=self.stale_ns
                and track is not None and now-track.last_seen_ns<=self.stale_ns
                and (started or now<=event["deadline_ns"]))

    def drain_events(self):
        with self.lock:
            events=list(self.events); self.events.clear(); return events

    def snapshot(self):
        now=self.clock()
        with self.lock:
            fresh=self.last_arrival_ns is not None and now-self.last_arrival_ns<=self.stale_ns
            rows=[]
            if fresh:
                for key,track in self.tracks.items():
                    if now-track.last_seen_ns>self.stale_ns or track.observations<3: continue
                    p=track.person
                    rows.append({"track_id":key,"score":p.score,"distance_m":p.distance_m,
                        "direction":"left" if p.cx<640*.35 else "right" if p.cx>640*.65 else "center",
                        "released_at_ns":track.released_at_ns,"last_seen_ns":track.last_seen_ns})
            return {"state":"READY" if fresh else "STALE","epoch":self.epoch,"source_key":self.source_key,
                "people":rows,"person_count":len(rows),"tracked_state_count":len(self.tracks),
                "latest_result_age_ms":(now-self.last_capture_ns)/1e6 if self.last_capture_ns else None,
                "counts":dict(self.counts),"scope":"anonymous tracked detections; no identity or ground-truth accuracy claim"}
