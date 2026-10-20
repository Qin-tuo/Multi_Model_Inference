# SPDX-License-Identifier: Apache-2.0
"""Small versioned face gallery backed by the application's SQLAlchemy dependency.

Enrollment is explicit. Matching labels do not authorize conversation or tools.
Embedding spaces are versioned so a replacement model cannot reuse incompatible
profiles silently. No images or camera frames are persisted in this store.
"""

from pathlib import Path
import threading
import time

import numpy as np
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    LargeBinary,
    Float,
    select,
    func,
    delete,
)

from app.face_tasks import normalize_embedding


class IdentityGallery:
    def __init__(self, path, capacity=32, threshold=0.363, margin=0.05, dimensions=128):
        if (
            dimensions not in (128, 512)
            or not -1 <= threshold <= 1
            or not 0 <= margin <= 2
        ):
            raise ValueError("Invalid face gallery feature contract")
        self.lock = threading.RLock()
        self.capacity = capacity
        self.threshold = threshold
        self.margin = margin
        self.dimensions = dimensions
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            "sqlite:///" + str(path),
            connect_args={"check_same_thread": False, "timeout": 2},
        )
        metadata = MetaData()
        self.table = Table(
            "vision_identities",
            metadata,
            Column("label", String(64), primary_key=True),
            Column("space", String(200), nullable=False),
            Column("embedding", LargeBinary, nullable=False),
            Column("created", Float, nullable=False),
        )
        metadata.create_all(self.engine)

    def enroll(self, label, space, embedding):
        label = label.strip()
        if not 1 <= len(label) <= 64 or not space or len(space) > 200:
            raise ValueError("Invalid identity label or model space")
        vector = normalize_embedding(embedding, self.dimensions)
        with self.lock, self.engine.connect() as connection:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            exists = connection.execute(
                select(self.table.c.label).where(self.table.c.label == label)
            ).first()
            count = connection.execute(
                select(func.count()).select_from(self.table)
            ).scalar_one()
            if not exists and count >= self.capacity:
                raise ValueError("Identity gallery is full")
            connection.execute(delete(self.table).where(self.table.c.label == label))
            connection.execute(
                self.table.insert().values(
                    label=label,
                    space=space,
                    embedding=vector.tobytes(),
                    created=time.time(),
                )
            )
            connection.commit()
        return {"label": label, "space": space}

    def match(self, space, embedding):
        vector = normalize_embedding(embedding, self.dimensions)
        with self.lock, self.engine.connect() as connection:
            rows = connection.execute(
                select(self.table.c.label, self.table.c.embedding)
                .where(self.table.c.space == space)
                .limit(self.capacity)
            ).all()
        scores = sorted(
            (
                (
                    float(
                        np.clip(
                            np.dot(
                                vector,
                                normalize_embedding(
                                    np.frombuffer(row.embedding, np.float32),
                                    self.dimensions,
                                ),
                            ),
                            -1,
                            1,
                        )
                    ),
                    row.label,
                )
                for row in rows
            ),
            reverse=True,
        )
        score, label = scores[0] if scores else (None, None)
        status = "unknown"
        if score is not None and score >= self.threshold:
            status = (
                "ambiguous"
                if len(scores) > 1 and score - scores[1][0] < self.margin
                else "matched"
            )
        return {
            "status": status,
            "label": label if status == "matched" else None,
            "score": score,
            "space": space,
            "threshold": self.threshold,
        }

    def list(self):
        with self.lock, self.engine.connect() as connection:
            return [
                dict(row._mapping)
                for row in connection.execute(
                    select(self.table.c.label, self.table.c.space, self.table.c.created)
                    .order_by(self.table.c.label)
                    .limit(self.capacity)
                )
            ]

    def remove(self, label):
        with self.lock, self.engine.begin() as connection:
            return bool(
                connection.execute(
                    delete(self.table).where(self.table.c.label == label)
                ).rowcount
            )

    def close(self):
        self.engine.dispose()
