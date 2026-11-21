"""add session bucket, meta description, and mapping confidence to passive exposures

Revision ID: 003_add_session_bucket
Revises: 002_extension_and_telemetry
Create Date: 2026-11-22 00:45:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "003_add_session_bucket"
down_revision: str | None = "002_extension_and_telemetry"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "passive_exposures",
        sa.Column(
            "session_bucket",
            sa.String(length=16),
            nullable=False,
            server_default="",
        ),
    )
    op.create_index(
        op.f("ix_passive_exposures_session_bucket"),
        "passive_exposures",
        ["session_bucket"],
        unique=False,
    )
    op.add_column(
        "passive_exposures",
        sa.Column(
            "meta_description",
            sa.String(length=220),
            nullable=True,
        ),
    )
    op.add_column(
        "passive_exposures",
        sa.Column(
            "mapping_confidence",
            sa.Float(),
            nullable=True,
        ),
    )
    op.create_unique_constraint(
        "uq_passive_exposure_session_bucket",
        "passive_exposures",
        ["user_id", "source_url", "session_bucket"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_passive_exposure_session_bucket",
        "passive_exposures",
        type_="unique",
    )
    op.drop_column("passive_exposures", "mapping_confidence")
    op.drop_column("passive_exposures", "meta_description")
    op.drop_index(
        op.f("ix_passive_exposures_session_bucket"),
        table_name="passive_exposures",
    )
    op.drop_column("passive_exposures", "session_bucket")
