"""create extension tokens and passive exposures

Revision ID: 002_extension_and_telemetry
Revises: 001_auth_tables
Create Date: 2026-11-21 23:15:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "002_extension_and_telemetry"
down_revision: str | None = "001_auth_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Create extension_tokens table
    op.create_table(
        "extension_tokens",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "name",
            sa.String(length=100),
            nullable=False,
            server_default="Chrome Extension",
        ),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_extension_tokens_user_id",
        "extension_tokens",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_extension_tokens_token_hash",
        "extension_tokens",
        ["token_hash"],
        unique=True,
    )

    # 2. Create passive_exposures table
    json_col_type = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")

    op.create_table(
        "passive_exposures",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "client_event_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "concept_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column("source_domain", sa.String(length=255), nullable=False),
        sa.Column("source_url", sa.String(length=256), nullable=False),
        sa.Column("page_title", sa.String(length=160), nullable=False),
        sa.Column(
            "media_type",
            sa.String(length=32),
            nullable=False,
            server_default="article",
        ),
        sa.Column("active_dwell_seconds", sa.Integer(), nullable=False),
        sa.Column(
            "estimated_coverage_pct",
            sa.Float(),
            nullable=False,
            server_default="0.0",
        ),
        sa.Column("viewed_headings", json_col_type, nullable=False),
        sa.Column("top_keywords", json_col_type, nullable=False),
        sa.Column(
            "is_revisit",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("days_since_last_visit", sa.Float(), nullable=True),
        sa.Column("video_metrics", json_col_type, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "received_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_passive_exposures_client_event_id",
        "passive_exposures",
        ["client_event_id"],
        unique=True,
    )
    op.create_index(
        "ix_passive_exposures_user_id",
        "passive_exposures",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_passive_exposures_concept_id",
        "passive_exposures",
        ["concept_id"],
        unique=False,
    )
    op.create_index(
        "ix_passive_exposures_source_domain",
        "passive_exposures",
        ["source_domain"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_table("passive_exposures")
    op.drop_table("extension_tokens")
