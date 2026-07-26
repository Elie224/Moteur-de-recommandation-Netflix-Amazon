"""Add eBay product fields.

Revision ID: 8b3d0d6e7a11
Revises: 472530110c52
"""
from typing import Sequence, Union

from alembic import op
from sqlalchemy.dialects import postgresql
import sqlalchemy as sa


revision: str = "8b3d0d6e7a11"
down_revision: Union[str, None] = "472530110c52"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("products", sa.Column("condition_description", sa.Text(), nullable=True))
    op.add_column("products", sa.Column("seller_feedback_percentage", sa.Numeric(5, 2), nullable=True))
    op.add_column("products", sa.Column("seller_feedback_score", sa.BigInteger(), nullable=True))
    op.add_column("products", sa.Column("shipping_currency", sa.String(10), nullable=True))
    op.add_column("products", sa.Column("additional_images", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")))
    op.add_column("products", sa.Column("item_end_date", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_products_brand", "products", ["brand"], unique=False)
    op.create_index("ix_products_availability", "products", ["availability"], unique=False)
    op.create_index("ix_products_marketplace", "products", ["marketplace"], unique=False)
    op.create_index("ix_products_item_end_date", "products", ["item_end_date"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_products_item_end_date", table_name="products")
    op.drop_index("ix_products_marketplace", table_name="products")
    op.drop_index("ix_products_availability", table_name="products")
    op.drop_index("ix_products_brand", table_name="products")
    op.drop_column("products", "item_end_date")
    op.drop_column("products", "additional_images")
    op.drop_column("products", "shipping_currency")
    op.drop_column("products", "seller_feedback_score")
    op.drop_column("products", "seller_feedback_percentage")
    op.drop_column("products", "condition_description")
