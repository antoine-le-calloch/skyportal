"""observingruns_end

Revision ID: 5516eb897b1c
Revises: bc7afebe918f
Create Date: 2023-10-08 10:10:00.585039

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "5516eb897b1c"
down_revision = "bc7afebe918f"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column(
        "observingruns", sa.Column("run_end_utc", sa.DateTime(), nullable=True)
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("observingruns", "run_end_utc")
    # ### end Alembic commands ###
