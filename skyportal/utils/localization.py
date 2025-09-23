import time

import sqlalchemy as sa

from baselayer.log import make_log
from skyportal.models import (
    Localization,
    LocalizationTile,
)

log_verbose = make_log("localization_verbose")


def get_localization(localization_dateobs, localization_name, session):
    """Get the localization ID and corresponding partitioned localization tiles table name.

    Parameters
    ----------
    localization_dateobs: datetime.datetime
        The date of the localization observation.
    localization_name: str or None
        The name of the localization. If None, the most recent localization with the given dateobs
        will be returned.
    session: `sqlalchemy.orm.session.Session`
        The database session to use for the query.

    Returns
    -------
    localization_id : `int`
        The ID of the localization.
    localizationtiles_table_name : `str`
        The name of the partitioned localization tiles table.
    """
    startTime = time.time()
    localization_dateobs_str = localization_dateobs.strftime("%Y-%m-%d %H:%M:%S")
    if localization_name is None:
        localization_id = session.scalars(
            sa.select(Localization.id)
            .where(Localization.dateobs == localization_dateobs_str)
            .order_by(Localization.created_at.desc())
        ).first()
    else:
        localization_id = session.scalars(
            sa.select(Localization.id)
            .where(Localization.dateobs == localization_dateobs_str)
            .where(Localization.localization_name == localization_name)
            .order_by(Localization.modified.desc())
        ).first()
    if localization_id is None:
        if localization_name is not None:
            raise ValueError(
                f"Localization {localization_dateobs_str} with name {localization_name} not found",
            )
        else:
            raise ValueError(
                f"Localization {localization_dateobs_str} not found",
            )

    partition_key = localization_dateobs
    localizationtile_partition_name = f"{partition_key.year}_{partition_key.month:02d}"
    localizationtilescls = LocalizationTile.partitions.get(
        localizationtile_partition_name, None
    )
    if localizationtilescls is None:
        localizationtilescls = LocalizationTile.partitions.get("def", LocalizationTile)
    else:
        if not (
            session.scalars(
                sa.select(localizationtilescls.id).where(
                    localizationtilescls.localization_id == localization_id
                )
            ).first()
        ):
            localizationtilescls = LocalizationTile.partitions.get(
                "def", LocalizationTile
            )

    endTime = time.time()
    log_verbose(f"get_localization took {endTime - startTime} seconds")

    return localization_id, localizationtilescls.__tablename__
