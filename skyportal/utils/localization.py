import time

import numpy as np
import sqlalchemy as sa
from mocpy import MOC

from baselayer.log import make_log
from skyportal.models import (
    GcnTag,
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


def get_localizations(session, dateobs, cumulative_probability):
    """Get all localizations between dateobs and now. For each localization,
    compute the MOC corresponding to the cumulative_probability threshold.

    Parameters
    ----------
    session: `sqlalchemy.orm.session.Session`
        The database session to use for the query.
    dateobs : datetime.datetime
        The starting date and time to filter localizations from. Only localizations
        with a date greater than or equal to this will be returned.
    cumulative_probability : float
        The cumulative probability threshold for the MOC. Only tiles contributing
        to this cumulative probability will be included in the MOC.

    Returns
    -------
    results : list of tuples
        A list of tuples, each containing a localization ID and its corresponding MOC.
    """
    # get the events since dateobs, that have a tag "GW"
    gw_dateobs = session.scalars(
        sa.select(GcnTag.dateobs).where(GcnTag.text == "GW", GcnTag.dateobs >= dateobs)
    ).all()
    if len(gw_dateobs) == 0:
        return []
    localizations = session.scalars(
        sa.select(Localization)
        .where(Localization.dateobs.in_(gw_dateobs))
        .order_by(Localization.dateobs.desc())
    ).all()

    results = []
    for loc in localizations:
        partition_key = loc.dateobs
        localizationtile_partition_name = (
            f"{partition_key.year}_{partition_key.month:02d}"
        )
        localizationtilescls = LocalizationTile.partitions.get(
            localizationtile_partition_name, None
        )
        if localizationtilescls is None:
            localizationtilescls = LocalizationTile.partitions.get(
                "def", LocalizationTile
            )
        else:
            exists = session.scalars(
                sa.select(localizationtilescls.id).where(
                    localizationtilescls.localization_id == loc.id
                )
            ).first()
            if not exists:
                localizationtilescls = LocalizationTile.partitions.get(
                    "def", LocalizationTile
                )

        partition = localizationtilescls.__tablename__
        stmt = f"""
            SELECT healpix
            FROM (
                SELECT {partition}.id,
                       {partition}.healpix,
                       SUM({partition}.probdensity *
                           (upper({partition}.healpix) - lower({partition}.healpix)) * 3.6331963520923245e-18
                       ) OVER (ORDER BY {partition}.probdensity DESC) AS cum_prob
                FROM {partition}
                WHERE {partition}.localization_id = {loc.id}
            ) AS lt
            WHERE lt.cum_prob <= {cumulative_probability}
        """
        tiles = session.execute(sa.text(stmt)).all()
        if len(tiles) == 0:
            continue
        healpix_list = [[tile.healpix.lower, tile.healpix.upper] for tile in tiles]
        moc = MOC.from_depth29_ranges(29, ranges=np.array(healpix_list))
        results.append((loc.id, moc))
    return results
