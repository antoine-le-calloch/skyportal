import time
from datetime import datetime, timedelta

import astropy.units as u
import numpy as np
import sqlalchemy as sa
from astropy.time import Time
from mocpy import MOC
from sqlalchemy import and_, or_

from baselayer.app.env import load_env
from baselayer.app.models import init_db
from baselayer.log import make_log
from skyportal.models import (
    DBSession,
    GcnEvent,
    GcnTag,
    Localization,
    LocalizationTag,
    LocalizationTile,
    Obj,
    Photometry,
)
from skyportal.utils.services import check_loaded

env, cfg = load_env()
log = make_log("crossmatch_alert_to_localizations")
log_verbose = make_log("crossmatch_alert_to_localizations_verbose")
init_db(**cfg["database"])

fallback_in_days = 2


def is_obj_in_localizations(ra, dec, localizations):
    matching_localizations = [
        dateobs
        for dateobs, moc in localizations
        if moc.contains_lonlat(ra * u.deg, dec * u.deg)
    ]
    return matching_localizations


def get_objs_created_after(session, created_after, snr_threshold):
    start_time = time.time()
    objs = session.scalars(
        sa.select(Obj)
        .join(Photometry)
        .where(
            and_(
                Obj.created_at > created_after,
                (Photometry.flux / Photometry.fluxerr) > snr_threshold,
            )
        )
    ).all()
    fallback_in_mjd = Time(datetime.utcnow() - timedelta(days=fallback_in_days)).mjd
    filtered_objs = []
    for obj in objs:
        # Check if the first detection was within the last day
        if (
            obj.photometry
            and min(obj.photometry, key=lambda p: p.mjd).mjd >= fallback_in_mjd
        ):
            filtered_objs.append(obj)
    if filtered_objs:
        print(
            f"Found {len(filtered_objs)} valid objects on {len(objs)} in {time.time() - start_time:.2f} seconds"
        )
    return filtered_objs


def get_gcn_events_dateobs(session, dateobs, get_first_only=False):
    stmt = (
        sa.select(GcnEvent.dateobs)
        .join(GcnTag)
        .join(Localization)
        .join(LocalizationTag)
        .where(
            or_(
                and_(
                    GcnTag.text.in_(["GW", "BNS", "NSBH", "SVOM"]),  # Tags to include
                    GcnTag.text.notin_(["BBH"]),  # Tags to exclude
                ),
                and_(GcnTag.text == "Fermi", LocalizationTag.text == "< 1000 sq.deg"),
            ),
            GcnEvent.dateobs > dateobs,
        )
        .order_by(GcnEvent.dateobs.desc())
        .distinct()
    )
    if get_first_only:
        return session.scalars(stmt).first()
    return session.scalars(stmt).all()


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
        A list of tuples, each containing a localization dateobs and its corresponding MOC.
    """
    # get the events since dateobs, that have a tag "GW"
    gcn_dateobs = get_gcn_events_dateobs(session, dateobs)
    if not gcn_dateobs:
        return []
    localizations = session.scalars(
        sa.select(Localization)
        .where(Localization.dateobs.in_(gcn_dateobs))
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
        results.append((loc.dateobs, moc))
    return results


@check_loaded(logger=log)
def service(*args, **kwargs):
    latest_gcn_date_obs = datetime.utcnow() - timedelta(days=fallback_in_days)
    latest_obj_created_time = datetime.utcnow() - timedelta(hours=2)
    cumulative_probability = 0.95
    snr_threshold = 5.0  # Minimum SNR for photometry to consider an object
    localizations = None
    while True:
        with DBSession() as session:
            # Check if new GCNs have been observed since the last observation
            new_latest_gcn_dateobs = get_gcn_events_dateobs(
                session, latest_gcn_date_obs, get_first_only=True
            )

            if new_latest_gcn_dateobs:
                # If new GCNs, fetch again localizations from the last 2 days
                log(f"New GCNs found, fetching skymaps")
                start_time = time.time()
                localizations = get_localizations(
                    session,
                    datetime.utcnow() - timedelta(days=fallback_in_days),
                    cumulative_probability,
                )
                log_verbose(
                    f"Fetching {len(localizations)} localizations and creating MOCs took {time.time() - start_time:.2f} seconds"
                )
                latest_gcn_date_obs = new_latest_gcn_dateobs

            # If no new GCNs, check for expired localizations and remove them
            elif localizations:
                for dateobs, moc in localizations.copy():
                    if dateobs < datetime.utcnow() - timedelta(days=fallback_in_days):
                        print(f"Removed expired skymap {dateobs}")
                        localizations.remove((dateobs, moc))

            # Retrieve objects created after last object creation time
            if localizations:
                objs = get_objs_created_after(
                    session,
                    max(
                        latest_obj_created_time,
                        datetime.utcnow() - timedelta(days=fallback_in_days),
                    ),
                    snr_threshold,
                )
                crossmatches = []
                start_time = time.time()
                for obj in objs:
                    matching_localizations = is_obj_in_localizations(
                        obj.ra, obj.dec, localizations
                    )
                    if matching_localizations:
                        crossmatches.append(
                            {"obj": obj, "localizations": matching_localizations}
                        )
                        # TODO: Do something with the object, e.g., publish somewhere
                        log_verbose(
                            f"{obj.id} in localizations {matching_localizations}"
                        )
                if objs:
                    log_verbose(
                        f"{datetime.utcnow()} Found {len(crossmatches)} crossmatches in {time.time() - start_time:.2f} seconds\n"
                    )
                    last_obj_created_time = max(
                        last_obj_created_time, max(obj.created_at for obj in objs)
                    )
            else:
                print("No skymaps available. Waiting...")
        time.sleep(20)


if __name__ == "__main__":
    try:
        service()
    except Exception as e:
        log(f"Error in crossmatch_alert_to_localizations service: {e}")
        raise e
