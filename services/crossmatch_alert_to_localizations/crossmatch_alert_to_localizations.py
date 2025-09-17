import time
from datetime import datetime, timedelta

import astropy.units as u
import sqlalchemy as sa
from astropy.time import Time
from sqlalchemy import and_

from baselayer.app.env import load_env
from baselayer.app.models import init_db
from baselayer.log import make_log
from skyportal.models import DBSession, GcnTag, Obj, Photometry
from skyportal.utils.localization import get_localizations
from skyportal.utils.services import check_loaded

env, cfg = load_env()
log = make_log("crossmatch_alert_to_localizations")
log_verbose = make_log("crossmatch_alert_to_localizations_verbose")
init_db(**cfg["database"])


def is_obj_in_localizations(ra, dec, localizations):
    matching_localizations = [
        loc_id
        for loc_id, moc in localizations
        if moc.contains_lonlat(ra * u.deg, dec * u.deg)
    ]
    return matching_localizations


def get_objs_created_after(session, created_after, snr_threshold):
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
    mjd_1day_ago = Time(datetime.utcnow() - timedelta(days=1)).mjd
    filtered_objs = []
    for obj in objs:
        # Check if the first detection was within the last day
        if (
            obj.photometry
            and min(obj.photometry, key=lambda p: p.mjd).mjd >= mjd_1day_ago
        ):
            filtered_objs.append(obj)
    return filtered_objs


@check_loaded(logger=log)
def service(*args, **kwargs):
    # Start by checking GCNs and objects from the last 2 days
    two_days_ago = datetime.utcnow() - timedelta(days=2)
    latest_gcn_date_obs = two_days_ago
    latest_obj_created_time = two_days_ago
    cumulative_probability = 0.95
    snr_threshold = 5.0  # Minimum SNR for photometry to consider an object
    localizations = None
    while True:
        with DBSession() as session:
            # Check if new GCNs with tag "GW" have been observed since the last observation
            new_latest_gcn_date_obs = session.scalars(
                sa.select(GcnTag.dateobs)
                .where(GcnTag.text == "GW", GcnTag.dateobs > latest_gcn_date_obs)
                .order_by(GcnTag.dateobs.desc())
            ).first()
            if (
                localizations is None or new_latest_gcn_date_obs
            ):  # If new GCNs, fetch again localizations from the last 2 days
                log(f"New GCNs found, recreating localizations list.")
                start_time = time.time()
                localizations = get_localizations(
                    session,
                    datetime.utcnow() - timedelta(days=2),
                    cumulative_probability,
                )
                log_verbose(
                    f"Fetching {len(localizations)} localizations and creating MOCs took {time.time() - start_time:.2f} seconds"
                )
                latest_gcn_date_obs = new_latest_gcn_date_obs or latest_gcn_date_obs

            # Retrieve objects created after last refresh time
            objs = get_objs_created_after(
                session, latest_obj_created_time, snr_threshold
            )
            start_time = time.time()
            for obj in objs:
                if localizations and is_obj_in_localizations(
                    obj.ra, obj.dec, localizations
                ):
                    log(f"Object {obj.id} is in a localization.")
            if objs:
                last_obj_created_time = max(
                    last_obj_created_time, max(obj.created_at for obj in objs)
                )

            if len(objs) > 0:
                log_verbose(
                    f"Crossmatching {len(objs)} objects took {time.time() - start_time:.2f} seconds"
                )
            else:
                log_verbose("No new objects to crossmatch. Waiting...")

        time.sleep(20)


if __name__ == "__main__":
    try:
        service()
    except Exception as e:
        log(f"Error in crossmatch_alert_to_localizations service: {e}")
        raise e
