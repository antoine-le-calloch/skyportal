import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import PropTypes from "prop-types";
import Form from "@rjsf/mui";
import validator from "@rjsf/validator-ajv8";
import makeStyles from "@mui/styles/makeStyles";
import CircularProgress from "@mui/material/CircularProgress";
import Link from "@mui/material/Link";
import { fetchPublicReleases } from "../../../ducks/public_pages/public_release";
import ReleasesList from "../../release/ReleasesList";

const useStyles = makeStyles(() => ({
  sourcePublishRelease: {
    marginBottom: "1rem",
    display: "flex",
    flexDirection: "column",
    padding: "0 0.3rem",
    "& .MuiGrid-item": {
      paddingTop: "0",
    },
  },
  noRelease: {
    display: "flex",
    justifyContent: "center",
    padding: "1.5rem 0",
    color: "gray",
  },
  releaseItem: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "0.5rem 1rem",
    border: "1px solid #e0e0e0",
  },
}));

const SourcePublishRelease = ({
  sourceReleaseId,
  setSourceReleaseId,
  setOptions,
}) => {
  const styles = useStyles();
  const dispatch = useDispatch();
  const [loading, setLoading] = useState(true);
  const releases = useSelector((state) => state.publicReleases);
  const manageSourcesAccess = useSelector(
    (state) => state.profile,
  ).permissions?.includes("Manage sources");

  useEffect(() => {
    setLoading(true);
    dispatch(fetchPublicReleases()).then(() => setLoading(false));
  }, [dispatch]);

  const formSchema = {
    type: "object",
    properties: {
      release: {
        type: "integer",
        title: "Release",
        oneOf: releases.map((item) => ({
          enum: [item.id],
          title: item.name,
        })),
      },
    },
  };

  const handleReleaseChange = (data) => {
    setSourceReleaseId(data.release);
    if (releases.length > 0 && data.release) {
      setOptions(releases.find((item) => item.id === data.release).options);
    }
  };

  return (
    <div className={styles.sourcePublishRelease}>
      <div style={{ display: "flex", justifyContent: "end" }}>
        <Link
          href="/public/releases"
          target="_blank"
          style={{ fontSize: "0.7rem" }}
        >
          Public releases
        </Link>
      </div>
      {manageSourcesAccess && (
        <>
          {releases.length > 0 ? (
            <Form
              formData={
                sourceReleaseId ? { release: sourceReleaseId } : undefined
              }
              onChange={({ formData }) => handleReleaseChange(formData)}
              schema={formSchema}
              liveValidate
              validator={validator}
              uiSchema={{
                release: {
                  "ui:placeholder": "Choose an option",
                },
                "ui:submitButtonOptions": { norender: true },
              }}
            />
          ) : (
            <div className={styles.noRelease}>
              {loading ? (
                <CircularProgress size={24} />
              ) : (
                <div>No releases available yet! Create the first one here.</div>
              )}
            </div>
          )}
        </>
      )}
      <ReleasesList />
    </div>
  );
};

SourcePublishRelease.propTypes = {
  sourceReleaseId: PropTypes.number,
  setSourceReleaseId: PropTypes.func.isRequired,
  setOptions: PropTypes.func.isRequired,
};

SourcePublishRelease.defaultProps = {
  sourceReleaseId: null,
};

export default SourcePublishRelease;
