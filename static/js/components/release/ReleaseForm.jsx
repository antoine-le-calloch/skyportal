import React from "react";
import { useDispatch, useSelector } from "react-redux";
import Form from "@rjsf/mui";
import validator from "@rjsf/validator-ajv8";
import PropTypes from "prop-types";
import { sourcePublishOptionsSchema } from "../source/source_publish/SourcePublishOptions";
import {
  submitPublicRelease,
  updatePublicRelease,
} from "../../ducks/public_pages/public_release";
import Button from "../Button";

const ReleaseForm = ({ release, setRelease, setIsSubmit }) => {
  const dispatch = useDispatch();
  const streams = useSelector((state) => state.streams);
  const groups = useSelector((state) => state.groups.userAccessible);
  const sourceOptionsSchema = sourcePublishOptionsSchema(streams, groups);
  const releaseSchema = {
    type: "object",
    properties: {
      name: {
        type: "string",
        title: "Name",
      },
      link_name: {
        type: "string",
        title: "Link name to use in the URL",
      },
      description: {
        type: "string",
        title: "Description",
      },
      group_ids: {
        type: "array",
        items: {
          type: "integer",
          anyOf: groups.map((group) => ({
            enum: [group.id],
            type: "integer",
            title: group.name,
          })),
        },
        uniqueItems: true,
        default: [],
        title: "Groups that can manage this release",
      },
      visible: {
        type: "boolean",
        title: "Visible",
        default: true,
      },
      options: {
        title: "Options for the sources in this release",
        ...sourceOptionsSchema,
      },
    },
    required: ["name", "link_name"],
  };

  const submitRelease = () => {
    const action = release.id
      ? updatePublicRelease(release.id, release)
      : submitPublicRelease(release);
    dispatch(action).then((response) => {
      if (response.status === "success") {
        setIsSubmit(true);
      }
    });
  };

  return (
    <Form
      formData={release}
      onChange={({ formData }) => setRelease(formData)}
      schema={releaseSchema}
      validator={validator}
      onSubmit={submitRelease}
      uiSchema={{
        description: {
          "ui:widget": "textarea",
          "ui:options": {
            rows: 3,
          },
        },
      }}
    >
      <div style={{ display: "flex", justifyContent: "center" }}>
        <Button primary type="submit">
          Submit
        </Button>
      </div>
    </Form>
  );
};

ReleaseForm.propTypes = {
  release: PropTypes.shape({
    id: PropTypes.number,
    name: PropTypes.string,
    description: PropTypes.string,
    groups: PropTypes.arrayOf(PropTypes.number),
    visible: PropTypes.bool,
    options: PropTypes.shape({
      include_photometry: PropTypes.bool,
      include_classifications: PropTypes.bool,
      streams: PropTypes.arrayOf(PropTypes.number),
      groups: PropTypes.arrayOf(PropTypes.number),
    }),
  }).isRequired,
  setRelease: PropTypes.func.isRequired,
  setIsSubmit: PropTypes.func.isRequired,
};

export default ReleaseForm;