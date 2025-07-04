import React, { useState } from "react";
import { useDispatch } from "react-redux";
import PropTypes from "prop-types";
import {
  createTheme,
  StyledEngineProvider,
  ThemeProvider,
  useTheme,
} from "@mui/material/styles";
import makeStyles from "@mui/styles/makeStyles";
import Paper from "@mui/material/Paper";
import DeleteIcon from "@mui/icons-material/Delete";
import IconButton from "@mui/material/IconButton";
import AddIcon from "@mui/icons-material/Add";
import Dialog from "@mui/material/Dialog";
import DialogTitle from "@mui/material/DialogTitle";
import DialogContent from "@mui/material/DialogContent";
import MUIDataTable from "mui-datatables";

import { showNotification } from "baselayer/components/Notifications";
import * as defaultSurveyEfficienciesActions from "../../ducks/default_survey_efficiencies";
import Button from "../Button";
import ConfirmDeletionDialog from "../ConfirmDeletionDialog";
import NewDefaultSurveyEfficiency from "./NewDefaultSurveyEfficiency";

const useStyles = makeStyles((theme) => ({
  container: {
    width: "100%",
    overflow: "scroll",
  },
  eventTags: {
    marginLeft: "0.5rem",
    "& > div": {
      margin: "0.25rem",
      color: "white",
      background: theme.palette.primary.main,
    },
  },
}));

// Tweak responsive styling
const getMuiTheme = (theme) =>
  createTheme({
    palette: theme.palette,
    components: {
      MUIDataTablePagination: {
        styleOverrides: {
          toolbar: {
            flexFlow: "row wrap",
            justifyContent: "flex-end",
            padding: "0.5rem 1rem 0",
            [theme.breakpoints.up("sm")]: {
              // Cancel out small screen styling and replace
              padding: "0px",
              paddingRight: "2px",
              flexFlow: "row nowrap",
            },
          },
          tableCellContainer: {
            padding: "1rem",
          },
          selectRoot: {
            marginRight: "0.5rem",
            [theme.breakpoints.up("sm")]: {
              marginLeft: "0",
              marginRight: "2rem",
            },
          },
        },
      },
    },
  });

const DefaultSurveyEfficiencyTable = ({
  default_survey_efficiencies,
  paginateCallback,
  totalMatches,
  sortingCallback,
  deletePermission,
}) => {
  const classes = useStyles();
  const theme = useTheme();

  const dispatch = useDispatch();

  const [setRowsPerPage] = useState(100);

  const [newDialogOpen, setNewDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [defaultSurveyEfficiencyToDelete, setDefaultSurveyEfficiencyToDelete] =
    useState(null);

  const openNewDialog = () => {
    setNewDialogOpen(true);
  };
  const closeNewDialog = () => {
    setNewDialogOpen(false);
  };

  const openDeleteDialog = (id) => {
    setDeleteDialogOpen(true);
    setDefaultSurveyEfficiencyToDelete(id);
  };
  const closeDeleteDialog = () => {
    setDeleteDialogOpen(false);
    setDefaultSurveyEfficiencyToDelete(null);
  };

  const deleteDefaultSurveyEfficiency = () => {
    dispatch(
      defaultSurveyEfficienciesActions.deleteDefaultSurveyEfficiency(
        defaultSurveyEfficiencyToDelete,
      ),
    ).then((result) => {
      if (result.status === "success") {
        dispatch(showNotification("Default survey efficiency deleted"));
        closeDeleteDialog();
      }
    });
  };

  const renderSurveyEfficiencyTitle = (dataIndex) => {
    const default_survey_efficiency = default_survey_efficiencies[dataIndex];

    return (
      <div>
        {
          default_survey_efficiency.default_observationplan_request
            .default_plan_name
        }
      </div>
    );
  };

  const renderModelName = (dataIndex) => {
    const default_survey_efficiency = default_survey_efficiencies[dataIndex];

    return (
      <div>
        {default_survey_efficiency
          ? default_survey_efficiency.payload.modelName
          : ""}
      </div>
    );
  };

  const renderMaxPhase = (dataIndex) => {
    const default_survey_efficiency = default_survey_efficiencies[dataIndex];

    return (
      <div>
        {default_survey_efficiency
          ? default_survey_efficiency.payload.maximumPhase
          : ""}
      </div>
    );
  };

  const renderMinPhase = (dataIndex) => {
    const default_survey_efficiency = default_survey_efficiencies[dataIndex];

    return (
      <div>
        {default_survey_efficiency
          ? default_survey_efficiency.payload.minimumPhase
          : ""}
      </div>
    );
  };

  const renderNumDetections = (dataIndex) => {
    const default_survey_efficiency = default_survey_efficiencies[dataIndex];

    return (
      <div>
        {default_survey_efficiency
          ? default_survey_efficiency.payload.numberDetections
          : ""}
      </div>
    );
  };

  const renderNumInjections = (dataIndex) => {
    const default_survey_efficiency = default_survey_efficiencies[dataIndex];

    return (
      <div>
        {default_survey_efficiency
          ? default_survey_efficiency.payload.numberInjections
          : ""}
      </div>
    );
  };

  const renderDetectionThreshold = (dataIndex) => {
    const default_survey_efficiency = default_survey_efficiencies[dataIndex];

    return (
      <div>
        {default_survey_efficiency
          ? default_survey_efficiency.payload.detectionThreshold
          : ""}
      </div>
    );
  };

  const renderLocCumprob = (dataIndex) => {
    const default_survey_efficiency = default_survey_efficiencies[dataIndex];

    return (
      <div>
        {default_survey_efficiency
          ? default_survey_efficiency.payload.localizationCumprob
          : ""}
      </div>
    );
  };

  const renderInjectionParameters = (dataIndex) => {
    const default_survey_efficiency = default_survey_efficiencies[dataIndex];

    return (
      <div>
        {default_survey_efficiency
          ? default_survey_efficiency.payload.optionalInjectionParameters
          : ""}
      </div>
    );
  };

  const renderDelete = (dataIndex) => {
    const default_survey_efficiency = default_survey_efficiencies[dataIndex];
    if (!deletePermission) {
      return null;
    }
    return (
      <div>
        <Button
          key={default_survey_efficiency.id}
          id="delete_button"
          classes={{
            root: classes.defaultSurveyEfficiencyDelete,
            disabled: classes.defaultSurveyEfficiencyDeleteDisabled,
          }}
          onClick={() => openDeleteDialog(default_survey_efficiency.id)}
          disabled={!deletePermission}
        >
          <DeleteIcon />
        </Button>
      </div>
    );
  };

  const handleTableChange = (action, tableState) => {
    switch (action) {
      case "changePage":
      case "changeRowsPerPage":
        setRowsPerPage(tableState.rowsPerPage);
        paginateCallback(
          tableState.page + 1,
          tableState.rowsPerPage,
          tableState.sortOrder,
        );
        break;
      case "sort":
        if (tableState.sortOrder.direction === "none") {
          paginateCallback(1, tableState.rowsPerPage, {});
        } else {
          sortingCallback(tableState.sortOrder);
        }
        break;
      default:
    }
  };

  const columns = [
    {
      name: "defaultSurveyEfficiency",
      label: "Default Plan",
      options: {
        filter: true,
        sort: true,
        sortThirdClickReset: true,
        customBodyRenderLite: renderSurveyEfficiencyTitle,
      },
    },
    {
      name: "modelName",
      label: "Model Name",
      options: {
        filter: false,
        sort: true,
        sortThirdClickReset: true,
        customBodyRenderLite: renderModelName,
      },
    },
    {
      name: "numInjections",
      label: "Number of Injections",
      options: {
        filter: false,
        sort: true,
        sortThirdClickReset: true,
        customBodyRenderLite: renderNumInjections,
      },
    },
    {
      name: "maxPhase",
      label: "Maximum Phase (days)",
      options: {
        filter: false,
        sort: true,
        sortThirdClickReset: true,
        customBodyRenderLite: renderMaxPhase,
      },
    },
    {
      name: "minPhase",
      label: "Minimum Phase (days)",
      options: {
        filter: false,
        sort: true,
        sortThirdClickReset: true,
        customBodyRenderLite: renderMinPhase,
      },
    },
    {
      name: "numDetections",
      label: "Number of Detections",
      options: {
        filter: false,
        sort: true,
        sortThirdClickReset: true,
        customBodyRenderLite: renderNumDetections,
      },
    },
    {
      name: "detectionThreshold",
      label: "Detection Threshold (sigma)",
      options: {
        filter: false,
        sort: true,
        sortThirdClickReset: true,
        customBodyRenderLite: renderDetectionThreshold,
      },
    },
    {
      name: "cumProb",
      label: "Cumulative Probability",
      options: {
        filter: false,
        sort: true,
        sortThirdClickReset: true,
        customBodyRenderLite: renderLocCumprob,
      },
    },
    {
      name: "injectionParameters",
      label: "Optional Injection Parameters",
      options: {
        filter: false,
        sort: true,
        sortThirdClickReset: true,
        customBodyRenderLite: renderInjectionParameters,
      },
    },
    {
      name: "delete",
      label: " ",
      options: {
        customBodyRenderLite: renderDelete,
      },
    },
  ];

  const options = {
    search: false,
    selectableRows: "none",
    elevation: 0,
    onTableChange: handleTableChange,
    jumpToPage: true,
    serverSide: true,
    pagination: false,
    count: totalMatches,
    filter: true,
    sort: true,
    customToolbar: () => (
      <IconButton
        name="new_default_survey_efficiency"
        onClick={() => {
          openNewDialog();
        }}
      >
        <AddIcon />
      </IconButton>
    ),
  };

  return (
    <div>
      <Paper className={classes.container}>
        <StyledEngineProvider injectFirst>
          <ThemeProvider theme={getMuiTheme(theme)}>
            <MUIDataTable
              title="Default Survey Efficiencies"
              data={default_survey_efficiencies}
              options={options}
              columns={columns}
            />
          </ThemeProvider>
        </StyledEngineProvider>
      </Paper>
      {newDialogOpen && (
        <Dialog
          open={newDialogOpen}
          onClose={closeNewDialog}
          style={{ position: "fixed" }}
          maxWidth="md"
        >
          <DialogTitle>New Default Survey Efficiency</DialogTitle>
          <DialogContent dividers>
            <NewDefaultSurveyEfficiency onClose={closeNewDialog} />
          </DialogContent>
        </Dialog>
      )}
      <ConfirmDeletionDialog
        deleteFunction={deleteDefaultSurveyEfficiency}
        dialogOpen={deleteDialogOpen}
        closeDialog={closeDeleteDialog}
        resourceName="default survey efficiency"
      />
    </div>
  );
};

DefaultSurveyEfficiencyTable.propTypes = {
  // eslint-disable-next-line react/forbid-prop-types
  default_survey_efficiencies: PropTypes.arrayOf(PropTypes.any).isRequired,
  paginateCallback: PropTypes.func.isRequired,
  sortingCallback: PropTypes.func,
  totalMatches: PropTypes.number,
  deletePermission: PropTypes.bool,
};

DefaultSurveyEfficiencyTable.defaultProps = {
  totalMatches: 0,
  sortingCallback: null,
  deletePermission: false,
};

export default DefaultSurveyEfficiencyTable;
