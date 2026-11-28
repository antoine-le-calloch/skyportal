import { useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

import StyledDataGrid, { DataGridToolbar } from "../StyledDataGrid";
import JsonCell from "../JsonCell";
import EditFollowupRequestDialog from "./EditFollowupRequestDialog";
import FollowupRequestActions from "./FollowupRequestActions";
import WatcherButton from "./WatcherButton";
import {
  PLAIN_LINKS_SX,
  allocationColumn,
  groupColumn,
  requestedColumn,
  requesterColumn,
  statusColumn,
  transactionsColumn,
} from "./columns";

const isObject = (value: any) => value !== null && typeof value === "object";

const DISPLAYED_COLUMNS = [
  "requester",
  "allocation",
  "start_date",
  "end_date",
  "mode",
  "request",
  "filter",
  "filters",
  "field_ids",
  "priority",
  "status",
  "modify",
  "watch",
];

const KEY_RANK: Record<string, number> = {
  start_date: 0,
  end_date: 1,
  observation_type: 2,
  priority: 4,
  status: 5,
};

const keyOrder = (a: string, b: string) =>
  (KEY_RANK[a] ?? 3) - (KEY_RANK[b] ?? 3) || (a < b ? -1 : Number(a > b));

const FollowupRequestListToolbar = () => (
  <DataGridToolbar showExport={false} showExpandAll />
);

interface FollowupRequestListsProps {
  followupRequests: any[];
  instrumentList: any[];
  instrumentFormParams: any;
  requestType?: string;
  showEmptyMessage?: boolean;
}

const FollowupRequestLists = ({
  followupRequests,
  instrumentList,
  instrumentFormParams,
  requestType = "triggered",
  showEmptyMessage = true,
}: FollowupRequestListsProps) => {
  // Not in the cell: its DataGrid column is virtualized, a remount would close the dialog.
  const [requestIdToEdit, setRequestIdToEdit] = useState<number | null>(null);
  const [columnVisibilityModels, setColumnVisibilityModels] = useState<
    Record<string, any>
  >({});

  const [schema, otherSchema] =
    requestType === "triggered"
      ? ["formSchema", "formSchemaForcedPhotometry"]
      : ["formSchemaForcedPhotometry", "formSchema"];
  const instruments = instrumentList.filter(
    (inst) => instrumentFormParams[inst.id]?.[schema] != null,
  );
  const requests = followupRequests.filter(
    (request) =>
      request?.payload?.request_type === requestType ||
      (request?.allocation?.instrument_id in instrumentFormParams &&
        instrumentFormParams[request.allocation.instrument_id]?.[otherSchema] ==
          null),
  );

  if (!requests.length || !instruments.length) {
    return showEmptyMessage ? (
      <Typography variant="body2" sx={{ color: "text.secondary" }}>
        No follow-up requests found.
      </Typography>
    ) : null;
  }

  const instLookUp = Object.fromEntries(
    instruments.map((inst: any) => [inst.id, inst]),
  );
  const requestsGroupedByInstId = requests.reduce(
    (grouped: any, request: any) => {
      (grouped[request.allocation.instrument.id] ||= []).push(request);
      return grouped;
    },
    {},
  );

  const getDataTableColumns = (keys: string[], instrument_id: string) => {
    const baseColumns = [
      statusColumn,
      requestedColumn,
      requesterColumn,
      groupColumn,
      allocationColumn,
    ];
    const formParams = instrumentFormParams[instrument_id];
    if (!formParams) return { columns: baseColumns, defaultVisibility: {} };
    const defaultVisibility: Record<string, boolean> = { Transactions: false };
    const { methodsImplemented } = formParams;
    const editable = methodsImplemented.update && requestType === "triggered";
    const columns: any[] = [
      {
        field: "actions",
        headerName: "Actions",
        renderHeader: () => null,
        width: 120,
        hideable: false,
        sortable: false,
        filterable: false,
        disableColumnMenu: true,
        renderCell: ({ row }: any) => (
          <Box sx={{ display: "flex", alignItems: "center", height: "100%" }}>
            <WatcherButton followupRequest={row} />
            <FollowupRequestActions
              request={row}
              methodsImplemented={methodsImplemented}
              editable={editable}
              refresh={false}
              onEdit={setRequestIdToEdit}
            />
          </Box>
        ),
      },
      ...baseColumns,
    ];

    if (formParams.formSchema?.properties?.station_name) {
      columns.push({
        field: "station",
        headerName: "Station",
        flex: 1,
        minWidth: 120,
        sortable: false,
        filterable: false,
        valueGetter: (_value: any, row: any) => row?.payload?.station_name,
      });
    }

    keys.forEach((key) => {
      const headerName = formParams.aliasLookup[key] ?? key;
      const field = `payload.${key}`;
      columns.push({
        field,
        headerName,
        flex: 1,
        minWidth: 120,
        sortable: false,
        filterable: false,
        valueGetter: (_value: any, row: any) => {
          const value = row.payload?.[key];
          return Array.isArray(value) && !value.some(isObject)
            ? value.join(",")
            : value;
        },
        renderCell: ({ value }: any) =>
          isObject(value) ? <JsonCell data={value} /> : String(value ?? ""),
      });
      if (!DISPLAYED_COLUMNS.includes(headerName.toLowerCase())) {
        defaultVisibility[field] = false;
      }
    });

    columns.push(transactionsColumn);
    return { columns, defaultVisibility };
  };

  const requestToEdit = requests.find(
    (request: any) => request.id === requestIdToEdit,
  );

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      {requestToEdit && (
        <EditFollowupRequestDialog
          followupRequest={requestToEdit}
          instrumentFormParams={instrumentFormParams}
          onClose={() => setRequestIdToEdit(null)}
          requestType={requestType}
        />
      )}
      {Object.entries(requestsGroupedByInstId).map(
        ([instrument_id, instrumentRequests]: [string, any]) => {
          const keys = [
            ...new Set<string>(
              instrumentRequests.flatMap((request: any) =>
                Object.keys(request.payload),
              ),
            ),
          ].sort(keyOrder);
          const { columns, defaultVisibility } = getDataTableColumns(
            keys,
            instrument_id,
          );

          return (
            <Box key={instrument_id}>
              <Box
                data-testid={`${instrument_id}-requests-header`}
                sx={{
                  display: "flex",
                  alignItems: "baseline",
                  gap: 1,
                  mb: 0.75,
                }}
              >
                <Typography sx={{ fontWeight: 600 }}>
                  {instLookUp[instrument_id]?.name}
                </Typography>
                <Typography variant="body2" sx={{ color: "text.secondary" }}>
                  {instrumentRequests.length}{" "}
                  {instrumentRequests.length === 1 ? "request" : "requests"}
                </Typography>
              </Box>
              <Box data-testid={`${instrument_id}_followupRequestsTable`}>
                <StyledDataGrid
                  height="auto"
                  getRowHeight={() => "auto"}
                  sx={PLAIN_LINKS_SX}
                  rows={instrumentRequests}
                  columns={columns}
                  columnVisibilityModel={
                    columnVisibilityModels[instrument_id] ?? defaultVisibility
                  }
                  onColumnVisibilityModelChange={(model: any) =>
                    setColumnVisibilityModels((prev) => ({
                      ...prev,
                      [instrument_id]: model,
                    }))
                  }
                  slots={{ toolbar: FollowupRequestListToolbar }}
                  showToolbar
                  hideFooter={instrumentRequests.length <= 25}
                />
              </Box>
            </Box>
          );
        },
      )}
    </Box>
  );
};

export default FollowupRequestLists;
