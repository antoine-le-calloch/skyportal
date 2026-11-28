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
  const [columnVisibilityModel, setColumnVisibilityModel] = useState<Record<
    string,
    boolean
  > | null>(null);

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
  const rows = [...requests].sort((a, b) =>
    a.created_at < b.created_at ? 1 : -1,
  );
  const instrumentIds = [
    ...new Set<number>(rows.map((row) => row.allocation.instrument_id)),
  ];
  const keys = [
    ...new Set<string>(rows.flatMap((row) => Object.keys(row.payload))),
  ].sort(keyOrder);
  const formParamsList = instrumentIds
    .map((id) => instrumentFormParams[id])
    .filter(Boolean);

  const defaultVisibility: Record<string, boolean> = { Transactions: false };
  const columns: any[] = [
    {
      field: "actions",
      headerName: "Actions",
      renderHeader: () => null,
      width: 150,
      hideable: false,
      sortable: false,
      filterable: false,
      disableColumnMenu: true,
      renderCell: ({ row }: any) => {
        const methodsImplemented =
          instrumentFormParams[row.allocation.instrument_id]
            ?.methodsImplemented ?? {};
        return (
          <Box sx={{ display: "flex", alignItems: "center", height: "100%" }}>
            <WatcherButton followupRequest={row} />
            <FollowupRequestActions
              request={row}
              methodsImplemented={methodsImplemented}
              editable={
                !!methodsImplemented.update && requestType === "triggered"
              }
              refresh={false}
              onEdit={setRequestIdToEdit}
            />
          </Box>
        );
      },
    },
    {
      field: "instrument",
      headerName: "Instrument",
      minWidth: 110,
      hideable: false,
      valueGetter: (_value: any, row: any) =>
        instLookUp[row.allocation.instrument_id]?.name ??
        row.allocation.instrument?.name,
      renderCell: ({ value }: any) => (
        <Box component="span" sx={{ fontWeight: 600 }}>
          {value}
        </Box>
      ),
    },
    statusColumn,
    requestedColumn,
    requesterColumn,
    groupColumn,
    allocationColumn,
  ];

  if (
    formParamsList.some((params) => params.formSchema?.properties?.station_name)
  ) {
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
    const headerName =
      formParamsList.map((params) => params.aliasLookup?.[key]).find(Boolean) ??
      key;
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

  const requestToEdit = requests.find(
    (request: any) => request.id === requestIdToEdit,
  );

  return (
    <Box
      data-testid={instrumentIds
        .map((id) => `${id}_followupRequestsTable`)
        .join(" ")}
    >
      {requestToEdit && (
        <EditFollowupRequestDialog
          followupRequest={requestToEdit}
          instrumentFormParams={instrumentFormParams}
          onClose={() => setRequestIdToEdit(null)}
          requestType={requestType}
        />
      )}
      <StyledDataGrid
        height="auto"
        density="compact"
        getRowHeight={() => "auto"}
        sx={[
          PLAIN_LINKS_SX,
          {
            "& .MuiDataGrid-row--dynamicHeight > .MuiDataGrid-cell": {
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            },
          },
        ]}
        rows={rows}
        columns={columns}
        columnVisibilityModel={columnVisibilityModel ?? defaultVisibility}
        onColumnVisibilityModelChange={setColumnVisibilityModel}
        initialState={{ pagination: { paginationModel: { pageSize: 5 } } }}
        pageSizeOptions={[5, 10, 25]}
        hideFooter={rows.length <= 5}
        slots={{ toolbar: FollowupRequestListToolbar }}
        showToolbar
      />
    </Box>
  );
};

export default FollowupRequestLists;
