import { useState, type ReactNode } from "react";
import Box from "@mui/material/Box";
import CircularProgress from "@mui/material/CircularProgress";
import IconButton from "@mui/material/IconButton";
import Tooltip from "@mui/material/Tooltip";
import CloudDownloadOutlinedIcon from "@mui/icons-material/CloudDownloadOutlined";
import DeleteOutlinedIcon from "@mui/icons-material/DeleteOutlined";
import EditOutlinedIcon from "@mui/icons-material/EditOutlined";
import SendOutlinedIcon from "@mui/icons-material/SendOutlined";

import { showNotification } from "baselayer/components/Notifications";
import { useAppDispatch } from "../../types/hooks";
import {
  useDeleteFollowupRequestMutation,
  useEditFollowupRequestMutation,
  useLazyGetPhotometryRequestQuery,
} from "../../ducks/source";

const ActionButton = ({
  title,
  loading = false,
  onClick,
  testId,
  children,
}: {
  title: string;
  loading?: boolean;
  onClick: () => void;
  testId?: string;
  children: ReactNode;
}) =>
  loading ? (
    <CircularProgress size={18} sx={{ m: "5px" }} />
  ) : (
    <Tooltip title={title}>
      <IconButton size="small" onClick={onClick} data-testid={testId}>
        {children}
      </IconButton>
    </Tooltip>
  );

interface FollowupRequestActionsProps {
  request: any;
  methodsImplemented: { get?: boolean; delete?: boolean; submit?: boolean };
  editable: boolean;
  refresh: boolean;
  onEdit: (id: number) => void;
}

const FollowupRequestActions = ({
  request,
  methodsImplemented,
  editable,
  refresh,
  onEdit,
}: FollowupRequestActionsProps) => {
  const dispatch = useAppDispatch();
  const [deleteFollowupRequest] = useDeleteFollowupRequestMutation();
  const [editFollowupRequest] = useEditFollowupRequestMutation();
  const [getPhotometryRequest] = useLazyGetPhotometryRequestQuery();
  const [loading, setLoading] = useState<string | null>(null);
  const [hasRetrieved, setHasRetrieved] = useState(false);
  const params = refresh ? { refreshRequests: true } : {};
  const { id, status } = request;
  const isDeleted = status === "deleted";

  const run = async (action: string, callback: () => Promise<unknown>) => {
    setLoading(action);
    await callback();
    setLoading(null);
  };

  const retrieve = async () => {
    const { data, error }: any = await getPhotometryRequest({ id, params });
    if (error) return;
    dispatch(
      data?.request_status?.includes("rejected")
        ? showNotification("Request has been rejected.", "warning")
        : showNotification(
            "Request successfully submitted, please wait for it to be processed.",
            "info",
          ),
    );
    setHasRetrieved(true);
  };

  const canRetrieve =
    methodsImplemented.get &&
    !hasRetrieved &&
    status !== "Photometry committed to database" &&
    (status.startsWith("pending") || status.startsWith("submitted"));

  return (
    <Box sx={{ display: "flex", alignItems: "center" }}>
      {methodsImplemented.delete && !isDeleted && (
        <ActionButton
          title="Delete"
          loading={loading === "delete"}
          onClick={() =>
            run("delete", () => deleteFollowupRequest({ id, params }))
          }
          testId={`deleteRequest_${id}`}
        >
          <DeleteOutlinedIcon fontSize="small" />
        </ActionButton>
      )}
      {canRetrieve && (
        <ActionButton
          title="Retrieve"
          loading={loading === "retrieve"}
          onClick={() => run("retrieve", retrieve)}
        >
          <CloudDownloadOutlinedIcon fontSize="small" />
        </ActionButton>
      )}
      {methodsImplemented.submit && status.includes("failed to submit") && (
        <ActionButton
          title="Submit again"
          loading={loading === "submit"}
          onClick={() =>
            run("submit", () =>
              editFollowupRequest({
                params: {
                  allocation_id: request.allocation.id,
                  obj_id: request.obj_id,
                  payload: request.payload,
                  ...params,
                },
                requestID: id,
              }),
            )
          }
        >
          <SendOutlinedIcon fontSize="small" />
        </ActionButton>
      )}
      {editable && !isDeleted && (
        <ActionButton
          title="Edit"
          onClick={() => onEdit(id)}
          testId={`editRequest_${id}`}
        >
          <EditOutlinedIcon fontSize="small" />
        </ActionButton>
      )}
    </Box>
  );
};

export default FollowupRequestActions;
