import numpy as np
import torchvision

decord = None
try:
    import decord

    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False


def get_frames_by_timestamps(
    video_path: str,
    timestamps: list[float] | np.ndarray,
    video_backend: str = "ffmpeg",
    video_backend_kwargs: dict | None = {},
    fps: None | float = None,
) -> np.ndarray:
    """Get frames from a video at specified timestamps.

    Args:
        video_path (str): Path to the video file.
        timestamps (list[int] | np.ndarray): Timestamps to retrieve frames for, in seconds.
        video_backend (str, optional): Video backend to use. Defaults to "ffmpeg".
        fps (float, optional): FPS of the video. Defaults to 30.
    Returns:
        np.ndarray: Frames at the specified timestamps.

    Only supports pyav
    """
    if video_backend == "decord":
        if not decord:
            raise ImportError("decord is not available. Install it with: pip install decord")
        vr = decord.VideoReader(video_path, **(video_backend_kwargs or {}))
        num_frames = len(vr)
        # Retrieve the timestamps for each frame in the video
        frame_ts: np.ndarray = vr.get_frame_timestamp(range(num_frames))
        # Map each requested timestamp to the closest frame index
        # Only take the first element of the frame_ts array which corresponds to start_seconds
        indices = np.abs(frame_ts[:, :1] - timestamps).argmin(axis=0)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    
    elif video_backend == "torchvision_av" or video_backend == "pyav":
        # set backend
        torchvision.set_video_backend("pyav")

        # set a video stream reader
        reader = torchvision.io.VideoReader(video_path, "video")

        # set the first and last requested timestamps
        # Note: previous timestamps are usually loaded, since we need to access the previous key frame
        first_ts = timestamps[0]
        last_ts = timestamps[-1]

        # access closest key frame of the first requested frame
        # Note: closest key frame timestamp is usally smaller than `first_ts` (e.g. key frame can be the first frame of the video)
        # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
        reader.seek(first_ts, keyframes_only=True)

        # Decode frames sequentially, storing the ones we need in a dictionary
        # to map timestamps to frame data. This allows for easy re-ordering later.
        found_frames_map = {}
        tolerance = 0.001  # 1ms tolerance for timestamp matching

        for frame in reader:
            current_ts = frame["pts"]

            # Use tolerance-based matching instead of exact match
            for ts in timestamps:
                if ts not in found_frames_map and abs(current_ts - ts) < tolerance:
                    found_frames_map[ts] = frame["data"]
                    break

            if current_ts >= last_ts + tolerance or len(found_frames_map) == len(
                timestamps
            ):
                break

        reader.container.close()
        reader = None

        # Debug: print timestamp matching results
        #print(
        #    f"[video_utils] Requested {len(timestamps)} timestamps: {timestamps[:4]}{'...' if len(timestamps) > 4 else ''}"
        #)
        #print(
        #    f"[video_utils] Found {len(found_frames_map)} frames with tolerance={tolerance}s"
        #)
        #if len(found_frames_map) < len(timestamps):
        #    missing = [ts for ts in timestamps if ts not in found_frames_map]
        #    print(
        #        f"[video_utils] WARNING: Missing timestamps: {missing[:4]}{'...' if len(missing) > 4 else ''}"
        #    )

        frames = np.array(list(found_frames_map.values()))
        return frames.transpose(0, 2, 3, 1)

    else:
        raise NotImplementedError
