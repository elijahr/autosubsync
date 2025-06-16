import os
import sys
import tempfile
import subprocess
import numpy as np
import requests
from . import srt_io


def is_http_url(path):
    """Check if a path is an HTTP or HTTPS URL"""
    return isinstance(path, str) and (path.startswith("http://") or path.startswith("https://"))


def validate_remote_video_url(url, verbose=False):
    """
    Validate that a remote URL points to a video file, not a webpage.

    Args:
        url (str): Remote video URL to validate
        verbose (bool): Enable verbose output

    Returns:
        bool: True if URL appears to be a valid video file, False otherwise

    Raises:
        RuntimeError: If URL validation fails with helpful error message
    """
    if not is_http_url(url):
        return True  # Local files are assumed valid

    try:
        response = requests.head(url, timeout=30, allow_redirects=True)

        if response.status_code == 404:
            raise RuntimeError(f"Video URL not found (404): {url}")
        elif response.status_code != 200:
            raise RuntimeError(f"Video URL returned status {response.status_code}: {url}")

        content_type = response.headers.get("Content-Type", "").lower()

        # Check if this is an HTML page instead of a video file
        if "text/html" in content_type:
            error_msg = f"URL points to a webpage, not a video file: {url}\n"
            if "archive.org" in url:
                error_msg += "For Archive.org files, you need the direct download URL, not the details page.\n"
                error_msg += "Try finding the direct download link on the Archive.org page."
            else:
                error_msg += "Please provide a direct video file URL, not a webpage URL."
            raise RuntimeError(error_msg)

        # Check if content type suggests a video file
        video_types = ["video/", "application/octet-stream", "binary/octet-stream"]
        if not any(vtype in content_type for vtype in video_types) and content_type:
            if verbose:
                print(f"Warning: Content-Type '{content_type}' may not be a video file")

        if verbose:
            print(f"URL validation passed for: {url}")
            print(f"Content-Type: {content_type}")

        return True

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to validate video URL: {url}\nError: {e}")


def get_video_duration(video_path_or_url, verbose=False):
    """Get the duration of a video file in seconds using ffprobe"""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path_or_url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as e:
        if verbose:
            print(f"Warning: Could not determine video duration: {e}")
    return None


def extract_audio_chunks_from_remote_video(url, output_dir, max_chunks=5, max_chunk_duration=60, verbose=False):
    """
    Extract audio chunks directly from a remote video URL using ffmpeg.
    This avoids downloading the entire video file.

    Args:
        url (str): Remote video URL
        output_dir (str): Directory to save audio chunks
        max_chunks (int): Maximum number of chunks to extract
        max_chunk_duration (int): Maximum duration per chunk in seconds
        verbose (bool): Enable verbose output

    Returns:
        list: Paths to extracted audio chunks
    """
    if verbose:
        print(f"Extracting audio chunks from remote video: {url}")

    # First, try to get the video duration directly from the URL
    duration = get_video_duration(url, verbose=verbose)

    if duration is None:
        if verbose:
            print("Could not determine video duration, using default sampling")
        # If we can't get duration, use predefined time points
        # Sample at common points: beginning, middle, end
        chunk_times = [
            (30, min(max_chunk_duration, 60)),  # 30s from start
            (300, min(max_chunk_duration, 60)),  # 5 minutes from start
            (900, min(max_chunk_duration, 60)),  # 15 minutes from start
        ]
        chunk_times = chunk_times[:max_chunks]
    else:
        if verbose:
            print(f"Video duration: {duration:.1f} seconds")

        # Calculate chunk positions based on actual duration
        chunk_times = []
        if duration <= max_chunk_duration:
            # If video is short, just use the whole thing
            chunk_times = [(0, min(duration, max_chunk_duration))]
        else:
            # Distribute chunks across the video duration
            interval = duration / (max_chunks + 1)
            for i in range(max_chunks):
                start_time = interval * (i + 1)
                chunk_duration = min(max_chunk_duration, duration - start_time)
                if chunk_duration > 10:  # Only include chunks with at least 10 seconds
                    chunk_times.append((start_time, chunk_duration))

    chunk_paths = []

    for i, (start_time, chunk_duration) in enumerate(chunk_times):
        chunk_path = os.path.join(output_dir, f"chunk_{i+1}.flac")

        if verbose:
            print(f"Extracting chunk {i+1}: {start_time:.1f}s - {start_time + chunk_duration:.1f}s")

        cmd = [
            "ffmpeg",
            "-y",  # overwrite if exists
            "-loglevel",
            "error" if not verbose else "info",
            "-ss",
            str(start_time),  # start time
            "-i",
            url,  # input URL
            "-t",
            str(chunk_duration),  # duration
            "-vn",  # no video
            "-sn",  # no subtitles
            "-ac",
            "1",  # convert to mono
            chunk_path,
        ]

        try:
            result = subprocess.run(cmd, timeout=300)  # 5 minute timeout per chunk
            if result.returncode == 0 and os.path.exists(chunk_path):
                chunk_paths.append(chunk_path)
                if verbose:
                    print(f"Successfully extracted chunk {i+1}")
            else:
                if verbose:
                    print(f"Failed to extract chunk {i+1}")
        except subprocess.TimeoutExpired:
            if verbose:
                print(f"Timeout extracting chunk {i+1}, skipping...")
            continue
        except Exception as e:
            if verbose:
                print(f"Error extracting chunk {i+1}: {e}")
            continue

    if not chunk_paths:
        raise RuntimeError("Failed to extract any audio chunks from remote video")

    if verbose:
        print(f"Successfully extracted {len(chunk_paths)} audio chunks")

    return chunk_paths


def combine_audio_chunks(chunk_paths, output_path, verbose=False):
    """
    Combine multiple audio chunks into a single audio file.

    Args:
        chunk_paths (list): List of paths to audio chunk files
        output_path (str): Path for the combined output file
        verbose (bool): Enable verbose output
    """
    if not chunk_paths:
        raise ValueError("No audio chunks to combine")

    if len(chunk_paths) == 1:
        # If only one chunk, just copy it
        import shutil

        shutil.copy2(chunk_paths[0], output_path)
        if verbose:
            print("Single chunk, copied directly")
        return

    if verbose:
        print(f"Combining {len(chunk_paths)} audio chunks")

    # Create a temporary file list for ffmpeg
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for chunk_path in chunk_paths:
            f.write(f"file '{chunk_path}'\n")
        filelist_path = f.name

    try:
        cmd = [
            "ffmpeg",
            "-y",  # overwrite if exists
            "-loglevel",
            "error" if not verbose else "info",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            filelist_path,
            "-c",
            "copy",
            output_path,
        ]

        result = subprocess.run(cmd, timeout=60)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)

        if verbose:
            print("Successfully combined audio chunks")

    finally:
        # Clean up the file list
        try:
            os.unlink(filelist_path)
        except:
            pass


def download_remote_video_with_ffmpeg(url, output_path, verbose=False):
    """
    Download a remote video file using ffmpeg.
    This is a fallback method when chunked extraction fails.
    """
    if verbose:
        print(f"Downloading remote video from {url} using ffmpeg...")

    cmd = [
        "ffmpeg",
        "-y",  # overwrite if exists
        "-loglevel",
        "error" if not verbose else "info",
        "-i",
        url,  # input URL
        "-c",
        "copy",  # copy streams without re-encoding
        output_path,
    ]

    try:
        result = subprocess.run(cmd, timeout=1800)  # 30 minute timeout
        if result.returncode == 0:
            if verbose:
                print(f"Download complete. File saved to {output_path}")
            return True
        else:
            if verbose:
                print(f"ffmpeg download failed with return code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        if verbose:
            print("Download timed out after 30 minutes")
        return False
    except Exception as e:
        if verbose:
            print(f"Download failed with error: {e}")
        return False


def import_sound(sound_path):
    import soundfile

    target_sample_rate = 20000
    warning_threshold = 8000

    # minimize memory usage by reading audio as 16-bit integers instead of
    # float or double. This should be the maximum precision of the samples
    # anyway
    samples, sample_rate = soundfile.read(sound_path, dtype="int16")
    data_range = 2**15

    # even 8-bit audio would work here, but already affects performance
    # samples, data_range = (samples / 256).astype(np.int8), 128

    ds_factor = max(int(np.floor(sample_rate / target_sample_rate)), 1)
    samples = samples[::ds_factor]
    sample_rate = sample_rate / ds_factor

    if sample_rate < warning_threshold:
        # probably never happens but checking anyway
        sys.stderr.write("warning: low sound sample rate %d Hz\n" % sample_rate)

    return samples, sample_rate, data_range


def build_sub_vec(subs, sample_rate, n, sub_filter=None):
    subvec = np.zeros(n, bool)
    to_index = lambda x: int(sample_rate * x)
    for line in subs:
        if sub_filter is not None and not sub_filter(line.text):
            continue
        subvec[to_index(line.begin) : to_index(line.end)] = 1
    return subvec


def import_subs(srt_filename, sample_rate, n, **kwargs):
    audio_length = n / float(sample_rate)
    subs = list(srt_io.read_file(srt_filename))
    if len(subs) > 0:
        subs_length = np.max([s.end for s in subs])
        rel_err = abs(subs_length - audio_length) / max(subs_length, audio_length)
        if rel_err > 0.25:  # warning threshold
            sys.stderr.write(
                " *** WARNING: subtitle and audio lengths "
                + "differ by %d%%. Wrong subtitle file?\n" % int(rel_err * 100)
            )

    else:
        sys.stderr.write(" *** WARNING: empty subtitle file\n")
    return build_sub_vec(subs, sample_rate, n, **kwargs)


def import_item(sound_file, subtitle_file, **kwargs):
    sound_data = import_sound(sound_file)
    samples, sample_rate, data_range = sound_data
    n = len(samples)
    sub_vec = import_subs(subtitle_file, sample_rate, n, **kwargs)
    return sound_data, sub_vec


def extract_sound(input_video_file, output_sound_file):
    convert_cmd = [
        "ffmpeg",
        "-y",  # overwrite if exists
        "-loglevel",
        "error",
        "-i",
        input_video_file,  # input
        "-vn",  # no video
        "-sn",  # no subtitles
        "-ac",
        "1",  # convert to mono
        output_sound_file,
    ]
    subprocess.call(convert_cmd)


def import_target_files(
    video_file,
    subtitle_file,
    verbose=False,
    anchor_points=None,
    anchor_duration_mins=None,
    anchor_duration_secs=None,
    **kwargs,
):
    "Import prediction target files using a temporary directory"

    tmp_dir = tempfile.mkdtemp()
    sound_file = os.path.join(tmp_dir, "sound.flac")
    temp_video_file = None
    temp_subtitle_file = None

    def clear():
        try:
            os.unlink(sound_file)
        except:
            pass

        if temp_video_file:
            try:
                os.unlink(temp_video_file)
            except:
                pass

        if temp_subtitle_file:
            try:
                os.unlink(temp_subtitle_file)
            except:
                pass

        # Clean up any chunk files in the temp directory
        try:
            import glob

            chunk_files = glob.glob(os.path.join(tmp_dir, "chunk_*.flac"))
            for chunk_file in chunk_files:
                try:
                    os.unlink(chunk_file)
                except:
                    pass
        except:
            pass

        try:
            os.rmdir(tmp_dir)
        except:
            pass

    try:
        # Handle remote subtitle file
        if is_http_url(subtitle_file):
            temp_subtitle_file = os.path.join(tmp_dir, "subtitles.srt")
            if not download_remote_subtitle_file(subtitle_file, temp_subtitle_file, verbose=verbose):
                raise RuntimeError("Failed to download remote subtitle file")
            actual_subtitle_file = temp_subtitle_file
        else:
            actual_subtitle_file = subtitle_file

        # Handle video files with chunking support for both local and remote
        if anchor_points is not None or anchor_duration_mins is not None or anchor_duration_secs is not None:
            if verbose:
                print("Using anchor-based processing for large file...")

            # Validate that only one duration parameter is specified
            if anchor_duration_mins is not None and anchor_duration_secs is not None:
                raise ValueError("Cannot specify both anchor_duration_mins and anchor_duration_secs. Use only one.")

            # Set defaults
            if anchor_points is None:
                anchor_points = 5  # Default to 5 anchor points
            if anchor_duration_mins is None and anchor_duration_secs is None:
                anchor_duration_mins = 5.0  # Default to 5 minutes per anchor

            # Convert duration to seconds
            if anchor_duration_mins is not None:
                anchor_duration_secs = anchor_duration_mins * 60

            # Try to get video duration
            duration = get_video_duration(video_file, verbose=verbose)

            if duration and duration > 0:
                # Calculate chunk specifications
                # We'll take evenly spaced chunks throughout the video
                chunk_duration = min(anchor_duration_secs, duration / anchor_points)  # Use duration in seconds
                chunk_specs = []

                for i in range(anchor_points):
                    start_time = (duration / anchor_points) * i
                    # Make sure we don't go past the end
                    actual_duration = min(chunk_duration, duration - start_time)
                    if actual_duration > 0:
                        chunk_specs.append((start_time, actual_duration))

                if verbose:
                    print(
                        f"Extracting {len(chunk_specs)} chunks of ~{chunk_duration/60:.1f}min each from {duration/60:.1f}min video"
                    )

                # Extract audio chunks (works for both local and remote files)
                chunk_paths = extract_audio_chunks_from_remote_video(
                    video_file,
                    tmp_dir,
                    max_chunks=anchor_points,
                    max_chunk_duration=int(chunk_duration),
                    verbose=verbose,
                )

                # Combine the chunks into a single audio file
                combine_audio_chunks(chunk_paths, sound_file, verbose=verbose)
            else:
                if verbose:
                    print("Could not determine video duration, falling back to full extraction...")
                # Fall back to full extraction
                if is_http_url(video_file):
                    temp_video_file = os.path.join(tmp_dir, "temp_video.mp4")
                    if download_remote_video_with_ffmpeg(video_file, temp_video_file, verbose=verbose):
                        extract_sound(temp_video_file, sound_file)
                    else:
                        raise RuntimeError("Failed to download remote video file")
                else:
                    extract_sound(video_file, sound_file)

        elif is_http_url(video_file):
            # Download entire remote file using ffmpeg
            temp_video_file = os.path.join(tmp_dir, "temp_video.mp4")
            if download_remote_video_with_ffmpeg(video_file, temp_video_file, verbose=verbose):
                extract_sound(temp_video_file, sound_file)
            else:
                raise RuntimeError("Failed to download remote video file")
        else:
            # Local file - traditional extraction
            extract_sound(video_file, sound_file)

        return import_item(sound_file, actual_subtitle_file, **kwargs)

    finally:
        clear()


def transform_srt(in_srt, out_srt, transform_func):
    with open(out_srt, "wb") as out_file:
        out_srt = srt_io.writer(out_file)
        for sub in srt_io.read_file(in_srt):
            out_srt.write(transform_func(sub.begin), transform_func(sub.end), sub.text)


def download_remote_subtitle_file(url, output_path, verbose=False):
    """
    Download a remote subtitle file using curl or wget.

    Args:
        url (str): Remote subtitle URL
        output_path (str): Local path to save the subtitle file
        verbose (bool): Enable verbose output

    Returns:
        bool: True if download succeeded, False otherwise
    """
    if verbose:
        print(f"Downloading remote subtitle file from {url}...")

    # Try curl first, then wget as fallback
    commands = [["curl", "-L", "-o", output_path, url], ["wget", "-O", output_path, url]]

    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode == 0 and os.path.exists(output_path):
                if verbose:
                    print(f"Successfully downloaded subtitle file to {output_path}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    if verbose:
        print(f"Failed to download subtitle file from {url}")
    return False


def process_audio_chunks_with_anchors(
    video_file,
    subtitle_file,
    anchor_points=5,
    anchor_duration_mins=None,
    anchor_duration_secs=None,
    verbose=False,
    **kwargs,
):
    """
    Process video in chunks to create multiple synchronization anchor points.
    Each chunk serves as a sync point for progressive drift correction.

    Args:
        video_file (str): Video file path or URL
        subtitle_file (str): Subtitle file path or URL
        anchor_points (int): Number of anchor points to create (default: 5)
        anchor_duration_mins (float): Duration per anchor in minutes (default: 5.0 if anchor_duration_secs not specified)
        anchor_duration_secs (float): Duration per anchor in seconds (alternative to anchor_duration_mins)
        verbose (bool): Enable verbose output

    Returns:
        list: List of (chunk_info, sound_data, subvec) tuples for each chunk
    """

    # Validate that only one duration parameter is specified
    if anchor_duration_mins is not None and anchor_duration_secs is not None:
        raise ValueError("Cannot specify both anchor_duration_mins and anchor_duration_secs. Use only one.")

    # Set defaults and convert to seconds
    if anchor_duration_mins is None and anchor_duration_secs is None:
        anchor_duration_mins = 5.0  # Default to 5 minutes

    if anchor_duration_mins is not None:
        duration_secs = anchor_duration_mins * 60
    else:
        duration_secs = anchor_duration_secs

    tmp_dir = tempfile.mkdtemp()
    temp_video_file = None
    temp_subtitle_file = None
    chunk_data = []

    def clear():
        if temp_video_file:
            try:
                os.unlink(temp_video_file)
            except:
                pass
        if temp_subtitle_file:
            try:
                os.unlink(temp_subtitle_file)
            except:
                pass
        # Clean up any chunk files
        try:
            import glob

            chunk_files = glob.glob(os.path.join(tmp_dir, "chunk_*.flac"))
            for chunk_file in chunk_files:
                try:
                    os.unlink(chunk_file)
                except:
                    pass
        except:
            pass
        try:
            os.rmdir(tmp_dir)
        except:
            pass

    try:
        # Validate remote video URL early to catch issues
        if is_http_url(video_file):
            validate_remote_video_url(video_file, verbose=verbose)
        
        # Handle remote subtitle file
        if is_http_url(subtitle_file):
            temp_subtitle_file = os.path.join(tmp_dir, "subtitles.srt")
            if not download_remote_subtitle_file(subtitle_file, temp_subtitle_file, verbose=verbose):
                raise RuntimeError("Failed to download remote subtitle file")
            actual_subtitle_file = temp_subtitle_file
        else:
            actual_subtitle_file = subtitle_file

        # Read subtitle file once
        subs = list(srt_io.read_file(actual_subtitle_file))
        if not subs:
            raise RuntimeError("Empty or invalid subtitle file")

        total_subtitle_duration = max(s.end for s in subs)

        # Get video duration
        if is_http_url(video_file):
            duration = get_video_duration(video_file, verbose=verbose)
        else:
            duration = get_video_duration(video_file, verbose=verbose)

        if duration is None:
            if verbose:
                print("Could not determine video duration, using subtitle duration as estimate")
            duration = total_subtitle_duration

        if verbose:
            print(f"Video duration: {duration:.1f}s, Subtitle duration: {total_subtitle_duration:.1f}s")

        # Calculate chunk time points
        if duration < duration_secs * 2:  # If video is shorter than 2 anchor durations
            # Short video, just use one chunk
            chunk_times = [(0, min(duration, duration_secs))]
        else:
            # Distribute chunks evenly across the video
            chunk_times = []
            interval = duration / anchor_points
            for i in range(anchor_points):
                start_time = interval * i
                remaining_time = duration - start_time
                chunk_duration = min(duration_secs, remaining_time)
                if chunk_duration >= 30:  # Only include chunks with at least 30 seconds
                    chunk_times.append((start_time, chunk_duration))

        if verbose:
            print(f"Processing {len(chunk_times)} chunks as synchronization anchors")

        # Process each chunk
        for i, (start_time, chunk_duration) in enumerate(chunk_times):
            if verbose:
                print(
                    f"Processing chunk {i+1}/{len(chunk_times)}: {start_time:.1f}s - {start_time + chunk_duration:.1f}s"
                )

            # Extract audio for this chunk
            chunk_audio_path = os.path.join(tmp_dir, f"chunk_{i+1}.flac")

            if is_http_url(video_file):
                # Extract chunk directly from remote URL
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error" if not verbose else "info",
                    "-ss",
                    str(start_time),
                    "-i",
                    video_file,
                    "-t",
                    str(chunk_duration),
                    "-vn",
                    "-sn",
                    "-ac",
                    "1",
                    chunk_audio_path,
                ]
            else:
                # Extract chunk from local file
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error" if not verbose else "info",
                    "-ss",
                    str(start_time),
                    "-i",
                    video_file,
                    "-t",
                    str(chunk_duration),
                    "-vn",
                    "-sn",
                    "-ac",
                    "1",
                    chunk_audio_path,
                ]

            try:
                result = subprocess.run(cmd, timeout=300)
                if result.returncode != 0 or not os.path.exists(chunk_audio_path):
                    if verbose:
                        print(f"Warning: Failed to extract chunk {i+1}, skipping...")
                        if result.returncode != 0:
                            print(f"  FFmpeg exit code: {result.returncode}")
                        if not os.path.exists(chunk_audio_path):
                            print(f"  Output file not created: {chunk_audio_path}")
                    continue
            except subprocess.TimeoutExpired:
                if verbose:
                    print(f"Warning: Timeout extracting chunk {i+1}, skipping...")
                continue

            # Import audio data for this chunk
            try:
                sound_data = import_sound(chunk_audio_path)
                samples, sample_rate, data_range = sound_data
                n = len(samples)

                # Build subtitle vector for this time range
                chunk_subs = []
                for sub in subs:
                    # Adjust subtitle times relative to chunk start
                    sub_start = sub.begin - start_time
                    sub_end = sub.end - start_time

                    # Only include subtitles that overlap with this chunk
                    if sub_end > 0 and sub_start < chunk_duration:
                        # Clamp to chunk boundaries
                        sub_start = max(0, sub_start)
                        sub_end = min(chunk_duration, sub_end)

                        if sub_end > sub_start:  # Valid overlap
                            # Create a temporary subtitle entry for this chunk
                            from collections import namedtuple

                            ChunkSub = namedtuple("ChunkSub", ["begin", "end", "text"])
                            chunk_subs.append(ChunkSub(sub_start, sub_end, sub.text))

                if chunk_subs:
                    # Filter kwargs to only include parameters that build_sub_vec accepts
                    sub_vec_kwargs = {}
                    if "sub_filter" in kwargs:
                        sub_vec_kwargs["sub_filter"] = kwargs["sub_filter"]

                    subvec = build_sub_vec(chunk_subs, sample_rate, n, **sub_vec_kwargs)

                    chunk_info = {
                        "index": i,
                        "start_time": start_time,
                        "duration": chunk_duration,
                        "subtitle_count": len(chunk_subs),
                    }

                    chunk_data.append((chunk_info, sound_data, subvec))

                    if verbose:
                        print(f"Chunk {i+1}: {len(chunk_subs)} subtitles, {n} audio samples")
                else:
                    if verbose:
                        print(f"Chunk {i+1}: No subtitles in this time range, skipping...")

            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to process chunk {i+1}: {e}")
                continue

        if not chunk_data:
            if verbose:
                print("No valid chunks could be processed. This may be due to:")
                print("1. Invalid or inaccessible video URL")
                print("2. Network connectivity issues")
                print("3. Video format not supported by ffmpeg")
                print("4. Archive.org details page URL instead of direct video file URL")
                print("\nTip: For Archive.org, try using the direct download URL instead of the details page URL")
            raise RuntimeError("No valid chunks could be processed")

        return chunk_data

    finally:
        clear()
