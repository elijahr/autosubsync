#!/usr/bin/python3
import argparse
import numpy as np
import gc
import os
import sys


def parse_skew(skew):
    "helper function, parse maybe fractional notation like 24/24 to float"
    if skew is None:
        return None
    if "/" in skew:
        a, b = [float(x) for x in skew.split("/")]
        return a / b
    else:
        return float(skew)


def synchronize(
    video_file,
    subtitle_file,
    output_file,
    verbose=False,
    parallelism=3,
    fixed_skew=None,
    model_file=None,
    return_parameters=False,
    anchor_points=None,
    anchor_duration_mins=None,
    anchor_duration_secs=None,
    **kwargs,
):
    """
    Automatically synchronize subtitles with audio in a video file.
    Uses FFMPEG to extract the audio from the video file and the command line
    tool "ffmpeg" must be available. Uses temporary files which are deleted
    automatically.

    Args:
        video_file (string): Input video file name (can be HTTP/HTTPS URL)
        subtitle_file (string): Input SRT subtitle file name
        output_file (string): Output (synchronized) SRT subtitle file name
        verbose (boolean): If True, print progress information to stdout
        return_parameters (boolean): If True, returns the syncrhonization
            parameters instead of just the success flag
        anchor_points (int): Number of anchor points for progressive synchronization.
            If >1, enables anchor-based sync with drift correction. Default: None (single-point sync)
        anchor_duration_mins (float): Duration in minutes for each anchor segment.
            Default: 5 minutes per anchor. Cannot be used with anchor_duration_secs.
        anchor_duration_secs (float): Duration in seconds for each anchor segment.
            Cannot be used with anchor_duration_mins.
        other arguments: Search parameters, see ``autosubsync --help``

    Returns:
        If return_parameters is False (default), returns
        True on success (quality of fit test passed), False if failed.

        If return_parameters is True, returns a tuple of four values

            success (boolean)   success flag as above
            quality (float)     metric used to determine the value of "success"
            skew (float)        best fit skew/speed (unitless)
            shift (float)       best fit shift in seconds

    """

    # Validate that only one duration parameter is specified
    if anchor_duration_mins is not None and anchor_duration_secs is not None:
        raise ValueError("Cannot specify both anchor_duration_mins and anchor_duration_secs. Use only one.")

    # these are here to enable running as python3 autosubsync/main.py
    from autosubsync import features
    from autosubsync import find_transform
    from autosubsync import model
    from autosubsync import preprocessing
    from autosubsync import quality_of_fit
    from autosubsync import srt_io

    # first check that the SRT file is valid before extracting any audio data
    if preprocessing.is_http_url(subtitle_file):
        # For remote subtitle files, validation happens during download
        pass
    else:
        srt_io.check_file(subtitle_file)

    # Check if we should use anchor-based synchronization
    if anchor_points is not None and anchor_points > 1:
        if verbose:
            print(f"Using anchor-based synchronization with {anchor_points} anchor points...")
        try:
            return synchronize_with_anchors(
                video_file,
                subtitle_file,
                output_file,
                verbose=verbose,
                parallelism=parallelism,
                fixed_skew=fixed_skew,
                model_file=model_file,
                return_parameters=return_parameters,
                anchor_points=anchor_points,
                anchor_duration_mins=anchor_duration_mins,
                anchor_duration_secs=anchor_duration_secs,
                **kwargs,
            )
        except RuntimeError as e:
            if verbose:
                print(f"Warning: Anchor-based synchronization failed: {e}")
                print("Falling back to traditional single-point synchronization...")
            # Fall through to traditional synchronization below

    # Traditional single-point synchronization
    if verbose:
        print("Using traditional single-point synchronization...")

    # argument parsing
    if model_file is None:
        from pkg_resources import resource_filename

        model_file = resource_filename(__name__, "../trained-model.bin")

    fixed_skew = parse_skew(fixed_skew)

    # load model
    trained_model = model.load(model_file)

    if verbose:
        print("Extracting audio using ffmpeg and reading subtitles...")
    sound_data, subvec = preprocessing.import_target_files(
        video_file,
        subtitle_file,
        verbose=verbose,
        anchor_points=anchor_points,
        anchor_duration_mins=anchor_duration_mins,
        anchor_duration_secs=anchor_duration_secs,
    )

    if verbose:
        print(
            ("computing features for %d audio samples " + "using %d parallel process(es)") % (len(subvec), parallelism)
        )

    features_x, shifted_y = features.compute(sound_data, subvec, parallelism=parallelism)

    if verbose:
        print("extracted features of size %s, performing speech detection" % str(features_x.shape))

    y_scores = model.predict(trained_model, features_x)

    # save some memory before parallelization fork so we look less bad
    del features_x, sound_data, subvec
    gc.collect()

    if verbose:
        print("computing best fit with %d frames" % len(y_scores))

    skew, shift, quality = find_transform.find_transform_parameters(
        shifted_y,
        y_scores,
        parallelism=parallelism,
        fixed_skew=fixed_skew,
        bias=trained_model[1],
        verbose=verbose,
        **kwargs,
    )

    success = bool(quality > quality_of_fit.threshold)
    if verbose:
        print("quality of fit: %g, threshold %g" % (quality, quality_of_fit.threshold))
        print("Fit complete. Performing resync, writing to " + output_file)

    transform_func = find_transform.parameters_to_transform(skew, shift)
    preprocessing.transform_srt(subtitle_file, output_file, transform_func)

    if verbose and success:
        print("success!")

    if return_parameters:
        return success, quality, skew, shift
    else:
        return success


def synchronize_with_anchors(
    video_file,
    subtitle_file,
    output_file,
    verbose=False,
    parallelism=3,
    fixed_skew=None,
    model_file=None,
    return_parameters=False,
    anchor_points=None,
    anchor_duration_mins=None,
    anchor_duration_secs=None,
    **kwargs,
):
    """
    Synchronize subtitles using multiple anchor points for progressive drift correction.
    This method processes the video in chunks, each serving as a sync anchor point.

    Args:
        video_file (string): Input video file name (can be HTTP/HTTPS URL)
        subtitle_file (string): Input SRT subtitle file name (can be HTTP/HTTPS URL)
        output_file (string): Output (synchronized) SRT subtitle file name
        verbose (boolean): If True, print progress information to stdout
        anchor_points (int): Number of anchor points for synchronization.
            If None, defaults to 5 anchor points.
        anchor_duration_mins (float): Duration in minutes for each anchor segment.
            If None, defaults to 5 minutes per anchor.
        other arguments: Search parameters, see ``autosubsync --help``

    Returns:
        If return_parameters is False (default), returns
        True on success, False if failed.

        If return_parameters is True, returns a tuple with anchor point data
    """
    from autosubsync import features
    from autosubsync import find_transform
    from autosubsync import model
    from autosubsync import preprocessing
    from autosubsync import quality_of_fit
    from autosubsync import srt_io

    # Validate that only one duration parameter is specified
    if anchor_duration_mins is not None and anchor_duration_secs is not None:
        raise ValueError("Cannot specify both anchor_duration_mins and anchor_duration_secs. Use only one.")

    # Set defaults
    if anchor_points is None:
        anchor_points = 5
    if anchor_duration_mins is None and anchor_duration_secs is None:
        anchor_duration_mins = 5.0  # Default to 5 minutes

    # Convert to consistent units for internal use
    if anchor_duration_mins is not None:
        duration_secs = anchor_duration_mins * 60
    else:
        duration_secs = anchor_duration_secs

    # Check that the SRT file is valid before processing
    if preprocessing.is_http_url(subtitle_file):
        # For remote files, we'll validate after download
        pass
    else:
        srt_io.check_file(subtitle_file)

    # Load model
    if model_file is None:
        from pkg_resources import resource_filename

        model_file = resource_filename(__name__, "../trained-model.bin")

    fixed_skew = parse_skew(fixed_skew)
    trained_model = model.load(model_file)

    if verbose:
        print("Processing video in chunks for anchor-based synchronization...")

    # Process video in chunks to get multiple anchor points
    chunk_data = preprocessing.process_audio_chunks_with_anchors(
        video_file,
        subtitle_file,
        anchor_points=anchor_points,
        anchor_duration_secs=duration_secs,
        verbose=verbose,
        **kwargs,
    )

    if verbose:
        print(f"Processing {len(chunk_data)} anchor chunks...")

    # Process each chunk to find sync parameters
    anchor_results = []

    for chunk_info, sound_data, subvec in chunk_data:
        if verbose:
            print(f"Analyzing chunk {chunk_info['index'] + 1} at {chunk_info['start_time']:.1f}s...")

        try:
            # Compute features for this chunk
            features_x, shifted_y = features.compute(sound_data, subvec, parallelism=parallelism)

            if features_x.size == 0:
                if verbose:
                    print(f"No features extracted for chunk {chunk_info['index'] + 1}, skipping...")
                continue

            # Perform speech detection
            y_scores = model.predict(trained_model, features_x)

            # Find transform parameters for this chunk
            skew, shift, quality = find_transform.find_transform_parameters(
                shifted_y,
                y_scores,
                parallelism=parallelism,
                fixed_skew=fixed_skew,
                bias=trained_model[1],
                verbose=False,  # Don't be verbose for individual chunks
                **kwargs,
            )

            anchor_point = {
                "chunk_info": chunk_info,
                "skew": skew,
                "shift": shift,
                "quality": quality,
                "absolute_time": chunk_info["start_time"],
                "audio_shift": shift,  # Shift relative to chunk start
            }

            anchor_results.append(anchor_point)

            if verbose:
                print(f"Chunk {chunk_info['index'] + 1}: skew={skew:.6f}, shift={shift:.3f}s, quality={quality:.3f}")

        except Exception as e:
            if verbose:
                print(f"Failed to process chunk {chunk_info['index'] + 1}: {e}")
            continue

    if not anchor_results:
        if verbose:
            print("No valid anchor points found, cannot synchronize")
        return False if not return_parameters else (False, 0.0, 1.0, 0.0)

    if verbose:
        print(f"Found {len(anchor_results)} valid anchor points")

    # Apply progressive synchronization using anchor points
    success = apply_anchor_based_synchronization(subtitle_file, output_file, anchor_results, verbose=verbose)

    if verbose and success:
        print("Anchor-based synchronization completed successfully!")

    if return_parameters:
        # Return aggregate metrics
        avg_quality = sum(ap["quality"] for ap in anchor_results) / len(anchor_results)
        avg_skew = sum(ap["skew"] for ap in anchor_results) / len(anchor_results)
        total_shift = anchor_results[0]["shift"] if anchor_results else 0.0
        return success, avg_quality, avg_skew, total_shift
    else:
        return success


def apply_anchor_based_synchronization(subtitle_file, output_file, anchor_points, verbose=False):
    """
    Apply synchronization using multiple anchor points for progressive drift correction.

    Args:
        subtitle_file (str): Input subtitle file (local or remote)
        output_file (str): Output synchronized subtitle file
        anchor_points (list): List of anchor point dictionaries
        verbose (bool): Enable verbose output

    Returns:
        bool: True if successful
    """
    from autosubsync import srt_io
    from autosubsync import preprocessing
    import tempfile

    # Download remote subtitle file if needed
    temp_subtitle_file = None
    if preprocessing.is_http_url(subtitle_file):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            temp_subtitle_file = f.name
        if not preprocessing.download_remote_subtitle_file(subtitle_file, temp_subtitle_file, verbose=verbose):
            return False
        actual_subtitle_file = temp_subtitle_file
    else:
        actual_subtitle_file = subtitle_file

    try:
        # Read all subtitles
        subs = list(srt_io.read_file(actual_subtitle_file))
        if not subs:
            if verbose:
                print("No subtitles found in file")
            return False

        if verbose:
            print(f"Applying progressive synchronization to {len(subs)} subtitle entries...")

        # Sort anchor points by time
        anchor_points.sort(key=lambda ap: ap["absolute_time"])

        # Apply synchronization segment by segment
        with open(output_file, "wb") as out_file:
            writer = srt_io.writer(out_file)

            for i, sub in enumerate(subs):
                # Find which anchor segment this subtitle belongs to
                current_anchor = None
                next_anchor = None

                # Find the anchor points that bracket this subtitle
                for j, anchor in enumerate(anchor_points):
                    if sub.begin >= anchor["absolute_time"]:
                        current_anchor = anchor
                        if j + 1 < len(anchor_points):
                            next_anchor = anchor_points[j + 1]
                    else:
                        break

                if current_anchor is None:
                    # Before first anchor, use first anchor's parameters
                    current_anchor = anchor_points[0]

                # Calculate synchronized times
                if next_anchor and current_anchor != next_anchor:
                    # Interpolate between anchors for smooth drift correction
                    segment_progress = (sub.begin - current_anchor["absolute_time"]) / (
                        next_anchor["absolute_time"] - current_anchor["absolute_time"]
                    )
                    segment_progress = max(0, min(1, segment_progress))  # Clamp to [0,1]

                    # Interpolate skew and shift
                    skew = current_anchor["skew"] + segment_progress * (next_anchor["skew"] - current_anchor["skew"])
                    shift = current_anchor["shift"] + segment_progress * (
                        next_anchor["shift"] - current_anchor["shift"]
                    )
                else:
                    # Use current anchor's parameters
                    skew = current_anchor["skew"]
                    shift = current_anchor["shift"]

                # Apply transformation
                def transform_time(t):
                    return t * skew + shift

                new_begin = transform_time(sub.begin)
                new_end = transform_time(sub.end)

                # Write synchronized subtitle
                writer.write(new_begin, new_end, sub.text)

        if verbose:
            print(f"Synchronized subtitles written to {output_file}")

        return True

    except Exception as e:
        if verbose:
            print(f"Failed to apply synchronization: {e}")
        return False

    finally:
        if temp_subtitle_file:
            try:
                os.unlink(temp_subtitle_file)
            except:
                pass


def cli(packaged_model=False):
    p = argparse.ArgumentParser(description=synchronize.__doc__.split("\n\n")[0])
    p.add_argument("video_file", help="Input video file")
    p.add_argument("subtitle_file", help="Input SRT subtitle file")
    p.add_argument("output_file", help="Output (auto-synchronized) SRT subtitle file")

    # Make model file an argument only in the non-packaged version
    if not packaged_model:
        p.add_argument("--model_file", default="trained-model.bin")

    p.add_argument("--max_shift_secs", default=20.0, type=float, help="Maximum subtitle shift in seconds (default 20)")
    p.add_argument("--parallelism", default=3, type=int, help="Number of parallel worker processes (default 3)")
    p.add_argument("--fixed_skew", default=None, help="Use a fixed skew (e.g. 1) instead of auto-detection")
    p.add_argument("--silent", action="store_true", help="Do not print progress information")

    # Anchor-based synchronization parameters
    p.add_argument(
        "--anchor_points",
        default=None,
        type=int,
        help="Number of anchor points for progressive synchronization. Use >1 to enable anchor-based sync with drift correction (default: 5)",
    )
    p.add_argument(
        "--anchor_duration_mins",
        default=None,
        type=float,
        help="Duration in minutes for each anchor segment (default: 5 minutes per anchor). Cannot be used with --anchor_duration_secs.",
    )
    p.add_argument(
        "--anchor_duration_secs",
        default=None,
        type=float,
        help="Duration in seconds for each anchor segment. Cannot be used with --anchor_duration_mins.",
    )
    args = p.parse_args()

    # Validate that only one duration parameter is specified
    if args.anchor_duration_mins is not None and args.anchor_duration_secs is not None:
        p.error("Cannot specify both --anchor_duration_mins and --anchor_duration_secs. Use only one.")

    if packaged_model:
        model_file = None
    else:
        model_file = args.model_file

    success = synchronize(
        args.video_file,
        args.subtitle_file,
        args.output_file,
        verbose=not args.silent,
        model_file=model_file,
        max_shift_secs=args.max_shift_secs,
        parallelism=args.parallelism,
        fixed_skew=args.fixed_skew,
        anchor_points=args.anchor_points,
        anchor_duration_mins=args.anchor_duration_mins,
        anchor_duration_secs=args.anchor_duration_secs,
    )

    if not success:
        sys.stderr.write("\nWARNING: low quality of fit. Wrong subtitle file?\n")
        sys.exit(1)


def cli_packaged():
    """Entry point in the packaged, pip-installable version"""
    cli(packaged_model=True)


if __name__ == "__main__":
    # Entry point for running from repository root folder
    sys.path.append(".")
    cli(packaged_model=False)
