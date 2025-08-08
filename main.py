#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 mohamed hamdi <haamdi@outlook.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import shutil
import signal
import subprocess
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Generator, Optional, Tuple

import click
from magic import Magic
from rich.console import Console
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

ERROR_LOG = "audioconv_errors.log"
CONVERTED_MARKER = "conv_"
current_process = None
current_output = None
current_temp_file = None
console = Console()


def signal_handler(sig, frame):
    """Handle signals and clean up resources"""
    global current_process, current_output, current_temp_file

    if current_process is not None:
        # Terminate the running ffmpeg process
        current_process.terminate()
        try:
            current_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            current_process.kill()

    # Clean up unfinished output file
    if current_output and current_output.exists():
        try:
            current_output.unlink()
            console.print(f"\nRemoved incomplete file: {current_output}")
        except Exception as e:
            console.print(f"\nError removing incomplete file: {e}")

    # Clean up temporary file
    if current_temp_file and current_temp_file.exists():
        try:
            current_temp_file.unlink()
            console.print(f"\nRemoved temporary file: {current_temp_file}")
        except Exception as e:
            console.print(f"\nError removing temporary file: {e}")

    # Remove error log if empty
    if Path(ERROR_LOG).exists() and Path(ERROR_LOG).stat().st_size == 0:
        Path(ERROR_LOG).unlink()

    console.print("\nExiting due to interrupt...")
    sys.exit(1)


def format_size(bytes: float) -> str:
    """Convert bytes to a human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(bytes) < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} TB"


def format_duration(seconds: float) -> str:
    """Convert seconds to human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


class ConversionStatus(Enum):
    NOT_AUDIO = 0
    ALREADY_CONVERTED = 1
    NEEDS_CONVERSION = 2
    ERROR = 3


def check_conversion_status(file_path: Path, output_format: str) -> ConversionStatus:
    """Check if file needs conversion"""
    # Check if output already exists
    output_path = file_path.with_suffix(f".{output_format}")
    if output_path.exists():
        return ConversionStatus.ALREADY_CONVERTED

    # Check if file has conversion marker
    if file_path.name.startswith(CONVERTED_MARKER):
        return ConversionStatus.ALREADY_CONVERTED

    # Check if file is audio
    try:
        mime = Magic(mime=True)
        mimetype = mime.from_file(file_path)
        if not mimetype.startswith("audio/"):
            # Also check with ffprobe for files that magic might miss
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                 str(file_path)],
                capture_output=True,
                text=True
            )
            if result.stdout.strip() != "audio":
                return ConversionStatus.NOT_AUDIO
    except Exception:
        return ConversionStatus.ERROR

    return ConversionStatus.NEEDS_CONVERSION


def get_audio_info(file_path: Path) -> Optional[dict]:
    """Get audio file information using ffprobe"""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration,bit_rate:stream=codec_name,sample_rate,channels",
            "-of", "json",
            str(file_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        import json
        data = json.loads(result.stdout)

        info = {
            "duration": float(data.get("format", {}).get("duration", 0)),
            "bitrate": int(data.get("format", {}).get("bit_rate", 0)),
        }

        if data.get("streams"):
            stream = data["streams"][0]
            info["codec"] = stream.get("codec_name", "unknown")
            info["sample_rate"] = int(stream.get("sample_rate", 0))
            info["channels"] = int(stream.get("channels", 0))

        return info
    except Exception:
        return None


def audio_generator(input_path: Path, process_all: bool, recursive: bool) -> Generator[Path, None, None]:
    """Generate all files to process"""
    if process_all:
        if recursive:
            for root, _, files in os.walk(input_path):
                for f in files:
                    yield Path(root) / f
        else:
            for f in os.listdir(input_path):
                file_path = input_path / f
                if file_path.is_file():
                    yield file_path
    else:
        for path in input_path:
            yield path


def convert_audio(
    file_path: Path,
    output_format: str,
    audio_quality: str,
    codec: str,
    sample_rate: Optional[int],
    keep_original: bool,
    progress: Optional[Progress] = None,
    task_id: Optional[TaskID] = None,
) -> Tuple[bool, int, Optional[dict]]:
    """Convert a single audio file"""
    global current_process, current_output, current_temp_file
    temp_file = None

    try:
        # Get audio info for reporting
        audio_info = get_audio_info(file_path)

        # Create output path with proper escaping
        output_path = file_path.with_suffix(f".{output_format}")
        current_output = output_path

        # Create temporary file in the same directory
        temp_file = tempfile.NamedTemporaryFile(
            dir=file_path.parent,
            delete=False,
            suffix=f".{output_format}"
        )
        temp_path = Path(temp_file.name)
        temp_file.close()
        current_temp_file = temp_path

        # Build ffmpeg command - simplified to match working bash version
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output files without asking
            "-hide_banner",
            "-v", "info",  # Show some info but not too verbose
            "-stats",
            "-i", str(file_path),
            "-c:a", codec,
            "-b:a", audio_quality,
            str(temp_path)
        ]

        # Run conversion
        current_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        if progress and task_id and audio_info and audio_info.get("duration", 0) > 0:
            # Track progress by parsing ffmpeg stderr output
            duration = audio_info["duration"]
            stderr_lines = []
            
            try:
                while True:
                    line = current_process.stderr.readline()
                    if not line:
                        break
                    line = line.strip()
                    stderr_lines.append(line)
                    
                    # Parse time from ffmpeg progress output
                    if "time=" in line:
                        try:
                            time_part = line.split("time=", 1)[1].strip()
                            time_tokens = time_part.split()
                            if time_tokens:
                                time_str = time_tokens[0]
                                # Convert time string (HH:MM:SS.ms) to seconds
                                time_parts = time_str.split(":")
                                if len(time_parts) == 3:
                                    hours, minutes, seconds = time_parts
                                    current_time = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                                    current_percent = min(int((current_time / duration) * 100), 100)
                                    task = next((t for t in progress.tasks if t.id == task_id), None)
                                    if task is not None:
                                        delta = current_percent - task.completed
                                        if delta > 0:
                                            progress.update(task_id, advance=delta)
                        except (ValueError, IndexError):
                            pass
                
                current_process.wait(timeout=300)  # 5 minute timeout
                progress.update(task_id, completed=100)
                stderr_output = "\n".join(stderr_lines)
            except subprocess.TimeoutExpired:
                current_process.kill()
                current_process.wait()
                raise Exception(f"FFmpeg conversion timed out after 5 minutes for {file_path.name}")
        else:
            # No progress tracking - use original method with timeout
            try:
                stdout, stderr_output = current_process.communicate(timeout=300)
            except subprocess.TimeoutExpired:
                current_process.kill()
                current_process.wait()
                raise Exception(f"FFmpeg conversion timed out after 5 minutes for {file_path.name}")

        if current_process.returncode != 0:
            raise subprocess.CalledProcessError(
                current_process.returncode, cmd, stdout, stderr_output
            )

        # Move temp file to final location
        temp_path.rename(output_path)

        # Get file sizes for reporting
        original_size = file_path.stat().st_size
        new_size = output_path.stat().st_size

        # Remove original if requested
        if not keep_original:
            file_path.unlink()

        return True, original_size - new_size, audio_info

    except Exception as e:
        if temp_path and temp_path.exists():
            temp_path.unlink()
        raise e
    finally:
        current_process = None
        current_output = None
        current_temp_file = None


def log_error(file_path: Path, error: str):
    """Log errors to the error log file"""
    with open(ERROR_LOG, "a") as f:
        f.write(f"Error processing {file_path}:\n{error}\n")
        f.write("-" * 40 + "\n")


def check_required_tools():
    """Check if required tools are installed"""
    required_tools = {
        "ffmpeg": "ffmpeg (required for audio conversion)",
        "ffprobe": "ffprobe (part of ffmpeg, required for audio analysis)",
    }
    missing_tools = []

    for tool, description in required_tools.items():
        if not shutil.which(tool):
            missing_tools.append(f"{tool} ({description})")

    if missing_tools:
        console.print(
            "[red]Error: The following required tools are not installed:[/red]"
        )
        for tool in missing_tools:
            console.print(f"  - {tool}")
        console.print("\n[yellow]Please install ffmpeg and try again.[/yellow]")
        sys.exit(1)


def get_codec_for_format(format: str) -> str:
    """Get the appropriate codec for the output format"""
    codec_map = {
        "m4a": "aac",
        "mp3": "libmp3lame",
        "ogg": "libvorbis",
        "opus": "libopus",
        "flac": "flac",
        "wav": "pcm_s16le",
    }
    return codec_map.get(format, "aac")


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("input_path", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
    "-f", "--format",
    "output_format",
    type=click.Choice(["m4a", "mp3", "ogg", "opus", "flac", "wav"]),
    default="m4a",
    help="Output audio format"
)
@click.option(
    "-q", "--quality",
    "audio_quality",
    type=str,
    default="128k",
    help="Audio bitrate (e.g., 128k, 192k, 320k)"
)
@click.option(
    "-c", "--codec",
    type=str,
    default=None,
    help="Override audio codec (auto-detected by default)"
)
@click.option(
    "-s", "--sample-rate",
    type=int,
    default=None,
    help="Output sample rate in Hz (e.g., 44100, 48000)"
)
@click.option(
    "-a", "--all",
    "process_all",
    is_flag=True,
    help="Process all audio files in directory"
)
@click.option(
    "-r", "--recursive",
    is_flag=True,
    help="Process files recursively in subdirectories"
)
@click.option(
    "-k", "--keep-original",
    is_flag=True,
    help="Keep original files after conversion"
)
def main(input_path, output_format, audio_quality, codec, sample_rate,
         process_all, recursive, keep_original):
    """Audio conversion tool with batch processing capabilities

    Convert audio files to various formats with customizable quality settings.
    Supports batch processing of entire directories with progress tracking.
    """
    check_required_tools()

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Clear previous error log
    if Path(ERROR_LOG).exists():
        Path(ERROR_LOG).unlink()

    # Validate input
    if not input_path and not process_all:
        raise click.UsageError("Must specify input files or use --all")
    if process_all and input_path:
        raise click.UsageError("Cannot specify both --all and input files")

    # Determine codec
    if not codec:
        codec = get_codec_for_format(output_format)

    base_path = Path.cwd() if process_all else Path(input_path[0]).parent

    # Collect all files to process
    files_list = list(audio_generator(
        base_path if process_all else input_path,
        process_all,
        recursive
    ))

    if not files_list:
        console.print("[yellow]No files to process[/yellow]")
        return

    # Filter for audio files that need conversion
    files_to_convert = []
    for f in files_list:
        status = check_conversion_status(f, output_format)
        if status == ConversionStatus.NEEDS_CONVERSION:
            files_to_convert.append(f)

    total_files = len(files_to_convert)
    if total_files == 0:
        console.print("[yellow]No audio files need conversion[/yellow]")
        return

    # Prepare progress display
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[white]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    )

    converted = 0
    errors = 0
    total_saved = 0
    total_duration = 0

    def get_queue_display(current_index, files, statuses):
        """Generate queue display lines"""
        start = max(0, current_index - 2)
        end = min(len(files), current_index + 3)
        lines = []

        for idx in range(start, end):
            if idx >= len(files):
                continue

            file_name = files[idx].name
            # Truncate long filenames
            if len(file_name) > 50:
                file_name = file_name[:47] + "..."

            if idx < current_index:
                status = statuses.get(idx, "pending")
                if status == "success":
                    prefix = "[green]✓[/green]"
                elif status == "error":
                    prefix = "[red]✗[/red]"
                elif status == "skipped":
                    prefix = "[cyan]→[/cyan]"
                else:
                    prefix = "[white]?[/white]"
            elif idx == current_index:
                prefix = "[yellow]▶[/yellow]"
            else:
                prefix = "  "

            lines.append(f"{prefix} {file_name}")

        return lines

    with Live(
        Group(
            progress,
            Panel(
                "Queue initializing...",
                title=f"Audio Conversion Queue ({output_format.upper()})",
                border_style="magenta",
            ),
        ),
        refresh_per_second=4,
        console=console,
    ) as live:
        total_task = progress.add_task(
            f"[cyan]Converting to {output_format.upper()}",
            total=total_files
        )
        statuses = {}

        for i, file_path in enumerate(files_to_convert):
            # Check if file still exists before processing
            if not file_path.exists():
                console.print(f"[yellow]File no longer exists, skipping: {file_path.name}[/yellow]")
                statuses[i] = "skipped"
                progress.update(total_task, advance=1)
                continue
            
            # Create individual file progress task
            file_task = progress.add_task(
                f"Converting {file_path.name}", total=100, start=False
            )
            progress.start_task(file_task)
            
            # Update queue display
            queue_display = get_queue_display(i, files_to_convert, statuses)
            queue_display.append("")
            queue_display.append(f"[bold]Format:[/bold] {output_format.upper()} @ {audio_quality}")
            queue_display.append(f"[bold]Codec:[/bold] {codec}")
            if sample_rate:
                queue_display.append(f"[bold]Sample Rate:[/bold] {sample_rate} Hz")
            queue_display.append("")
            queue_display.append(f"Space difference: {format_size(abs(total_saved))}")
            if total_duration > 0:
                queue_display.append(f"Total duration: {format_duration(total_duration)}")

            queue_lines = "\n".join(queue_display)
            queue_panel = Panel(
                queue_lines,
                title=f"Audio Conversion Queue ({output_format.upper()})",
                border_style="magenta",
            )
            live.update(Group(progress, queue_panel))

            try:
                success, size_diff, audio_info = convert_audio(
                    file_path,
                    output_format,
                    audio_quality,
                    codec,
                    sample_rate,
                    keep_original,
                    progress,
                    file_task,
                )
                if success:
                    total_saved += size_diff
                    converted += 1
                    statuses[i] = "success"
                    if audio_info and audio_info.get("duration"):
                        total_duration += audio_info["duration"]
                else:
                    statuses[i] = "skipped"
            except FileNotFoundError:
                console.print(f"[yellow]File was moved/deleted during processing: {file_path.name}[/yellow]")
                statuses[i] = "skipped"
            except Exception as e:
                log_error(file_path, str(e))
                errors += 1
                statuses[i] = "error"

            # Remove individual file progress task and update total
            progress.remove_task(file_task)
            progress.update(total_task, advance=1)

    # Print summary
    console.print(f"\n[green]Converted {converted}/{total_files} files successfully[/green]")
    if total_saved > 0:
        console.print(f"Space saved: [green]{format_size(total_saved)}[/green]")
    else:
        console.print(f"Space used: [red]{format_size(abs(total_saved))}[/red]")
    if total_duration > 0:
        console.print(f"Total audio duration: {format_duration(total_duration)}")
    if errors > 0:
        console.print(f"[red]Encountered {errors} errors - see {ERROR_LOG} for details[/red]")
    else:
        if Path(ERROR_LOG).exists():
            Path(ERROR_LOG).unlink()


if __name__ == "__main__":
    main()
