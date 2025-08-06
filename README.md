<!--
SPDX-FileCopyrightText: 2025 mohamed hamdi <haamdi@outlook.com>

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# audconv

Command-line audio conversion tool with batch processing.

## Usage

```bash
# Convert single file
audconv audio.wav -f mp3

# Batch convert directory
audconv --all -f opus

# Recursive processing
audconv --all --recursive -f flac
```

## Options

- `-f, --format`: Output format (m4a, mp3, ogg, opus, flac, wav)
- `-q, --quality`: Bitrate (128k, 192k, 320k)
- `-a, --all`: Process all files in directory
- `-r, --recursive`: Include subdirectories
- `-k, --keep-original`: Keep source files

## Requirements

- Python 3.12+
- ffmpeg/ffprobe

## Installation

```bash
make configure  # Setup
make build      # Build executable
make install    # Install to ~/.local/bin
```