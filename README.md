# scope-date-type (DATE / TYPE)

DATE / TYPE is a Scope video pipeline that turns live typing in the prompt box into temporal-photography structure.

This plugin is *read-only* with respect to prompts:
- it reads prompt text as a live data stream
- it does not interpret semantics
- it does not use keyword triggers
- it does not call an LLM

Instead, it measures *how writing is happening* (cadence, revision, pause, density) and uses those signals to control time-slicing and repeated exposure.

## Conceptual grounding (why this is not a gimmick)

Two families of temporal photography motivate the visual grammar:

1) Slit-scan / strip-scan:
   - treat one image axis as time, so columns/rows embody different moments.

2) Marey-style chronophotography:
   - repeated exposures on a single plate collapse many instants into one composite trace.

A third principle (text as architecture) motivates why the “band field” is an intentional form:
- the screen becomes a columnar partition with adjustable permeability (soft seams).

Finally, a date constraint makes the apparatus “of the day”:
- the system is deterministic within a day but changes across days.
- date changes sampling/accumulation rules (not the camera input).

## Install

In Scope: Settings → Plugins → install from local folder (during development)
or install from a git URL once you push it:

    git+https://github.com/YOUR_USERNAME/scope-date-type.git

## Use

Select the main pipeline: **DATE / TYPE**

Type in the prompt box while a camera/video feed is running.

### Modes

- **slit_scan** (default): time becomes columns/rows
- **multi_exposure**: repeated exposure “plate” (weighted accumulation)

### Controls

- Enabled: toggle effect
- Buffer Length (load-time): temporal window size (reload pipeline to change)
- Band Count: number of partitions (slit-scan)
- Orientation: vertical/horizontal
- Mix: crossfade between original and effect
- Permeability: softens seams with a 1D blur along scan axis
- Text Influence: how strongly typing dynamics shape the system
- Date Influence: how strongly today's date alters the sampling rules
- Memory Decay: decay speed for multi-exposure
- Exposure Strength: intensity of accumulation

## What the prompt controls (non-semantic)

The plugin derives structural signals:
- length (characters/words)
- cadence (change rate)
- revision intensity (deletion pressure)
- hesitation (time since last change)
- punctuation/whitespace density

These drive phase shifts, persistence, and partition behavior.
No keywords. No topic detection.

## Manual test plan

1) Type slowly (few changes, long pauses):
   - slit-scan should stabilize, seams soften; multi-exposure should clear.

2) Type fast (continuous edits):
   - slit-scan should become more active; multi-exposure should accumulate more trace density.

3) Hold backspace / delete:
   - revision increases structural fragmentation (without any special words).

4) Paste a paragraph:
   - cadence spike should cause a quick temporal reconfiguration.

5) Clear the prompt:
   - abrupt structural event; system should remain stable, not crash.

6) Toggle Date Influence 0 → 1:
   - with 0, apparatus depends only on typing signals
   - with 1, daily permutation/bias should be evident (deterministic per day)
