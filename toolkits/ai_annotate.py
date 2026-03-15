#!/usr/bin/env python3
"""AI-based trajectory annotation using a VLM API (e.g. GPT-4o).

Scores trajectories group-by-group: for each group, keyframes from all
trajectories are sent in a single API call together with the task instruction.
Supports optional multi-perspective evaluation.

Usage::

    python toolkits/ai_annotate.py \\
        --dataset data.json \\
        --output data_annotated.json \\
        --api_url https://api3.xhub.chat/v1/chat/completions \\
        --api_key sk-xxx \\
        --model gpt-4o \\
        --n_keyframes 8

    # With perspectives:
    python toolkits/ai_annotate.py \\
        --dataset data.json --output data_annotated.json \\
        --api_url ... --api_key ... --model gpt-4o \\
        --perspectives task_completion motion_quality safety
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rlinf.data.preference_data import get_keyframes
from rlinf.data.trajectory_dataset import ScoringResult, TrajectoryDataset

# ---------------------------------------------------------------------------
# Default prompt template
# ---------------------------------------------------------------------------

DEFAULT_PROMPT_TEMPLATE = """\
You are evaluating {num_trajectories} robot manipulation trajectories for the task:
"{language_instruction}"

Each trajectory is shown as a sequence of keyframes (images taken at regular intervals).
Compare all trajectories and provide scores.

{perspectives}

Output ONLY a valid JSON object in this exact format:
{output_format}
"""

_PERSPECTIVE_DESCRIPTIONS = {
    "task_completion": "How well does the robot complete the described task?",
    "motion_quality": "How smooth and efficient is the robot's motion?",
    "safety": "Does the robot avoid collisions and dangerous movements?",
}


def _render_perspectives(perspectives: Optional[list[str]]) -> str:
    if not perspectives:
        return "Provide a single overall score (0-10) for each trajectory."
    lines = ["Evaluate each trajectory on the following perspectives:"]
    for i, p in enumerate(perspectives, 1):
        desc = _PERSPECTIVE_DESCRIPTIONS.get(p, f"Evaluate the {p} aspect.")
        lines.append(f"{i}. {p}: {desc}")
    lines.append(
        "\nFor each trajectory, provide a score (0-10) for EACH perspective, "
        "plus an overall score (average of perspectives)."
    )
    return "\n".join(lines)


def _render_output_format(
    num_trajectories: int, perspectives: Optional[list[str]]
) -> str:
    if not perspectives:
        entries = []
        for i in range(1, num_trajectories + 1):
            entries.append(
                f'    {{"trajectory_id": {i}, "score": <0-10>, '
                f'"rationale": "<brief explanation>"}}'
            )
        return '{\n  "trajectories": [\n' + ",\n".join(entries) + "\n  ]\n}"
    entries = []
    for i in range(1, num_trajectories + 1):
        persp_block = ", ".join(
            f'"{p}": {{"score": <0-10>, "rationale": "..."}}'
            for p in perspectives
        )
        entries.append(
            f'    {{"trajectory_id": {i}, '
            f'"perspectives": {{{persp_block}}}, '
            f'"overall_score": <0-10>, '
            f'"overall_rationale": "..."}}'
        )
    return '{\n  "trajectories": [\n' + ",\n".join(entries) + "\n  ]\n}"


# ---------------------------------------------------------------------------
# Keyframe extraction + base64 encoding
# ---------------------------------------------------------------------------


def _load_keyframes_base64(
    traj, n_keyframes: int
) -> list[str]:
    """Load keyframes from a trajectory and encode as base64 JPEG."""
    from rlinf.data.preference_data import EpisodeRecord

    episode = EpisodeRecord(
        task_description=traj.language_instruction,
        cumulative_reward=traj.cumulative_reward or 0.0,
        success=traj.success or False,
        episode_length=traj.episode_length or 0,
        video_path=traj.video_path,
    )
    frames = get_keyframes(episode, n_keyframes)
    encoded = []
    for frame in frames:
        from io import BytesIO

        from PIL import Image

        img = Image.fromarray(frame)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        encoded.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return encoded


# ---------------------------------------------------------------------------
# API message construction
# ---------------------------------------------------------------------------


def build_group_messages(
    group_trajectories: list,
    group_keyframes: list[list[str]],
    language_instruction: str,
    perspectives: Optional[list[str]],
    prompt_template: str,
) -> list[dict]:
    """Build API messages for a single group annotation call."""
    num_t = len(group_trajectories)
    persp_text = _render_perspectives(perspectives)
    fmt_text = _render_output_format(num_t, perspectives)
    rendered = prompt_template.format(
        num_trajectories=num_t,
        language_instruction=language_instruction,
        perspectives=persp_text,
        output_format=fmt_text,
    )

    content: list[dict] = [{"type": "text", "text": rendered}]
    for traj_idx, (traj, kf_b64) in enumerate(
        zip(group_trajectories, group_keyframes)
    ):
        content.append({
            "type": "text",
            "text": (
                f"\n--- Trajectory {traj_idx + 1} "
                f"(model: {traj.model_name}) ---"
            ),
        })
        for frame_idx, b64 in enumerate(kf_b64):
            content.append({
                "type": "text",
                "text": f"Frame {frame_idx + 1}/{len(kf_b64)}:",
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

    return [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator of robot manipulation trajectories."
            ),
        },
        {"role": "user", "content": content},
    ]


# ---------------------------------------------------------------------------
# API call + response parsing
# ---------------------------------------------------------------------------


def call_vlm_api(
    api_url: str,
    api_key: str,
    model_name: str,
    messages: list[dict],
    max_retries: int = 3,
) -> str:
    """Call the VLM API and return the response text."""
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": 2048,
                    "temperature": 0.1,
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable")


def parse_ai_response(
    response_text: str,
    num_trajectories: int,
    perspectives: Optional[list[str]],
) -> Optional[dict]:
    """Parse the AI response JSON, returning structured scores.

    Returns:
        dict with keys "trajectories", each containing scores/rationales.
        None if parsing fails.
    """
    import re

    text = response_text.strip()

    # Strategy 1: direct parse
    for candidate in [text]:
        try:
            data = json.loads(candidate)
            if _validate_response(data, num_trajectories, perspectives):
                return data
        except json.JSONDecodeError:
            pass

    # Strategy 2: extract ```json ... ``` block
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if _validate_response(data, num_trajectories, perspectives):
                return data
        except json.JSONDecodeError:
            pass

    # Strategy 3: extract outermost { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if _validate_response(data, num_trajectories, perspectives):
                return data
        except json.JSONDecodeError:
            pass

    return None


def _validate_response(
    data: dict, num_trajectories: int, perspectives: Optional[list[str]]
) -> bool:
    if "trajectories" not in data:
        return False
    if len(data["trajectories"]) != num_trajectories:
        return False
    for entry in data["trajectories"]:
        if perspectives:
            if "perspectives" not in entry:
                return False
        else:
            if "score" not in entry:
                return False
    return True


# ---------------------------------------------------------------------------
# Main annotation loop
# ---------------------------------------------------------------------------


def annotate_with_ai(
    dataset_path: str,
    output_path: str,
    api_url: str,
    api_key: str,
    model_name: str,
    prompt_template: Optional[str] = None,
    n_keyframes: int = 8,
    scorer_name: Optional[str] = None,
    perspectives: Optional[list[str]] = None,
    max_retries: int = 3,
) -> None:
    """Annotate a TrajectoryDataset using a VLM API."""
    if prompt_template is None:
        prompt_template = DEFAULT_PROMPT_TEMPLATE
    if scorer_name is None:
        scorer_name = f"ai_{model_name}"

    dataset = TrajectoryDataset.load(dataset_path)
    total_groups = len(dataset.groups)
    print(f"Loaded dataset with {len(dataset.trajectories)} trajectories, "
          f"{total_groups} groups")

    # Collect scores per perspective (or overall)
    if perspectives:
        perspective_scores: dict[str, dict[int, Optional[float]]] = {
            p: {} for p in perspectives
        }
        perspective_rationales: dict[str, dict[int, str]] = {
            p: {} for p in perspectives
        }
        overall_scores: dict[int, Optional[float]] = {}
        overall_rationales: dict[int, str] = {}
    else:
        overall_scores = {}
        overall_rationales = {}

    for g_idx, group in enumerate(dataset.groups):
        group_trajs = [dataset.trajectories[i] for i in group]
        lang = group_trajs[0].language_instruction

        print(f"  Annotating group {g_idx + 1}/{total_groups} "
              f"({len(group)} trajectories)...")

        # Load keyframes
        group_kf = []
        for traj in group_trajs:
            kf = _load_keyframes_base64(traj, n_keyframes)
            group_kf.append(kf)

        # Build and call API
        messages = build_group_messages(
            group_trajs, group_kf, lang, perspectives, prompt_template
        )
        try:
            response_text = call_vlm_api(
                api_url, api_key, model_name, messages, max_retries
            )
        except Exception as e:
            print(f"    API call failed for group {g_idx}: {e}")
            continue

        parsed = parse_ai_response(response_text, len(group), perspectives)
        if parsed is None:
            print(f"    Failed to parse response for group {g_idx}")
            continue

        # Extract scores
        for t_idx_in_group, entry in enumerate(parsed["trajectories"]):
            global_idx = group[t_idx_in_group]
            if perspectives:
                for p in perspectives:
                    p_data = entry.get("perspectives", {}).get(p, {})
                    score = p_data.get("score")
                    if score is not None and 0 <= score <= 10:
                        perspective_scores[p][global_idx] = float(score)
                    rationale = p_data.get("rationale", "")
                    if rationale:
                        perspective_rationales[p][global_idx] = rationale
                os = entry.get("overall_score")
                if os is not None and 0 <= os <= 10:
                    overall_scores[global_idx] = float(os)
                or_text = entry.get("overall_rationale", "")
                if or_text:
                    overall_rationales[global_idx] = or_text
            else:
                score = entry.get("score")
                if score is not None and 0 <= score <= 10:
                    overall_scores[global_idx] = float(score)
                rationale = entry.get("rationale", "")
                if rationale:
                    overall_rationales[global_idx] = rationale

    # Write scoring results
    if perspectives:
        for p in perspectives:
            dataset.scoring_results.append(ScoringResult(
                scorer_name=scorer_name,
                scorer_type="ai",
                scores=perspective_scores[p],
                prompt=prompt_template,
                model_name=model_name,
                perspective_name=p,
                rationales=perspective_rationales.get(p) or None,
            ))
        # Also add overall
        dataset.scoring_results.append(ScoringResult(
            scorer_name=scorer_name,
            scorer_type="ai",
            scores=overall_scores,
            prompt=prompt_template,
            model_name=model_name,
            perspective_name=None,
            rationales=overall_rationales or None,
        ))
    else:
        dataset.scoring_results.append(ScoringResult(
            scorer_name=scorer_name,
            scorer_type="ai",
            scores=overall_scores,
            prompt=prompt_template,
            model_name=model_name,
            perspective_name=None,
            rationales=overall_rationales or None,
        ))

    dataset.save(output_path)
    scored_count = len([v for v in overall_scores.values() if v is not None])
    print(f"Done. Scored {scored_count}/{len(dataset.trajectories)} trajectories.")
    print(f"Saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="AI trajectory annotation")
    parser.add_argument("--dataset", required=True, help="TrajectoryDataset JSON path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--api_url", required=True, help="VLM API endpoint URL")
    parser.add_argument("--api_key", required=True, help="API key")
    parser.add_argument("--model", default="gpt-4o", help="Model name")
    parser.add_argument(
        "--n_keyframes", type=int, default=8, help="Keyframes per trajectory"
    )
    parser.add_argument("--scorer_name", default=None, help="Scorer name override")
    parser.add_argument(
        "--perspectives", nargs="*", default=None,
        help="Perspective names (e.g. task_completion motion_quality safety)",
    )
    parser.add_argument(
        "--max_retries", type=int, default=3, help="Max API retries per group"
    )
    args = parser.parse_args()

    annotate_with_ai(
        dataset_path=args.dataset,
        output_path=args.output,
        api_url=args.api_url,
        api_key=args.api_key,
        model_name=args.model,
        n_keyframes=args.n_keyframes,
        scorer_name=args.scorer_name,
        perspectives=args.perspectives,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
