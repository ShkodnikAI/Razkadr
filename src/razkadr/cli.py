"""Command line entry point.

Subcommands: validate (real, naryad 002/035), eval-check (real, naryad 003:
integrity of the reference set), index (the full indexing
pipeline: ingest + frames + transcript + chunks + ocr + embeddings index,
naryads 005-010 glued together with cache, resume and a final summary in
naryad 011), gallery (real, naryad 041: static HTML of the run's frames
and selection results), select (real, naryad 014: preset + hybrid search
-> draft Storyboard) and stubs for the remaining pipeline commands
(text/build/eval) with a unified interface
--config/--force/--runs-dir/--resume; their logic arrives with later
naryads.
Exit codes: 0 ok, 1 invalid document, 2 file/IO error, 3 command not
implemented.
"""

import argparse
import json
import sys
from pathlib import Path

from razkadr import __version__

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_FILE_ERROR = 2
EXIT_NOT_IMPLEMENTED = 3

# stub command -> owning naryad (shown in help and stub message)
STUB_COMMANDS = {
    "text": "017",
    "build": "028",
    "eval": "015",
}


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate a JSON document. 0 = ok, 1 = invalid, 2 = file error.

    Without --kind the file is validated as a Storyboard (naryad 002);
    --kind selects a stage artifact schema (naryad 035, docs/artifacts.md).
    Presets (naryad 013) are YAML by definition: when the file does not
    parse as JSON, --kind preset falls back to a YAML load.
    """
    path = Path(args.file)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"razkadr validate: cannot read {path}: {exc.strerror or exc}", file=sys.stderr)
        return EXIT_FILE_ERROR
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        if args.kind != "preset":
            print(f"razkadr validate: invalid JSON in {path}: {exc}", file=sys.stderr)
            return EXIT_FILE_ERROR
        try:
            import yaml

            data = yaml.safe_load(raw)
        except yaml.YAMLError as yexc:
            print(f"razkadr validate: invalid YAML in {path}: {yexc}", file=sys.stderr)
            return EXIT_FILE_ERROR

    from razkadr.validation import validate_document

    issues = validate_document(data, kind=args.kind)
    if issues:
        for issue in issues:
            print(issue, file=sys.stderr)
        return EXIT_INVALID
    print("ok")
    return EXIT_OK


def _cmd_evalcheck(args: argparse.Namespace) -> int:
    """Integrity of the reference set (naryad 003): every video has a
    labels file and a SOURCES entry; labels are valid against
    eval_labels.v1.json, have the role minimums (1 hero, 1 ingredients,
    5 steps), no duplicate times within --tolerance-s (default 2, the
    contract's matching tolerance) and timecodes inside the video
    duration via ffprobe.

    0 = "N видео, N разметок, нарушений нет", 1 = violations listed,
    2 = the check itself cannot run (missing directory, no ffprobe).
    """
    from razkadr.evalcheck import EvalCheckError, run_eval_check, summarize

    try:
        result = run_eval_check(Path(args.dir), tolerance_s=args.tolerance_s)
    except EvalCheckError as exc:
        print(f"razkadr eval-check: {exc}", file=sys.stderr)
        return EXIT_FILE_ERROR
    if not result.ok:
        for violation in result.violations:
            print(violation, file=sys.stderr)
    print(summarize(result), file=sys.stderr if not result.ok else sys.stdout)
    return EXIT_OK if result.ok else EXIT_INVALID


def _fmt_amount(amount: float) -> str:
    """Cost for the console: 6-decimal precision, trailing zeros trimmed."""
    return f"{amount:.6f}".rstrip("0").rstrip(".")


def _cmd_index(args: argparse.Namespace) -> int:
    """The indexing pipeline (naryad 011): ingest (005), frames (006),
    transcript (007), chunks (008), ocr (009), embeddings index (010).

    Stages run in order through RunContext; each stage is cached by its
    inputs+params and a re-run with --resume RUN_ID continues from the
    failed stage without recomputing the earlier ones (--force re-runs
    everything). After a successful run run.json carries the final
    summary (stage times, provider cost, artifact counts) and the console
    prints it. Stage and provider failures print a clear message, the
    resume hint and exit 2.
    """
    from razkadr.config import load_config
    from razkadr.providers.base import ProviderError
    from razkadr.runctx import RunContext
    from razkadr.stages.chunks import ChunksError, run_chunks
    from razkadr.stages.frames import FramesError, run_frames
    from razkadr.stages.index import IndexStageError, run_index
    from razkadr.stages.ingest import IngestError, run_ingest
    from razkadr.stages.ocr import OcrError, ocr_params, plan_ocr_candidates, run_ocr
    from razkadr.stages.transcript import TranscriptError, run_transcript

    config = load_config(args.config)
    runs_dir = args.runs_dir or config.get("runs_dir", "runs")
    if args.resume and not (Path(runs_dir) / args.resume / "run.json").is_file():
        print(
            f"razkadr index: nothing to resume: {runs_dir}/{args.resume} has no run.json",
            file=sys.stderr,
        )
        return EXIT_FILE_ERROR
    ctx = RunContext(runs_dir, run_id=args.resume, config=config, force=args.force)
    try:
        source = run_ingest(ctx, args.input, config)
        frames_doc = run_frames(ctx, config)
        transcript_doc = run_transcript(ctx, config)
        chunks_doc = run_chunks(ctx, config)
        ocr_doc = run_ocr(ctx, config)
        manifest, space_counts = run_index(ctx, config)
    except (IngestError, FramesError, TranscriptError, ChunksError, OcrError, IndexStageError, ProviderError) as exc:
        print(f"razkadr index: {exc}", file=sys.stderr)
        last = ctx.stages[-1] if ctx.stages else None
        if last and last.get("status") == "failed":
            print(f"  run {ctx.run_id} stopped at stage '{last['name']}'", file=sys.stderr)
            print(f"  resume: razkadr index {args.input} --resume {ctx.run_id}", file=sys.stderr)
        return EXIT_FILE_ERROR
    has_audio = (ctx.dir / "audio.wav").is_file()
    audio = "audio.wav" if has_audio else "none (source has no audio track)"
    frames = frames_doc["frames"]
    rejected = [f for f in frames if f.get("rejected")]
    accepted = len(frames) - len(rejected)
    # the plan is a pure CPU function (no provider calls), so the summary
    # works for cached stages too (naryad 009: the share is in the summary);
    # when no [vlm] provider is configured the stage is skipped: nothing sent
    vlm_configured = bool((config.get("vlm") or {}).get("provider"))
    plan = plan_ocr_candidates(frames_doc, ctx.dir, ocr_params(config)) if vlm_configured else []
    sent = len(plan)
    share = sent / accepted if accepted else 0.0
    transcript_s = sum(s["t_end"] - s["t_start"] for s in transcript_doc["segments"])
    # the final summary goes into run.json (schema run.v1.1) before printing
    summary = ctx.write_summary(
        artifacts={
            "frames_candidates": len(frames),
            "frames_accepted": accepted,
            "frames_rejected": len(rejected),
            "ocr_results": len(ocr_doc["results"]),
            "ocr_frames_sent": sent,
            "transcript_segments": len(transcript_doc["segments"]),
            "transcript_s": round(transcript_s, 3),
            "chunks": len(chunks_doc["chunks"]),
            "index_vectors": manifest["num_vectors"] if manifest else 0,
        }
    )
    stage_line = "ingest ok, frames ok, transcript ok, chunks ok, ocr ok"
    stage_line += ", index ok" if manifest is not None else ", index skipped"
    print(f"run {ctx.run_id}: {stage_line}")
    print(
        f"  video:      {source['width']}x{source['height']} @ {source['fps']:g} fps, {source['duration_s']:g} s"
    )
    print(f"  audio:      {audio}")
    print(
        f"  frames:     {accepted} accepted, {len(rejected)} rejected of {len(frames)} candidates"
    )
    print(
        f"  ocr:        {len(ocr_doc['results'])} results ({sent} of {accepted} frames sent, {share:.1%})"
    )
    if has_audio:
        print(
            f"  transcript: lang={transcript_doc['lang']}, {len(transcript_doc['segments'])} segments, {round(transcript_s, 1):g}s"
        )
        print(f"  chunks:     {len(chunks_doc['chunks'])}")
    if manifest is not None:
        counts = " ".join(
            f"{space} {space_counts.get(space, 0)}" for space in ("frames", "chunks", "ocr")
        )
        print(
            f"  index:      {manifest['num_vectors']} vectors ({counts}), "
            f"dim {manifest['dim']}, {manifest['provider']}/{manifest['model']}"
        )
    else:
        print("  index:      skipped (nothing to embed or no [embeddings] provider)")
    cost = summary["provider_cost"]
    if cost["calls"]:
        print(
            f"  cost:       {_fmt_amount(cost['amount'])} {cost['currency']} "
            f"({cost['calls']} calls, {cost['cache_hits']} cache hits)"
        )
    else:
        print("  cost:       0 (no provider calls)")
    times = summary["stage_times_s"]
    print("  time:       " + ", ".join(f"{name} {times[name]:g}s" for name in times))
    print(f"  dir:        {ctx.dir}")
    return EXIT_OK


def _cmd_gallery(args: argparse.Namespace) -> int:
    """Static HTML gallery of a run (naryad 041): frames sheet with copyable
    timecodes, selection view when storyboard.json exists, ground-truth
    labels marking with --labels. Writes <run_dir>/gallery.html (or --out).
    0 = ok, 2 = the gallery cannot be built (no run dir, no frames.json,
    invalid storyboard/labels JSON)."""
    from razkadr.gallery import GalleryError, run_gallery

    try:
        target = run_gallery(
            args.run_dir,
            storyboard=args.storyboard,
            labels=args.labels,
            every_s=args.every_s,
            tolerance_s=args.tolerance_s,
            out=args.out,
        )
    except GalleryError as exc:
        print(f"razkadr gallery: {exc}", file=sys.stderr)
        return EXIT_FILE_ERROR
    total = data_summary(args.run_dir)
    print(f"gallery: {target} ({total})")
    return EXIT_OK


def data_summary(run_dir: str) -> str:
    """One-line frame totals for the gallery console message."""
    from razkadr.gallery import load_frames

    try:
        frames = load_frames(Path(run_dir))
    except Exception:  # noqa: BLE001 - summary is best-effort only
        return "frames: ?"
    rejected = sum(1 for f in frames if f.get("rejected"))
    return f"frames: {len(frames) - rejected} accepted, {rejected} rejected, {len(frames)} total"


def _cmd_select(args: argparse.Namespace) -> int:
    """Episode selection (naryad 014): preset + hybrid search (012) -> a
draft Storyboard without text, valid per naryad 002. Episodes marked
    user_edited in the existing storyboard survive re-selection. Writes
    <run_dir>/storyboard.json (or --out) and the selection report
    select.json next to it. 0 = ok, 2 = the selection cannot run."""
    from pathlib import Path as _Path

    from razkadr.config import load_config
    from razkadr.presets import PresetError
    from razkadr.providers.base import ProviderError
    from razkadr.providers.layer import ProviderLayer
    from razkadr.runctx import RunContext
    from razkadr.stages.select import SelectError, run_select

    run_dir = _Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"razkadr select: run directory not found: {run_dir}", file=sys.stderr)
        return EXIT_FILE_ERROR
    preset_path = _Path(args.preset)
    if not preset_path.is_file():
        print(
            f"razkadr select: preset not found: {preset_path} (pass a YAML path, e.g. --preset presets/recipe.yaml)",
            file=sys.stderr,
        )
        return EXIT_FILE_ERROR
    config = load_config(args.config)
    try:
        ctx = RunContext(run_dir.parent, run_id=run_dir.name, config=config)
        layer = ProviderLayer(config, ctx)
        embeddings = layer.embeddings(stage="select")
        llm = layer.llm(stage="select") if (config.get("llm") or {}).get("provider") else None
        storyboard, report = run_select(run_dir, preset_path, config, embeddings=embeddings, llm=llm)
    except (SelectError, PresetError, ProviderError) as exc:
        print(f"razkadr select: {exc}", file=sys.stderr)
        return EXIT_FILE_ERROR
    out = _Path(args.out) if args.out else run_dir / "storyboard.json"
    report_path = run_dir / "select.json"
    try:
        out.write_text(json.dumps(storyboard, ensure_ascii=False, indent=2), encoding="utf-8")
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"razkadr select: cannot write artifacts: {exc}", file=sys.stderr)
        return EXIT_FILE_ERROR
    counts: dict[str, int] = {}
    for ep in storyboard["episodes"]:
        counts[ep["role"]] = counts.get(ep["role"], 0) + 1
    role_line = ", ".join(f"{role} {n}" for role, n in counts.items()) or "no episodes"
    print(f"select: {out} ({len(storyboard['episodes'])} episodes: {role_line})")
    if report["warnings"]:
        for warning in report["warnings"]:
            print(f"  warning: {warning}", file=sys.stderr)
    print(f"  report:   {report_path}")
    return EXIT_OK


def _cmd_stub(args: argparse.Namespace) -> int:
    print(
        f"razkadr {args.cmd}: not implemented yet (planned: naryad No{args.naryad})",
        file=sys.stderr,
    )
    return EXIT_NOT_IMPLEMENTED


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="Path to razkadr.toml (default: search)")
    parser.add_argument("--force", action="store_true", help="Re-run stages, ignore stage cache")
    parser.add_argument(
        "--runs-dir", default=None, help="Directory for runs (default: config runs_dir)"
    )
    parser.add_argument(
        "--resume", default=None, metavar="RUN_ID", help="Reuse an existing run directory"
    )


def build_parser() -> argparse.ArgumentParser:
    """The razkadr CLI parser (split out of main for tests and help checks)."""
    parser = argparse.ArgumentParser(prog="razkadr", description="Video in, deck out.")
    parser.add_argument("--version", action="version", version=f"razkadr {__version__}")
    parser.add_argument(
        "--verbose", action="store_true", help="Debug logging (JSON lines to stderr)"
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    from razkadr.validation import KINDS

    p_validate = sub.add_parser("validate", help="Validate a Storyboard or artifact JSON file")
    p_validate.add_argument(
        "--kind",
        choices=sorted(KINDS),
        default="storyboard",
        help="Document kind: storyboard (default) or a stage artifact",
    )
    p_validate.add_argument("file", help="Path to the JSON file")
    p_validate.set_defaults(func=_cmd_validate)

    p_index = sub.add_parser(
        "index",
        help="Ingest a video, extract frames, transcribe, chunk the text, OCR captions, embed (naryads No005-No011)",
    )
    p_index.add_argument("input", help="Path to a video file, or a URL for yt-dlp")
    _add_common_args(p_index)
    p_index.set_defaults(func=_cmd_index)

    p_evalcheck = sub.add_parser(
        "eval-check",
        help="Check the reference set: labels, SOURCES, timecodes (naryad No003)",
    )
    p_evalcheck.add_argument("dir", help="Reference set directory, e.g. data/eval")
    p_evalcheck.add_argument(
        "--tolerance-s",
        type=float,
        default=2.0,
        help="Time matching tolerance in seconds (default: 2)",
    )
    p_evalcheck.set_defaults(func=_cmd_evalcheck)

    p_gallery = sub.add_parser(
        "gallery",
        help="Static HTML page: frames with timecodes, selection view, labels (naryad No041)",
    )
    p_gallery.add_argument("run_dir", help="Run directory (contains frames.json), e.g. runs/<id>")
    p_gallery.add_argument(
        "--storyboard",
        default=None,
        metavar="PATH",
        help="Storyboard JSON (default: <run_dir>/storyboard.json when present)",
    )
    p_gallery.add_argument(
        "--labels",
        default=None,
        metavar="PATH",
        help="Ground-truth labels JSON (data/eval/labels/<id>.json) to mark reference frames",
    )
    p_gallery.add_argument(
        "--every",
        dest="every_s",
        type=float,
        default=0.0,
        metavar="N",
        help="Label sheet density: at most one frame every N seconds (0 = all, default)",
    )
    p_gallery.add_argument(
        "--tolerance-s",
        type=float,
        default=2.0,
        help="Labels matching tolerance in seconds (default: 2)",
    )
    p_gallery.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Output HTML path (default: <run_dir>/gallery.html)",
    )
    p_gallery.set_defaults(func=_cmd_gallery)

    p_select = sub.add_parser(
        "select",
        help="Episode selection: preset + index -> draft Storyboard (naryad No014)",
    )
    p_select.add_argument("run_dir", help="Run directory with the index (razkadr index output)")
    p_select.add_argument(
        "--preset",
        default="presets/recipe.yaml",
        help="Preset YAML path (default: presets/recipe.yaml)",
    )
    p_select.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Output Storyboard path (default: <run_dir>/storyboard.json)",
    )
    p_select.add_argument("--config", default=None, help="Path to razkadr.toml (default: search)")
    p_select.set_defaults(func=_cmd_select)

    for name, naryad in STUB_COMMANDS.items():
        p_stub = sub.add_parser(name, help=f"[stub] planned by naryad No{naryad}")
        _add_common_args(p_stub)
        p_stub.set_defaults(func=_cmd_stub, cmd=name, naryad=naryad)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from razkadr.log import setup_logging

    setup_logging(verbose=getattr(args, "verbose", False))

    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_OK
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
