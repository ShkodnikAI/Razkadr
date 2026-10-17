"""Command line entry point.

Subcommands: validate (real, naryad 002/035), eval-check (real, naryad 003:
integrity of the reference set), index (the full indexing
pipeline: ingest + frames + transcript + chunks + ocr + embeddings index,
naryads 005-010 glued together with cache, resume and a final summary in
naryad 011), gallery (real, naryad 041: static HTML of the run's frames
and selection results), select (real, naryad 014: preset + hybrid search
-> draft Storyboard), eval (real, naryad 015: hit@k over the reference
set), layout (real, naryad 021: Storyboard -> slides.json), render (real,
naryad 022: slides.json -> deck.pptx on the template), export-pdf (real,
naryad 024: deck.pptx -> deck.pdf via LibreOffice headless), template-check
(real, naryad 023: user template report per the naming convention) and stubs for
the remaining pipeline commands (text/build) with a unified interface
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


def _eval_pct(hits: int, moments: int) -> str:
    return "n/a" if not moments else f"{hits / moments:.1%}"


def _cmd_eval(args: argparse.Namespace) -> int:
    """Eval harness (naryad 015): hit@k of the hybrid search over the
    labeled reference set for each --config, cost per video minute,
    reproducible report.json + report.md. 0 = ok, 2 = cannot run."""
    from razkadr.evalharness import EvalError, run_eval

    k_list = [int(part) for part in str(args.k).split(",") if part.strip()]
    try:
        json_path, md_path = run_eval(
            args.dir,
            args.config,
            k_list=k_list,
            tolerance=args.tolerance_s,
            preset_path=args.preset,
            out_dir=args.out_dir,
            keep_runs=args.keep_runs,
        )
    except EvalError as exc:
        print(f"razkadr eval: {exc}", file=sys.stderr)
        return EXIT_FILE_ERROR
    report = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"eval: {md_path} (+ {json_path.name})")
    for name, result in report["results"].items():
        roles = ", ".join(
            f"{role} {_eval_pct(data['hits_by_k']['3'], data['moments'])}"
            for role, data in result["roles"].items()
            if data["videos"]
        )
        print(f"  {name}: {roles} ({result['cost']['calls']} calls, {result['cost']['amount']:.6f} {result['cost']['currency']})")
    for warning in report["warnings"]:
        print(f"  warning: {warning}", file=sys.stderr)
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


def _cmd_layout(args: argparse.Namespace) -> int:
    """Layout engine (naryad 021): Storyboard -> slides.json. 0 = ok,
    1 = invalid document, 2 = file/IO or config error."""
    from razkadr.config import load_config
    from razkadr.stages.layout import LayoutError, layout_storyboard, write_slides
    from razkadr.validation import validate_document

    path = Path(args.storyboard)
    try:
        storyboard = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"razkadr layout: cannot read {path}: {exc.strerror or exc}", file=sys.stderr)
        return EXIT_FILE_ERROR
    except json.JSONDecodeError as exc:
        print(f"razkadr layout: invalid JSON in {path}: {exc}", file=sys.stderr)
        return EXIT_FILE_ERROR
    issues = validate_document(storyboard, "storyboard")
    if issues:
        print("razkadr layout: storyboard is invalid:", file=sys.stderr)
        for issue in issues[:5]:
            print(f"  - {issue.path}: {issue.reason}", file=sys.stderr)
        return EXIT_INVALID
    try:
        config = load_config(args.config) if getattr(args, "config", None) else load_config()
        slides = layout_storyboard(storyboard, config, lang=args.lang)
    except LayoutError as exc:
        print(f"razkadr layout: {exc}", file=sys.stderr)
        return EXIT_FILE_ERROR
    out = Path(args.out) if args.out else path.with_name("slides.json")
    write_slides(slides, out)
    total_warnings = sum(len(s["warnings"]) for s in slides["slides"]) + len(slides["warnings"])
    print(f"razkadr layout: {len(slides['slides'])} slide(s) -> {out}" + (f" ({total_warnings} warning(s))" if total_warnings else ""))
    return EXIT_OK


def _cmd_export_html(args: argparse.Namespace) -> int:
    """Self-contained HTML deck (naryad 026). 0 = ok, 2 = file error."""
    from razkadr.render.html import ExportError, export_html

    slides_path = Path(args.slides)
    try:
        slides_doc = json.loads(slides_path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"razkadr export-html: cannot read {slides_path}: {exc.strerror or exc}", file=sys.stderr)
        return EXIT_FILE_ERROR
    except json.JSONDecodeError as exc:
        print(f"razkadr export-html: invalid JSON in {slides_path}: {exc}", file=sys.stderr)
        return EXIT_FILE_ERROR
    storyboard = None
    storyboard_path = Path(args.storyboard) if args.storyboard else slides_path.with_name("storyboard.json")
    if storyboard_path.is_file():
        try:
            storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"razkadr export-html: storyboard {storyboard_path} unreadable ({exc}); timecodes stay as text",
                file=sys.stderr,
            )
    out = Path(args.out)
    try:
        export_html(slides_doc, out, storyboard=storyboard, base_dir=slides_path.parent)
    except ExportError as exc:
        print(f"razkadr export-html: {exc}", file=sys.stderr)
        return EXIT_FILE_ERROR
    print(f"razkadr export-html: {out} (+ {out.parent / 'deck_assets'}/)")
    return EXIT_OK


def _cmd_render(args: argparse.Namespace) -> int:
    """PPTX deck on the reference template (naryad 022). 0 = ok, 1 = invalid
    slides document, 2 = file error (template/frames)."""
    from razkadr.render.pptx import RenderError, render_pptx

    slides_path = Path(args.slides)
    try:
        slides_doc = json.loads(slides_path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"razkadr render: cannot read {slides_path}: {exc.strerror or exc}", file=sys.stderr)
        return EXIT_FILE_ERROR
    except json.JSONDecodeError as exc:
        print(f"razkadr render: invalid JSON in {slides_path}: {exc}", file=sys.stderr)
        return EXIT_FILE_ERROR
    from razkadr.validation import validate_document

    issues = validate_document(slides_doc, kind="slides")
    if issues:
        print("razkadr render: slides document is invalid:", file=sys.stderr)
        for issue in issues:
            print(issue, file=sys.stderr)
        return EXIT_INVALID
    storyboard = None
    storyboard_path = Path(args.storyboard) if args.storyboard else slides_path.with_name("storyboard.json")
    if storyboard_path.is_file():
        try:
            storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"razkadr render: storyboard {storyboard_path} unreadable ({exc}); notes degrade to ids",
                file=sys.stderr,
            )
    out = Path(args.out)
    try:
        stats = render_pptx(
            slides_doc,
            out,
            template=args.template,
            storyboard=storyboard,
            base_dir=slides_path.parent,
            min_font_pt=args.min_font_pt,
        )
    except RenderError as exc:
        print(f"razkadr render: {exc}", file=sys.stderr)
        return EXIT_FILE_ERROR
    suffix = f", {len(stats['warnings'])} warnings" if stats["warnings"] else ""
    print(f"razkadr render: {out} ({stats['slides']} slides, {stats['pictures']} pictures{suffix})")
    return EXIT_OK


def _cmd_export_pdf(args: argparse.Namespace) -> int:
    """Deck PDF via LibreOffice headless (naryad 024). 0 = ok, 1 = export
    failure (no soffice / timeout / empty result), 2 = input unreadable."""
    from razkadr.config import load_config
    from razkadr.render.pdf import PdfExportError, export_pdf, missing_fonts

    deck = Path(args.deck)
    if not deck.is_file():
        print(f"razkadr export-pdf: cannot read {deck}: not a file", file=sys.stderr)
        return EXIT_FILE_ERROR
    config = load_config(args.config)
    timeout_s = int(config.get("pdf_timeout_s", 120))
    out = Path(args.out) if args.out else deck.with_suffix(".pdf")
    try:
        export_pdf(deck, out, timeout_s=timeout_s)
    except PdfExportError as exc:
        print(f"razkadr export-pdf: {exc}", file=sys.stderr)
        return EXIT_INVALID
    for face in sorted(missing_fonts(deck)):
        print(
            f"razkadr export-pdf: font not found in system: {face} "
            "(the PDF will use a substitute; install the font for the intended look)",
            file=sys.stderr,
        )
    print(f"razkadr export-pdf: {out}")
    return EXIT_OK


def _cmd_template_check(args: argparse.Namespace) -> int:
    """User template report (naryad 023). 0 = renderable, 1 = something is
    missing (the report lists what), 2 = the file is not a PPTX."""
    from razkadr.render.template_check import TemplateCheckError, check_template

    path = Path(args.template)
    if not path.is_file():
        print(f"razkadr template-check: cannot read {path}: not a file", file=sys.stderr)
        return EXIT_FILE_ERROR
    try:
        ok, lines = check_template(path)
    except TemplateCheckError as exc:
        print(f"razkadr template-check: {exc}", file=sys.stderr)
        return EXIT_FILE_ERROR
    for line in lines:
        print(line)
    if ok:
        print(f"razkadr template-check: {path} is ready for render")
        return EXIT_OK
    print(f"razkadr template-check: {path} is not ready - see docs/templates.md", file=sys.stderr)
    return EXIT_INVALID


def _cmd_export_json(args: argparse.Namespace) -> int:
    """Storyboard -> canonical JSON with re-validation (naryad 026).
    0 = ok, 1 = invalid, 2 = file error."""
    from razkadr.render.html import export_json
    from razkadr.validation import validate_document

    path = Path(args.storyboard)
    try:
        storyboard = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"razkadr export-json: cannot read {path}: {exc.strerror or exc}", file=sys.stderr)
        return EXIT_FILE_ERROR
    except json.JSONDecodeError as exc:
        print(f"razkadr export-json: invalid JSON in {path}: {exc}", file=sys.stderr)
        return EXIT_FILE_ERROR
    issues = validate_document(storyboard, "storyboard")
    if issues:
        print("razkadr export-json: storyboard is invalid:", file=sys.stderr)
        for issue in issues[:5]:
            print(f"  - {issue.path}: {issue.reason}", file=sys.stderr)
        return EXIT_INVALID
    out = Path(args.out)
    export_json(storyboard, out)
    print(f"razkadr export-json: {out}")
    return EXIT_OK


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

    p_eval = sub.add_parser(
        "eval",
        help="Eval harness: hit@k over the labeled reference set, cost, report (naryad No015)",
    )
    p_eval.add_argument("dir", help="Reference set directory (videos/ + labels/), e.g. data/eval")
    p_eval.add_argument(
        "--config",
        action="append",
        required=True,
        help="Pipeline/search config toml (repeat for comparison: A.toml B.toml)",
    )
    p_eval.add_argument("--k", default="1,3", help="Comma-separated k values (default: 1,3)")
    p_eval.add_argument(
        "--tolerance-s", type=float, default=2.0, help="Hit tolerance in seconds (default: 2)"
    )
    p_eval.add_argument(
        "--preset", default="presets/recipe.yaml", help="Preset YAML (default: presets/recipe.yaml)"
    )
    p_eval.add_argument(
        "--out-dir", default="eval", help="Report output directory (default: eval/)"
    )
    p_eval.add_argument(
        "--keep-runs", action="store_true", help="Keep the per-video run directories (default: removed)"
    )
    p_eval.set_defaults(func=_cmd_eval)

    p_layout = sub.add_parser(
        "layout",
        help="Layout engine: Storyboard -> slides.json with slots and warnings (naryad No021)",
    )
    p_layout.add_argument("storyboard", help="Path to a Storyboard JSON (e.g. examples/storyboard.valid.json)")
    p_layout.add_argument(
        "--lang",
        choices=("ru", "en"),
        default=None,
        help="Slide caption language (default: project.output_lang, else ru); episode text is not translated",
    )
    p_layout.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Output slides path (default: <storyboard dir>/slides.json)",
    )
    p_layout.add_argument("--config", default=None, help="Path to razkadr.toml (default: search)")
    p_layout.set_defaults(func=_cmd_layout)

    p_export_html = sub.add_parser(
        "export-html",
        help="Self-contained HTML deck with timecode links (naryad No026)",
    )
    p_export_html.add_argument("slides", help="Path to slides.json (razkadr layout output)")
    p_export_html.add_argument(
        "--out", default="deck.html", metavar="PATH", help="Output HTML path (default: deck.html)"
    )
    p_export_html.add_argument(
        "--storyboard",
        default=None,
        metavar="PATH",
        help="Storyboard JSON for timecode links (default: storyboard.json next to slides; links degrade to text without it)",
    )
    p_export_html.set_defaults(func=_cmd_export_html)

    p_export_json = sub.add_parser(
        "export-json",
        help="Re-validate a Storyboard and write canonical JSON (naryad No026)",
    )
    p_export_json.add_argument("storyboard", help="Path to a Storyboard JSON")
    p_export_json.add_argument(
        "--out", default="storyboard.out.json", metavar="PATH", help="Output path (default: storyboard.out.json)"
    )
    p_export_json.set_defaults(func=_cmd_export_json)

    p_render = sub.add_parser(
        "render",
        help="slides.json -> deck.pptx on the template (naryad No022)",
    )
    p_render.add_argument("slides", help="Path to slides.json (razkadr layout output)")
    p_render.add_argument(
        "--out", default="deck.pptx", metavar="PATH", help="Output PPTX path (default: deck.pptx)"
    )
    p_render.add_argument(
        "--template",
        default=None,
        metavar="PATH",
        help="PPTX template (default: templates/razkadr-default.pptx)",
    )
    p_render.add_argument(
        "--storyboard",
        default=None,
        metavar="PATH",
        help="Storyboard JSON for speaker notes (default: storyboard.json next to slides; notes degrade to ids without it)",
    )
    p_render.add_argument(
        "--min-font-pt",
        type=int,
        default=12,
        metavar="N",
        help="Auto-fit floor; text that still overflows stays at N pt and warns (default: 12)",
    )
    p_render.set_defaults(func=_cmd_render)

    p_export_pdf = sub.add_parser(
        "export-pdf",
        help="Deck PDF from PPTX via LibreOffice headless (naryad No024)",
    )
    p_export_pdf.add_argument("deck", help="Path to deck.pptx (razkadr render output)")
    p_export_pdf.add_argument(
        "--out", default=None, metavar="PATH", help="Output PDF path (default: <deck>.pdf next to it)"
    )
    p_export_pdf.add_argument(
        "--config", default=None, help="Path to razkadr.toml (pdf_timeout_s, default 120)"
    )
    p_export_pdf.set_defaults(func=_cmd_export_pdf)

    p_check = sub.add_parser(
        "template-check",
        help="Check a user PPTX template against the naming convention (naryad No023)",
    )
    p_check.add_argument("template", help="Path to the user template .pptx")
    p_check.set_defaults(func=_cmd_template_check)

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
