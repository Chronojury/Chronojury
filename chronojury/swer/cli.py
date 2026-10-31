"""Run SWER on prepared case directories using an OpenAI-compatible endpoint."""
import argparse
import os
from pathlib import Path
from .config import REASONING_CONFIG


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing <case_id>/data.json")
    parser.add_argument("--output-dir", type=Path, default=Path(REASONING_CONFIG["output_dir"]))
    parser.add_argument("--work-dir", type=Path, default=Path.cwd(),
                        help="Base for relative media paths and generated segment/network/frame files")
    parser.add_argument('--network-dir', type=Path,
                        help='Following-network directory; defaults to a separate directory per juror count')
    parser.add_argument("--start", type=int, default=1, help="1-based position in numeric case-ID order")
    parser.add_argument("--end", type=int, help="Inclusive position in numeric case-ID order")
    parser.add_argument("--model", default=os.environ.get("CHRONOJURY_MODEL", REASONING_CONFIG["model"]),
                        help="Model served by the configured endpoint")
    parser.add_argument("--api-base", help="Single /v1 endpoint; alternatively use OPENAI_BASE_URL")
    parser.add_argument("--api-bases", help="Comma-separated equivalent /v1 replicas; overrides --api-base")
    parser.add_argument("--temperature", type=float, default=REASONING_CONFIG["temperature"],
                        help="Verdict sampling temperature")
    parser.add_argument("--max-completion-tokens", type=int, default=REASONING_CONFIG["max_completion_tokens"],
                        help="Token budget for segmentation and verdicts")
    parser.add_argument("--max-workers", type=int, default=REASONING_CONFIG["max_workers"],
                        help="Concurrent cases")
    parser.add_argument("--prefetch-cases", type=int, default=REASONING_CONFIG["prefetch_cases"],
                        help="Additional case JSONs to prefetch on CPU; 0 disables prefetch")
    parser.add_argument("--sleep", type=float, default=REASONING_CONFIG["sleep"],
                        help="Delay before segmentation and text-verdict requests, in seconds")
    parser.add_argument("--no-skip", action="store_true")
    parser.add_argument("--num-agents", type=int,
                        default=int(os.environ.get("CHRONOJURY_NUM_AGENTS", "6")),
                        help="Number of juror agents")
    args = parser.parse_args()
    if args.start < 1 or (args.end is not None and args.end < args.start) or args.max_workers < 1:
        parser.error("Invalid case range or worker count")
    if args.prefetch_cases < 0:
        parser.error("--prefetch-cases must be nonnegative")
    if args.num_agents < 2:
        parser.error("--num-agents must be at least 2")
    if not 0 <= args.temperature <= 2 or args.max_completion_tokens < 1:
        parser.error("Invalid sampling parameters")

    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    if args.api_base:
        os.environ["OPENAI_BASE_URL"] = args.api_base
        os.environ.pop("CHRONOJURY_API_BASES", None)
    if args.api_bases:
        os.environ["CHRONOJURY_API_BASES"] = args.api_bases
    from . import config
    config.NUM_AGENTS = args.num_agents
    config.DEBATE_AGENTS[:] = [f"Agent_{i}" for i in range(args.num_agents)]
    config.FOLLOWING_NETWORKS_DIR = args.network_dir or (
        'following_networks' if args.num_agents == 6 else f'following_networks_{args.num_agents}'
    )
    os.environ["CHRONOJURY_NUM_AGENTS"] = str(args.num_agents)
    os.chdir(args.work_dir)

    from .runner import BatchDebateGenerator
    generator = BatchDebateGenerator(
        output_dir=str(output_dir), model_name=args.model, T=args.temperature,
        sleep=args.sleep, max_workers=args.max_workers, skip_existing=not args.no_skip,
        max_completion_tokens=args.max_completion_tokens,
        prefetch_cases=args.prefetch_cases,
    )
    report = generator.process_directory(input_dir, args.start, args.end)
    if report.get("summary", {}).get("failed_cases", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
