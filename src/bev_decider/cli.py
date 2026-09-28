import argparse
import json
import sys

from .decider import DEFAULT_MODEL, load


def main():
    parser = argparse.ArgumentParser(prog="bev-decider")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("serve", "decide"):
        p = sub.add_parser(name)
        p.add_argument("--model", default=DEFAULT_MODEL, help="Hub repo id or local folder")
        p.add_argument("--revision", default=None)
        p.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
        p.add_argument("--max_state_tokens", type=int, default=None)
        p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16",
                       help="bf16 (default) is faster on GPU / Apple Silicon; fp32 is exact (perfect option-order invariance)")
    sub.choices["serve"].add_argument("--host", default="127.0.0.1")
    sub.choices["serve"].add_argument("--port", type=int, default=8008)
    sub.choices["decide"].add_argument("request", nargs="?", default="-",
                                       help='JSON file with {"state": ..., "questions": {...}} (default: stdin)')
    args = parser.parse_args()

    decider = load(args.model, args.revision, args.device, args.max_state_tokens, args.precision)
    if args.command == "decide":
        request = json.load(sys.stdin if args.request == "-" else open(args.request))
        print(json.dumps({"answers": decider.decide(request["state"], request["questions"])}, indent=2))
    else:
        try:
            import uvicorn
            from .serve import create_app
        except ImportError:
            sys.exit('the server needs the serve extra: pip install "bev-decider[serve]"')
        uvicorn.run(create_app(decider), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
