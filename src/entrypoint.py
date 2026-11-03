"""Parse once, validate configuration, then compose one explicit operation."""

import json
from dataclasses import asdict

from compaction.compaction import Compactor
from data.data import build_sft_dataset
from hparam.hparam import application_parser, load_config
from training.training import llamafactory_arguments, train_sft


def main(argv=None):
    parser = application_parser()
    args = parser.parse_args(argv)
    parameters = load_config(args.config, args.operation)
    if args.print_config:
        effective = asdict(parameters)
        if args.operation == "train-sft":
            effective = llamafactory_arguments(parameters)
        print(json.dumps(effective, indent=2))
        return
    if args.operation == "compact":
        if args.input is None or args.output is None:
            parser.error("compact requires --input and --output")
        if args.output.exists():
            raise FileExistsError("Choose a new result path: " + str(args.output))
        context = args.input.read_text(encoding="utf8")
        result = Compactor(parameters).compact(context)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf8") as stream:
            json.dump(
                {
                    "text": result.text,
                    "metadata": result.metadata,
                    "configuration": asdict(parameters),
                },
                stream,
                indent=2,
                ensure_ascii=False,
            )
            stream.write("\n")
    elif args.operation == "prepare-sft":
        build_sft_dataset(parameters)
    elif args.operation == "train-sft":
        train_sft(parameters)
    else:
        raise ValueError("Unsupported operation: " + args.operation)


if __name__ == "__main__":
    main()
