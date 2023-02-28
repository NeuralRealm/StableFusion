import argparse

from .. import __version__
from .run_api import RunStableFusionAPICommand
from .run_app import RunStableFusionAppCommand


def main():
    parser = argparse.ArgumentParser(
        "StableFusion CLI",
        usage="stablefusion <command> [<args>]",
        epilog="For more information about a command, run: `stablefusion <command> --help`",
    )
    parser.add_argument("--version", "-v", help="Display stablefusion version", action="store_true")
    commands_parser = parser.add_subparsers(help="commands")

    # Register commands
    RunStableFusionAppCommand.register_subcommand(commands_parser)
    #RunStableFusionAPICommand.register_subcommand(commands_parser)

    args = parser.parse_args()

    if args.version:
        print(__version__)
        exit(0)

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    command = args.func(args)
    command.run()


if __name__ == "__main__":
    main()
