from .base import SubcommandPlugin
import argparse
from cvs.input.generate.base import _discover_generators, _run_generator


class GeneratePlugin(SubcommandPlugin):
    def get_name(self):
        return "generate"

    def get_parser(self, subparsers):
        parser = subparsers.add_parser("generate", help="Generate configuration files or templates")
        parser.add_argument("generator", nargs="?", help="Name of the generator to use")
        parser.add_argument("generator_args", nargs=argparse.REMAINDER, help="Arguments for the generator")
        parser.set_defaults(_plugin=self)
        return parser

    def get_epilog(self):
        return """
Generate Commands:
  cvs generate                       List available generators
  cvs generate cluster_json --help   Show help for cluster_json generator"""

    def run(self, args):
        generators = _discover_generators()
        if args.generator is None:
            if generators:
                print("Available generators:")
                for name, plugin in sorted(generators.items()):
                    print(f"  {name} - {plugin.get_description()}")
            else:
                print("No generators found in cvs/generate/ directory.")
        else:
            if getattr(args, "extra_pytest_args", None):
                args.generator_args.extend(args.extra_pytest_args)
                args.extra_pytest_args = []
            if args.generator_args and args.generator_args[0] in ["-h", "--help"]:
                _run_generator(args.generator, ["-h"])
            else:
                _run_generator(args.generator, args.generator_args)
