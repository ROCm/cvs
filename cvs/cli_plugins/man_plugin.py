import sys

from .list_plugin import ListPlugin
from cvs.lib.man_lib import find_parameters, iter_parameters, render_json, render_text
from cvs.parsers.config_registry import TEST_CONFIG_DOCS, documented_tests, get_config_doc, resolve_sample_path


class ManPlugin(ListPlugin):
    def get_name(self):
        return "man"

    def get_parser(self, subparsers):
        parser = subparsers.add_parser("man", help="Explain the config parameters for a test")
        parser.add_argument("test", nargs="?", help="Test to explain. Omit to list tests that have a man page.")
        parser.add_argument("parameter", nargs="?", help="Optional: show only parameters matching this name")
        parser.add_argument("--json", action="store_true", dest="as_json", help="Emit the reference as JSON")
        parser.set_defaults(_plugin=self)
        return parser

    def get_epilog(self):
        return """
Man Commands:
  cvs man                              List tests that have a config parameter reference
  cvs man rccl_perf                    Explain every config parameter for rccl_perf
  cvs man rccl_perf nic_model          Explain a single parameter
  cvs man rccl_perf --json             Emit the reference as JSON"""

    @staticmethod
    def _parameters_for(doc):
        """Flatten every documented section of a test into one parameter list."""
        params = []
        for section in doc.sections:
            params.extend(iter_parameters(section.model, prefix=section.key))
        return params

    def _list_documented(self):
        print("\nConfig parameter references")
        print("=" * 80)
        for test_name in documented_tests():
            print(f"\n  • {test_name}")
            print(f"      {TEST_CONFIG_DOCS[test_name].summary}")

        print(f"\n{'=' * 80}")
        print(f"Total: {len(TEST_CONFIG_DOCS)} tests with a config parameter reference")

        undocumented = set()
        for tests in self.test_map.values():
            undocumented.update(set(tests) - set(TEST_CONFIG_DOCS))
        if undocumented:
            print(f"{len(undocumented)} other test suites have no reference yet.")
        print("\nUse 'cvs man <test>' to explain a test's config parameters.\n")

    def run(self, args):
        # main() parses with parse_known_args, so an unrecognised flag lands here
        # silently instead of erroring. Reject it rather than ignore it.
        unknown = getattr(args, "extra_pytest_args", None)
        if unknown:
            self._emit_error(f"Error: unrecognized arguments: {' '.join(unknown)}", args.as_json)
            self._emit_error("Use 'cvs man --help' to see the available options.", args.as_json)
            sys.exit(1)

        if not args.test:
            self._list_documented()
            return

        doc = get_config_doc(args.test)
        if not doc:
            self._emit_error(f"Error: no config parameter reference for '{args.test}'", args.as_json)
            if self._find_test(args.test) is not None:
                msg = "This test exists but is not documented yet. Use 'cvs man' to see what is."
                self._emit_error(msg, args.as_json)
            else:
                self._emit_error("Use 'cvs list' to see available tests.", args.as_json)
            sys.exit(1)

        params = self._parameters_for(doc)
        title = f"cvs man {args.test}"

        if args.parameter:
            matches = find_parameters(params, args.parameter)
            if not matches:
                self._emit_error(f"Error: no parameter matching '{args.parameter}' in {args.test}", args.as_json)
                self._emit_error(f"Use 'cvs man {args.test}' to see every parameter.", args.as_json)
                sys.exit(1)
            params = matches
            title = f"{title} {args.parameter}"

        sample_paths = [resolve_sample_path(sample) for sample in doc.samples]

        if args.as_json:
            print(render_json(params, test=args.test, config_files=sample_paths))
            return

        print(render_text(params, title=title, summary=doc.summary))
        print(f"Sample config: {', '.join(sample_paths)}\n")
