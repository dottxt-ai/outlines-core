import sys

# There is a conflict between asv.statistics and the standard library's statistics module.
# This is a workaround to use the standard library's median function.
if "asv.statistics" in sys.modules:

    def median(data):
        import statistics

        return statistics.median(data)

    asv_statistics = sys.modules["asv.statistics"]
    asv_statistics.median = median  # type: ignore
