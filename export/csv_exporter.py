import csv


class CsvExporter:

    def export_value_bets(self, filepath, value_bets):

        fieldnames = ["match_id", "market", "probability", "odds"]

        with open(filepath, "w", newline="", encoding="utf-8") as f:

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for vb in value_bets:
                writer.writerow(
                    {
                        "match_id": vb.get("match_id"),
                        "market": vb.get("market"),
                        "probability": vb.get("probability"),
                        "odds": vb.get("odds"),
                    }
                )
