import os
import webbrowser


class ReportViewerHelper:

    def open_file(self, filepath):

        if not os.path.exists(filepath):
            return False

        return webbrowser.open(f"file://{os.path.abspath(filepath)}")

    def open_dashboard(self):

        return self.open_file("outputs/dashboard.html")

    def open_advanced_report(self):

        return self.open_file("outputs/advanced_report.html")
