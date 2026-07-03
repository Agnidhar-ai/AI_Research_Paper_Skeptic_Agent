from pathlib import Path

from agents.skeptic_agent import SkepticAgent
from config import settings
from tools.document_loader import SUPPORTED_DOCUMENT_EXTENSIONS, load_document_text
from tools.report_generator import save_report


def main() -> None:
    papers_dir = Path(settings.data_dir) / "papers"
    reports_dir = Path(settings.outputs_dir) / "reports"
    paper_files = sorted(
        path
        for path in papers_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
    )

    if not paper_files:
        print(f"No PDF papers found in {papers_dir}")
        return

    agent = SkepticAgent()
    for paper_path in paper_files:
        text = load_document_text(paper_path)
        report = agent.review(text, source_name=paper_path.name)
        output_path = save_report(report, reports_dir, paper_path.stem)
        print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
