import typer
from description_summary.fine_tuner.data import process_dataset_summaries
from description_summary.fine_tuner.train import train
from labeling.argilla_description_summarization import upload_records, download_records
from description_summary.fine_tuner.predict import run_inference_on_json, run_evaluate_on_csv

app = typer.Typer()

app.command("load_training_data")(process_dataset_summaries)
app.command("train")(train)
app.command("argilla_upload")(upload_records)
app.command("argilla_download")(download_records)
app.command("inference")(run_inference_on_json)
app.command("evaluate")(run_evaluate_on_csv)

if __name__ == "__main__":
    app()
