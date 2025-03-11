import logging

import evaluate
import pandas as pd


class SummaryEvaluator:
    def __init__(self, test_data_path: str = None):
        self._test_data_path = test_data_path
        logging.info("Loading ROUGE")
        self._rouge = evaluate.load('rouge')
        logging.info("Loading BLEU")
        self._bleu = evaluate.load('bleu')
        logging.info("Loading METEOR")
        self._meteor = evaluate.load('meteor')

        self._predicted_data = []
        self._references = []

        # self.get_data_from_csv()

    @staticmethod
    def read_csv(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, header=0, index_col=0)
        return df.sort_values(by='id')

    def get_data_from_csv(self) -> None:
        df_references = self.read_csv("data/descriptions_processed_gpt-4o_2024-10-14.csv")
        df_predictions = self.read_csv(self._test_data_path)
        for (id_1, row1), (id_2, row2) in zip(df_references.iterrows(), df_predictions.iterrows()):
            if int(id_1) == int(id_2):
                self._references.append(row1["description_openai"])
                self._predicted_data.append(row2["description_openai"])

    def evaluate_model_generated_summary(self, predictions, references):
        rouge_results = self._rouge.compute(predictions=predictions, references=references)
        bleu_results = self._bleu.compute(predictions=predictions, references=references)
        meteor_results = self._meteor.compute(predictions=predictions, references=references)
        print(f"ROUGE: {rouge_results}")
        print(f"BLEU: {bleu_results}")
        print(f"METEOR: {meteor_results}")
        return rouge_results, bleu_results, meteor_results


if __name__ == "__main__":
    data = ["data/descriptions_processed_llama3.2:1b_2024-10-17.csv",
            "data/descriptions_processed_llama3.2:3b_2024-10-16.csv",
            "data/descriptions_processed_phi3.5_2024-10-16.csv"]
    for file in data:
        summary = SummaryEvaluator(file)
        summary.evaluate_model_generated_summary()
