# __all__ = ["ArgillaLabelParams", "parse_command_line_argilla"]
# from pydantic import BaseModel
# from typing import Optional
# import argparse
#
#
# class ArgillaLabelParams(BaseModel):
#     dataset_name: str
#     suggestion: Optional[bool] = False
#     upload_file: Optional[str] = None
#     download_file_json: Optional[str] = None
#     download_file_dataset: Optional[bool] = None
#     num_of_records: Optional[int] = None
#
#
# def parse_command_line_argilla() -> ArgillaLabelParams:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", "-DS", type=str, default="funding_event_analysis")
#     parser.add_argument("--upload", "-U", type=str)
#     parser.add_argument("--download_json", "-J", type=str, default=None)
#     parser.add_argument("--download_ds", "-D", type=bool, default=False)
#     parser.add_argument("--suggest", "-S", type=bool, default=False)
#     parser.add_argument("--records", "-R", type=int, default=None)
#
#     args = parser.parse_args()
#     return ArgillaLabelParams(
#         dataset_name=args.dataset,
#         suggestion=args.suggest,
#         upload_file=args.upload,
#         download_file_json=args.download_json,
#         download_file_dataset=args.download_ds,
#         num_of_records=args.records,
#     )
