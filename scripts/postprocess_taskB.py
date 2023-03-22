import re

import pandas as pd
import typer
from thefuzz import process

# These are all related to the output files
SYSTEM_OUTPUT = "SystemOutput"

# These are valid section headers for task B
TASK_B_SECTION_HEADER_MAP = {
    "FAMILY HISTORY": "subjective",
    "PHYSICAL EXAMINATION": "objective_exam",
    "ALLERGIES": "subjective",
    "EXAM": "objective_exam",
    "PAST HISTORY": "subjective",
    "PAST MEDICAL HISTORY": "subjective",
    "REVIEW OF SYSTEMS": "subjective",
    "CURRENT MEDICATIONS": "subjective",
    "ASSESSMENT AND PLAN": "assessment_and_plan",
    "PROCEDURE": "subjective",
    "RESULTS": "objective_results",
    "MEDICATIONS": "subjective",
    "INSTRUCTIONS": "assessment_and_plan",
    "IMPRESSION": "assessment_and_plan",
    "SURGICAL HISTORY": "subjective",
    "CHIEF COMPLAINT": "subjective",
    "SOCIAL HISTORY": "subjective",
    "HPI": "subjective",
    "PHYSICAL EXAM": "objective_exam",
    "PLAN": "assessment_and_plan",
    "HISTORY OF PRESENT ILLNESS": "subjective",
    "ASSESSMENT": "assessment_and_plan",
    "MEDICAL HISTORY": "subjective",
    "VITALS": "objective_exam",
    "VITALS REVIEWED": "objective_exam",
}

TASK_B_SECTION_HEADER_ENCODE = {
    "FAMILY HISTORY": "*&0&*",
    "PHYSICAL EXAMINATION": "*&1&*",
    "ALLERGIES": "*&2&*",
    "EXAM": "*&3&*",
    "PAST HISTORY": "*&4&*",
    "PAST MEDICAL HISTORY": "*&5&*",
    "REVIEW OF SYSTEMS": "*&6&*",
    "CURRENT MEDICATIONS": "*&7&*",
    "ASSESSMENT AND PLAN": "*&8&*",
    "PROCEDURE": "*&9&*",
    "RESULTS": "*&10&*",
    "MEDICATIONS": "*&11&*",
    "INSTRUCTIONS": "*&12&*",
    "IMPRESSION": "*&13&*",
    "SURGICAL HISTORY": "*&14&*",
    "CHIEF COMPLAINT": "*&15&*",
    "SOCIAL HISTORY": "*&16&*",
    "HPI": "*&17&*",
    "PHYSICAL EXAM": "*&18&*",
    "PLAN": "*&19&*",
    "HISTORY OF PRESENT ILLNESS": "*&20&*",
    "ASSESSMENT": "*&21&*",
    "MEDICAL HISTORY": "*&22&*",
    "VITALS": "*&23&*",
    "VITALS REVIEWED": "*&24&*",
}

TASK_B_HEADER = [
    "PHYSICAL EXAMINATION",
    "PHYSICAL EXAM",
    "EXAM",
    "ASSESSMENT AND PLAN",
    "VITALS REVIEWED",
    "VITALS",
    "FAMILY HISTORY",
    "ALLERGIES",
    "PAST HISTORY",
    "PAST MEDICAL HISTORY",
    "REVIEW OF SYSTEMS",
    "CURRENT MEDICATIONS",
    "PROCEDURE",
    "RESULTS",
    "MEDICATIONS",
    "INSTRUCTIONS",
    "IMPRESSION",
    "SURGICAL HISTORY",
    "CHIEF COMPLAINT",
    "SOCIAL HISTORY",
    "HPI",
    "PLAN",
    "HISTORY OF PRESENT ILLNESS",
    "ASSESSMENT",
    "MEDICAL HISTORY",
]


def check_complete_word(header, ground_truth):
    for true_header in ground_truth:
        if true_header in header:
            if len(header) > len(true_header):
                if header[len(true_header)].isalpha() == False or header[-len(true_header) - 1].isalpha() == False:
                    return True
            else:
                return True
    return False


def main(submission_fp: str = typer.Argument("Filepath (or URL) to the submission file (should be a CSV file).")) -> None:
    """Postprocesses the submission file for Task B.

    Example usage:
    python scripts/postprocess_taskB.py "./outputs/taskB_wanglab_run1.csv"
    """
    submission_df = pd.read_csv(submission_fp)

    similarities = []
    for i, output in enumerate(submission_df[SYSTEM_OUTPUT]):
        continuous_cap = filter(None, [x.strip() for x in re.findall(r"\b[A-Z\s]+\b", output)])
        continuous_cap_filter = [x for x in continuous_cap if len(x) > 2]
        section_headers = []
        for header in continuous_cap_filter:
            if not check_complete_word(header, TASK_B_SECTION_HEADER_MAP):
                processed_header, similarity = process.extractOne(header, list(TASK_B_SECTION_HEADER_MAP.keys()))
                if similarity > 75 and similarity < 100:
                    processed_output = output.replace(header, processed_header)
                    section_headers.append(processed_header)
                    similarities.append(similarity)
                    print(f"header: {header}, processed_header: {processed_header}, similarity: {similarity}")
                    submission_df.at[i, SYSTEM_OUTPUT] = processed_output

    for i, output in enumerate(submission_df[SYSTEM_OUTPUT]):
        for true_header in TASK_B_HEADER:
            output = output.replace(true_header, TASK_B_SECTION_HEADER_ENCODE[true_header])
        for true_header in TASK_B_HEADER:
            output = output.replace(TASK_B_SECTION_HEADER_ENCODE[true_header], f"\n\n{true_header}\n\n")
        # Replace all instances of 3 or more consecutive newlines with 2 newlines
        submission_df.at[i, SYSTEM_OUTPUT] = re.sub(r"\n{3,}", "\n\n", output.strip())
        submission_df.at[i, SYSTEM_OUTPUT] = re.sub(r'\n:\n', "", submission_df.at[i, SYSTEM_OUTPUT])

    # Save postprocessed submission file to disk
    submission_df.to_csv(submission_fp, index=False)


if __name__ == "__main__":
    typer.run(main)
