# -*- coding: utf-8 -*-
"""Run AFTER the expert fills the validation files. Computes agreement (Cohen kappa + % accuracy)
between the expert ratings and the system's automatic judgments."""
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from pathlib import Path
VAL = Path("saudi_labor_law_voice_agent_project/14_human_validation")

def yn(s):
    return str(s).strip().upper().startswith("Y")

# Answers: expert vs system
a = pd.read_excel(VAL/"expert_validation_ANSWERS.xlsx")
a = a[a["expert_answer_correct (Y/N)"].astype(str).str.strip()!=""]
if len(a):
    exp = a["expert_answer_correct (Y/N)"].map(yn).astype(int)
    sysj = a["system_judged_correct"].fillna(0).astype(int)
    print(f"Answers rated by expert: {len(a)}")
    print(f"  Expert accuracy of system answers: {exp.mean()*100:.0f}%")
    print(f"  Agreement system-vs-expert (Cohen kappa): {cohen_kappa_score(sysj, exp):.3f}")
    print(f"  Raw agreement: {(sysj==exp).mean()*100:.0f}%")
else:
    print("Fill expert_validation_ANSWERS.xlsx first.")

# Questions: how many the expert accepted
q = pd.read_excel(VAL/"expert_validation_QUESTIONS.xlsx")
q = q[q["expert_question_valid (Y/N)"].astype(str).str.strip()!=""]
if len(q):
    print(f"\nQuestions rated: {len(q)} | expert-accepted valid: {q['expert_question_valid (Y/N)'].map(yn).mean()*100:.0f}%")
