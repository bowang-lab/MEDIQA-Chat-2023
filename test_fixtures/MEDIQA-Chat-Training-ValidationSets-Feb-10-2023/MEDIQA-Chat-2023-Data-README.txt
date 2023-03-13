===========================================
MEDIQA-Chat 2023 Training/Validation Data
===========================================

Task A:
=======

The training set consists of 1,201 pairs of conversations and associated section headers and contents. 
The validation set consists of 100 pairs of conversations and their summaries. 


The full list of normalized section headers: 

1. fam/sochx [FAMILY HISTORY/SOCIAL HISTORY]
2. genhx [HISTORY of PRESENT ILLNESS]
3. pastmedicalhx [PAST MEDICAL HISTORY]
4. cc [CHIEF COMPLAINT]
5. pastsurgical [PAST SURGICAL HISTORY]
6. allergy
7. ros [REVIEW OF SYSTEMS]
8. medications
9. assessment
10. exam
11. diagnosis
12. disposition
13. plan
14. edcourse [EMERGENCY DEPARTMENT COURSE]
15. immunizations
16. imaging
17. gynhx [GYNECOLOGIC HISTORY]
18. procedures
19. other_history
20. labs


Task B:
=======
The training set consists of 67 pairs of conversations and full notes. The validation set includes 20 pairs of conversations and clinical notes. 

Full encounter notes are expected to have at least one of four overall section divisions demarked by the first-occuring of its related section headers :

| note_division | section_headers
---------------------------------------------------------------------------
| subjective | chief complaint, history of present illness, hpi, subjective
---------------------------------------------------------------------------
| objective_exam | physical exam, exam
---------------------------------------------------------------------------
| objective_results | results, findings
---------------------------------------------------------------------------
| assessment_and_plan | assessment, plan

Depending on the encounter, objective_exam and objective_results may not be relevant.
We encourage review the sample data as well as the evaluation script to understand the best demarkation headers for your generated note.


Task C:
=======
The training set consists of 67 pairs of full doctor-patient conversations and notes and the validation set includes 20 pairs of full conversations and clinical notes (same as Task-B datasets). The Task-A training and validation sets (1,301 pairs) could be used as additional training data. 

