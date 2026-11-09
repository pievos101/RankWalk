# Experiment script snapshot

These are the actual scripts used in the cloud evaluation. They preserve original cloud paths and historical queue PIDs for provenance; do not execute `finish_queue.sh`, `clinical_chain.sh`, `diagnostic_chain.sh`, or `live_report.py` unchanged on another machine.

Use `../prepare_local_run.py` to create relocated copies of the scientific scripts in an ignored work directory. The LaTeX report itself already uses relative paths and compiles independently. The training environments and external method dependencies must be installed separately; versions and commits are recorded under `../report/provenance/`.
