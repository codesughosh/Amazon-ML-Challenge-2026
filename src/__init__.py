"""Amazon ML Challenge 2026 - shared source package.

Usage from the project root (or a notebook in notebooks/ after a chdir):

    from src.utils import seed_everything, timer, describe_df, load_cached
    from src.metrics import smape, f1_macro, score
    from src.cv import make_folds, run_cv, Tracker
    from src.submit import write_submission, blend
"""
