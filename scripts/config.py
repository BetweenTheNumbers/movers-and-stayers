"""
Shared cohort configuration.

The draft-year window is defined HERE and nowhere else. Every step script
imports these values, so changing the window is a one-line edit.

Why 1996-2019:
  - 1996 was the first draft in which the Arizona Diamondbacks and Tampa Bay
    Devil Rays participated (they began MLB play in 1998). This is the start
    of the modern 30-team-era draft pool.
  - 2019 is the last draft class with enough development runway for a fair
    "did he reach MLB" comparison.

IMPORTANT: this window filters DRAFT YEARS only. Outcomes are not year-capped.
`reached_mlb` (highLevel == 'MLB') and the FanGraphs career WAR join both use
each player's FULL major-league history regardless of when he was drafted.

Scripts run as `python scripts/<name>.py` from the project root, which puts
scripts/ on sys.path, so `from config import ...` resolves to this file.
"""

START_YEAR = 1996
END_YEAR = 2019

# Human-readable label used in console headers and figure captions.
COHORT_LABEL = f"{START_YEAR}-{END_YEAR}"


def cohort_filter(df, year_col='year'):
    """Return df restricted to the configured draft-year window."""
    return df[(df[year_col] >= START_YEAR) & (df[year_col] <= END_YEAR)].copy()
