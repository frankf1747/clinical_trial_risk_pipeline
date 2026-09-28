from ctrisk.spark.clean_trials import build_sponsor_outcomes


def test_every_finished_interventional_study_with_a_lead_sponsor(aact):
    rows = build_sponsor_outcomes(aact["studies"], aact["sponsors"]).collect()
    got = {r.nct_id: (r.sponsor_name, r.terminated, str(r.start_date), str(r.completion_date)) for r in rows}
    # Any phase counts as sponsor history; studies without a lead sponsor, observational,
    # withdrawn, and unfinished studies are excluded.
    assert got == {
        "NCT001": ("Merck", 0, "2012-03-01", "2014-01-01"),
        "NCT002": ("State University", 1, "2015-06-15", "2016-01-01"),
        "NCT009": ("Pfizer", 1, "2018-05-01", "2019-01-01"),
    }
