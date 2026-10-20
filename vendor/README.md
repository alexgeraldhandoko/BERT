# External scoring code

`evaluate-v2.0.py` is the unmodified scorer supplied in the CS4248 Canvas Project folder. It is external code written by the SQuAD authors, not a group implementation. The course specifies this SQuAD 2.0 scorer even though the supplied train and dev datasets are SQuAD 1.1. With answerable-only SQuAD 1.1 input and no no-answer probability file, it reports `exact` and `f1` for those questions. A fixture in `tests/test_baseline.py` checks that combination.

Course link: https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/

Author-maintained source: https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v2.0.py

The copied course file has SHA-256 `710840ce2c2c61716334c056777032f24660b83eb9869a451036e6a24d8103c4`. Its author's repository uses the MIT license; the license notice is in this directory.

