# BCI Lab in the browser

Goal: an interactive page that shows real motor-imagery EEG, runs NeuroHand's
EEGNet in the browser and turns its decision into a prosthetic-hand command.

## Plan

- [x] Get BCI Competition IV 2a without MOABB (the BNCI server is blocked in the
      cloud sandbox): use the `.npz` port from github.com/bregydoc/bcidatasetIV2a
- [x] `src/visualization/bci_lab_export.py`: preprocess (4–38 Hz, 0–4 s, z-score),
      train `src/models/eegnet.py` leave-one-subject-out ("new user"), calibrate on
      each subject's own trials (80/20 split per subject), compute mu/beta ERD,
      export `bci-lab/lab-data.js`
- [x] `bci-lab/eegnet.js`: EEGNet forward pass in plain JS (no dependencies)
- [x] `bci-lab/index.html`: EEG trace player, ERD topomap, electrode lesion and
      noise experiments, class probabilities, hand with servo angles,
      "inside EEGNet" (temporal filter spectra, spatial filters, evidence over
      time), test results, grand-average ERD atlas, method and limitations
- [x] Tests: data loading, ERD math, int16 round trip, JS vs PyTorch parity
- [x] Link the lab from the GitHub Pages portal and the README

## Review

- Held-out test trials (20 % of each subject, 463 in total), 4 classes, chance 25 %:
  - new user (EEGNet trained on the other 8 subjects): **46.9 %**, κ = 0.29
  - after calibration on ~207 of the subject's own trials: **61.3 %**, κ = 0.48
  - for reference, one model trained on everybody at once: 59.4 %
- First calibration attempt did nothing (59.4 % → 59.4 %): the pooled base model had
  already seen the subject's calibration trials, and early stopping on ~30 validation
  trials always picked epoch 0–1. Fixed by leave-one-subject-out base models and
  choosing the number of calibration epochs by 3-fold CV inside the calibration trials.
- Grand-average mu ERD (0.5–2.5 s) is contralateral as expected: right hand C3 −18 %
  vs C4 −12 %, left hand C4 −13 % vs C3 −8 %; tongue imagery gives mu ERS over the hand
  areas (+12…+19 %), feet the weakest ERD (CPz −8 %) and the lowest accuracy (53 %).
- JS forward pass matches PyTorch to ~1e-6 in probability (`tests/test_bci_lab_export.py`
  and the parity number shown on the page).
- Test accuracy is same-session; cross-session accuracy will be lower, which is why the
  roadmap keeps online calibration.
