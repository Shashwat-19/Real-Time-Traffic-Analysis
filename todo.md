# To-do

Work left after the code review fixes were pushed to `origin/main` (commit `949ae67`).

## Must do

- [ ] **Rotate the TomTom API key.** Create a new key in the TomTom account and revoke the old one. The old key is still in the history of commits up to `da67f51`.
- [ ] **Create a local `.env` file.** Copy `.env.example` to `.env` and fill in the new key. Git ignores `.env`, so do not commit it.
- [ ] **Retrain the model.** Models without the new `.sha256` and `.meta.json` files are refused. Run `python -m src.train`, then `python -m src.evaluate`.

## Should do

- [ ] **Check the dashboard starts.** Run `python -m streamlit run src/app.py`. The plain `streamlit run src/app.py` fails with `No module named 'src'`.
- [ ] **Build and run the Docker setup.** Install Docker, then run `docker compose up --build`. The image has not been built or tested yet.
- [ ] **Check the scheduler can write `./data`.** The container runs as UID 1000. On Linux, run `sudo chown -R 1000:1000 data` if writes fail.
- [ ] **Confirm the GitHub Actions run passes.** The CI workflow has only been checked locally.

## Decide

- [ ] **Scrub the old key from git history** (optional). This rewrites every commit hash after the change and needs a force-push. Only do it if the repo is public or shared.
- [ ] **Fix the notebook label problem.** `BangaloreTraffic.ipynb` labels congestion from the same speeds it trains on, so its accuracy is misleading. A sound label needs future readings.
- [ ] **Decide on `road_closure`.** It may partly encode the congestion label, since closed roads probably report speed 0. Verify against the TomTom docs.
- [ ] **Decide on the duplicate rule.** It drops repeated identical readings, so jams with unchanged speeds are stored once.
- [ ] **Decide on the model cache.** The API keeps the model until restart. Decide whether it should reload on file change, as the dashboard does.

## Later

- [ ] Read the whole history in `src/data_storage.py` only for the tail the API needs (the file is still read in full).
- [ ] Add a lockfile (optional; not requested in the review).
- [ ] Verify the real TomTom payload shape; tests use fixtures only.
