# Deploying ContractLens on Streamlit Community Cloud (free)

Do these in order. About 20 minutes the first time.

## 1. Get a free Gemini key
https://aistudio.google.com -> "Get API key". No card needed. Keep it private.

## 2. Run the agent locally ONCE and save real results
    pip install -r requirements.txt
    export GEMINI_API_KEY=your_key            # Windows PowerShell: $env:GEMINI_API_KEY="your_key"
    python run_agent.py contract_18           # test one contract first
    python run_agent.py --all --sleep 5       # all 30; results go to data/cache/agent/
    python evaluate.py --method both          # your REAL agent-vs-rules numbers

If the model name is rejected, set `GEMINI_MODEL` to one your key lists in AI Studio.
If you see many 429 errors, raise `--sleep`. Re-run later: finished contracts are skipped unless `--force`.

Why: the live demo then works even if the free quota or wifi fails. The app's "Replay saved run" mode
shows these real results instantly, and the Accuracy tab shows the real agent score.

## 3. Push to GitHub
Create a new repo (public or private) and upload the project folder, **including `data/cache/agent/`**.
Never commit `.streamlit/secrets.toml` (already in `.gitignore`).

## 4. Deploy
1. https://share.streamlit.io -> sign in with GitHub -> **Create app**.
2. Repository: yours. Branch: `main`. Main file path: `app.py`.
3. **Advanced settings -> Secrets**, paste:

        GEMINI_API_KEY = "your_key"
        APP_PASSWORD = "choose-something"      # optional but recommended

4. Deploy. First build takes a few minutes.

## 5. Before the class
- Open the link on the classroom network/phone hotspot once. Free apps sleep after inactivity; wake it up.
- Default demo path (safe): **Replay saved run** on 2-3 contracts, then Accuracy, Review queue, Impact.
- Live path: choose "Live agent (Gemini)", enter APP_PASSWORD, run contract_18 (has an amendment).
  Keep Replay as the fallback if the quota (free tier is rate-limited) or network fails.

## Cautions
- A public link with a live key lets strangers spend your free quota. Set `APP_PASSWORD`.
- Streamlit Cloud's filesystem is temporary: runs made live online are not kept. Save real runs locally and commit them.
- "Scripted simulation" mode is NOT an AI. It tests the interface and is labelled as such. Never quote its numbers.
- Never paste the key into code, chat or the repo.
