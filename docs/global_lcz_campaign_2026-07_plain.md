# Teaching a Computer to Recognize Neighborhood Types from Space — a Plain-Language Report

*(June–July 2026. The technical version of this report, with all the numbers and file paths,
is `global_lcz_campaign_2026-07.md`.)*

## What is this project about?

Cities are made of different kinds of "neighborhood textures": dense high-rise blocks,
leafy suburbs, industrial estates, open fields, water, and so on. Scientists call these
**Local Climate Zones (LCZs)** — 17 standard categories that matter for things like urban
heat and climate planning.

We want a computer to look at satellite data and say, for every ~300 m square of land,
which of the 17 categories it belongs to.

Instead of raw satellite photos, we use **embeddings**: big AI "foundation models" (from
Google and others) have already digested years of satellite imagery and summarize every
spot on Earth as a list of numbers — a kind of fingerprint of that place. We train our own,
much smaller model to read those fingerprints and name the neighborhood type. We use three
different fingerprint providers: **Tessera**, **AlphaEarth**, and **Seamless**.

## How do we measure success?

The hard, honest test: the model trains on a set of cities, and is then examined on **10
cities it has never seen** — Munich, Jakarta, Nairobi, Santiago, Tehran, and others. That
simulates the real use case: applying the map to a city nobody labeled by hand.

The score is **kappa**: agreement with the human labels *after* subtracting lucky guesses.
0 = no better than chance, 1 = perfect. On this task, moving up by even 0.01 (one "point")
is meaningful.

**Where we started: 0.619. Where we ended: 0.687 honestly, 0.649 for a single model.**
Here is the story of how — including the three ideas that failed, which taught us the most.

---

## Experiment 1: Does a bigger brain help? (No.)

We first tuned the training recipe carefully (learning speed, warm-up, regularization) —
that gave us the 0.619 starting point. Then we tried much larger neural networks, up to
~5× the size. **Result: no improvement at all.** The model wasn't too small; the problem
lies elsewhere. This told us to stop spending effort on the model and start looking at the
data.

## Experiment 2: Squeezing the labels harder (mostly failed) — and the first win

Some neighborhood types are rare in the training data (e.g., lightweight low-rise
buildings), so we tried several standard tricks to make the model pay more attention to
rare categories: showing rare examples more often, adjusting the scoring, and feeding the
model all three fingerprint types at once. **Every one of these made things worse.** Our
recipe already handled rarity as well as it could — piling on a second fix over-corrects.

The one thing that *did* work: train three separate models — one per fingerprint provider —
and let them **vote**. Different providers make different mistakes, so the vote cancels
errors out. That alone jumped the score from 0.619 to **0.650**. Committee beats any single
expert. This became the backbone of everything after.

## Experiment 3: Learning from unlabeled land (the "noisy student") — our biggest single win

Human-labeled examples are limited. But unlabeled land is unlimited, and there already
exists a rough, free, computer-made LCZ map of the whole world (by Demuzere and colleagues).
It's not accurate enough to trust blindly — but it's a useful second opinion.

So we did the following, called **noisy-student training**:

1. Pick ~286,000 new patches of land around the world that no human has labeled.
2. Ask two "opinions" about each patch: our trained model (the **teacher**) and the rough
   world map. **Keep a patch only when both agree** and the teacher is confident. Rare
   categories get a slightly gentler rule so they aren't filtered out entirely.
3. Train a fresh model (the **student**) on the human labels *plus* these ~67,000 new
   "pretty sure" examples — counted at half weight, since they might be wrong.

The student beat its teacher: **0.619 → 0.642**. We then repeated the loop with the student
as the new teacher: **0.650** on the second round. A third round wasn't worth it — each
round helps less, because the student increasingly just agrees with the committee it came
from.

**A failure worth remembering**: in one round we raised the bar to "teacher must be ≥80%
confident" — and the score *dropped*. Why? Our training style deliberately keeps the model
humble: it almost never says more than ~90%, even when it's right. So "80% confident" was a
much stricter filter than it sounds, and almost nothing passed. Lesson: **a confidence
threshold only means something relative to how confident that particular model is capable
of being.**

We also ran the same trick on the AlphaEarth model (the weakest of the three). It improved
a little (0.513 → 0.522), but the improvement didn't carry into the committee vote — teaching
a model on its own filtered opinions mostly makes it more like itself.

## Experiment 4: Smarter voting

A plain vote treats all three models equally, but they aren't equal. Two refinements:

- **Weighted voting**: search for the best mix on a practice set. Best mix: 20% Tessera,
  70% AlphaEarth, 10% Seamless → **0.692**. (Surprising 70% for the weakest model — read on.)
- **Confidence calibration**: it turns out all three models are *under-confident* —
  they whisper when they should speak. The AlphaEarth model whispers the most. If you
  mathematically "turn up its volume" first (temperature calibration), the best mix becomes
  a sensible, balanced 35/40/25 and lands at nearly the same score. In other words, the
  strange 70% weight wasn't saying "this model is best" — it was compensating for its
  whispering.

## Experiment 5: The exam-leak audit — catching ourselves cheating

Here we nearly fooled ourselves. A fancier combiner (a small learned model that merges the
three opinions) scored a spectacular **0.776**. Too good to be true — and it was.

The flaw: our "practice set" (validation) and "final exam" (test) contain patches from **the
same 10 cities** — often just a couple of kilometers apart. A flexible combiner tuned on the
practice set can quietly memorize each city's local quirks ("in this city, when the models
disagree like *this*, the answer is usually *that*") and replay them on the exam. That's
not skill that would transfer to an 11th city.

The fix: **leave-one-city-out grading.** Tune the combiner on 9 cities, grade it on the
10th, rotate through all ten, so it is always graded on a city it never tuned on. Under
fair grading:

- the fancy combiner collapsed to **0.612 — worse than a simple vote**. Its 0.776 was
  almost entirely memorized city quirks. Idea abandoned.
- the simple weighted vote barely moved: **0.687** vs 0.692. Its gain is real. This is the
  number we report with a completely clean conscience.

## Experiment 6: Helping the model acclimatize to each city (failed, informatively)

The scores vary hugely by city: Munich ~0.90, but Nairobi and Santiago ~0.47. The model
clearly struggles with cities unlike its training set. A popular cheap remedy: when the
model arrives in a new city, let it **re-tune its internal sense of "normal"** (technically,
its batch-normalization statistics) using unlabeled data from that city — like eyes
adjusting to local light. We tested two standard variants (AdaBN and TENT), carefully using
only unlabeled data.

**It backfired: scores dropped ~5 points overall.** It genuinely helped two cities (Tehran
+12!) but wrecked others (Santiago −16). The diagnosis is the valuable part: the method
assumes new cities *look* different but *contain* the same mix of neighborhood types. In
reality the opposite dominates — Santiago doesn't just look different, it **is made of a
different mix of categories** than the training cities. Re-tuning "normal" on such a city
teaches the model the wrong normal. So the city-to-city struggle is about *what's there*,
not *how it looks* — which rules out this whole family of quick fixes and points to
different remedies.

---

## Final scoreboard

| What | Score (kappa) |
|---|---|
| Where we started (best single model, June) | 0.619 |
| Best single model now (noisy student, round 2) | **0.650** |
| Three-model committee, honest weighted vote | **0.687** |
| Same, with a small tuning asterisk | 0.692 |
| ~~Fancy learned combiner~~ | ~~0.776~~ — memorized city quirks, rejected |

## What we learned (the short version)

1. **A committee of different data sources beats any single model** — and beats every
   architecture tweak we tried.
2. **Free rough labels + careful filtering = real gains** (+3 points), but the trick wears
   out after about two rounds.
3. **Our models are systematically under-confident** — any rule that reads their confidence
   must account for that.
4. **Always ask "could this have been memorized?"** The fair leave-one-city-out re-grading
   cost minutes and overturned our most impressive-looking result.
5. **The hard cities are hard because of what they contain, not how they look** — so the
   next real improvement must deal with different category mixes (or new labels), not
   appearance adjustments.

## What could come next

- Estimate each new city's category mix from the model's own predictions and correct for it
  (directly targets the Experiment 6 failure).
- Get some human labels for under-represented city types (a Nairobi/Santiago-like city
  would help most) and for the rarest categories, which no amount of self-training can fix.
- Otherwise: the numbers above are stable, honestly graded, and ready to be written up.
