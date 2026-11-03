# Viva questions (answers are based on this project's actual code and results)

**1. Why parse logs with Drain instead of using raw text?**
Raw lines contain changing values (block IDs, IPs, numbers), so no two lines look identical. Drain groups lines into templates such as `Receiving block <BLK> src: /<IP> dest: /<IP>` (HDFS gave 25 templates from 104,815 lines). Each template becomes an event ID, which models can count or predict. See `src/parse.py`.

**2. How did you build sequences, and why different methods for HDFS and BGL?**
HDFS logs have a block ID, so one sequence = all events of one block, labelled from the label file. BGL has no such ID, so I use sliding windows (size 20, step 10); a window is anomalous if any line in it is. See `src/sequences.py`.

**3. Why a chronological split instead of a random one?**
In real monitoring you train on the past and detect in the future. A random split would let the model see later behaviour, hide drift and inflate scores. I split by time (first 70% train, last 30% test). A consequence on BGL: its alerts cluster early, so train is 7.4% anomalous but test only 0.6%, which is why fixed thresholds over-flag and BGL precision is very low (0.03-0.06).

**4. Why train on normal data only?**
Anomalies are rare and varied, so labels for them are scarce. DeepLog-style and Isolation Forest models learn what normal looks like and flag departures. The only labelled use I made of train labels was choosing the event-ID Isolation Forest threshold on the train split, never on test data.

**5. How does the DeepLog-style LSTM detect anomalies?**
It reads the last 10 events and predicts the next one. If the true next event is not in the top-k (k=9) predictions at any position, the sequence is flagged; an event never seen in training always counts as a miss. On HDFS it gave precision 0.93 but recall 0.46, since many anomalies follow valid event orders.

**6. How does Isolation Forest work here and why did it do badly on HDFS?**
Each sequence becomes a vector of event counts; random trees isolate unusual vectors in fewer splits, giving a higher score. On HDFS over 75% of anomalous test blocks had the same count vector as normal blocks, so there was nothing to isolate (F1 0.19). That is a limitation of count features, not a bug I could fix.

**7. What is the main idea of the cross-dataset part?**
Event IDs mean different things in different datasets, so a model can't transfer. I embed each template's text with MiniLM (384-d) so both datasets share a vector space, then use an LSTM that predicts the next event's embedding (score = 1 − cosine) and an Isolation Forest on mean embeddings. Honest result: mixed and direction-dependent. HDFS→BGL with the Isolation Forest still ranks well (ROC-AUC 0.88) but F1 is only 0.06; HDFS→BGL with the LSTM fails (AUC 0.44) and BGL→HDFS is weak (0.37 / 0.60). Likely causes are different vocabularies, anomaly definitions and log styles; BGL test also has only 38 anomalous windows (0.6%), so F1 is very low everywhere on BGL and I rely on ROC-AUC/PR-AUC.

**8. How did you pick the threshold for a new dataset, and how reliable is it?**
Score a sample of the target's normal training sequences with the source model and use the 95th percentile (no anomaly labels). It is fragile: the sensitivity table shows F1 for HDFS→HDFS EmbIForest going from 0.37 (90th percentile) to 0.00 (95th) because many scores are tied. That is why I also report threshold-free ROC-AUC.

**9. How do you explain a flagged sequence?**
Leave-one-event-out: remove each event, re-score, and rank events by the score drop. For an HDFS block the `IOException` line dropped the score by 0.66 versus about 0.02 for others. The top lines go to a local `llama3.2:1b` via Ollama for a plain-English explanation, cached on disk, with a fallback if Ollama is down. The ranking is the dependable part; the 1B model's text can be vague, and the method can miss anomalies caused by missing events.

**10. How did you test on attacks, and what can you claim?**
I generated a clearly labelled SYNTHETIC SSH auth.log with a fast brute-force burst and a slow drip placed after the training period, and ran it through the same pipeline. Models trained on that log's normal traffic caught the fast burst (17/18 windows) but not reliably the slow drip. An earlier run had one result (BGL-trained EmbLSTM, "35/35") that looked perfect, so I checked it: its predictions were nearly constant, so it was an artifact, not detection; after retraining on 200k BGL lines it dropped to 0/35. I also caught and fixed a bug where the burst had leaked into the training portion. I can claim the pipeline runs end-to-end on another log type; I cannot claim it detects real attacks. Real attack simulation (e.g. hydra) would have to be done only on my own isolated VMs.
