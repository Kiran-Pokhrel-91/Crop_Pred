"""Feature engineering shared by training and inference.

Builds temporal aggregations, index summaries, climate anomalies, terrain
interactions, encodings, and missing-value indicators. The fitted feature
specification is persisted with the model so inference reproduces training-time
transformations exactly.
"""
