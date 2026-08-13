"""Tests for jacobian module — integration with real models only.

compute_jacobian_for_tokens is tightly coupled to Qwen3.5 API (rotary_emb,
GDN cache, layer_types). Unit tests with mocks are fragile. Verification
happens via Exp 1 (identity: same model twice -> L_bridge < 0.01).
"""
