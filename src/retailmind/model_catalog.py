"""Stable candidate identifiers and user-facing labels shared by reports and apps."""

TIMESTRAL = "timestral209m_lora"
LSTM = "lstm"
TIMESFM3 = "timesfm3_head_tuned"
MODELS = (TIMESTRAL, LSTM, TIMESFM3)
LABELS = {
    TIMESTRAL: "Timestral 209M LoRA",
    LSTM: "Trained LSTM",
    TIMESFM3: "Google TimesFM 3.0 head-tuned",
}
POLICY_LABELS = {model: f"{name} reorder plan" for model, name in LABELS.items()}
