/* Air Quality Dashboard — Professional Theme (v3) */

:root {
  --bg: #0f141b;
  --bg-soft: #141b24;
  --panel: #171f2a;
  --text: #e6e9ee;
  --muted: #a8b1bf;
  --border: rgba(255, 255, 255, 0.08);
  --accent: #4f6f8f;
  --accent-soft: rgba(79, 111, 143, 0.2);
  --radius: 10px;
  --font-main: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}

[data-testid="stAppViewContainer"] {
  background: var(--bg);
}

/* Hide Streamlit deploy/menu chrome */
[data-testid="stToolbar"],
[data-testid="stDeployButton"] {
  display: none !important;
  visibility: hidden !important;
}

/* Hide Streamlit header artifact */
[data-testid="stHeader"] {
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
}

section[data-testid="stMain"] {
  padding-top: 0.5rem;
}

.block-container {
  max-width: 1200px;
  padding-top: 2.25rem;
  padding-bottom: 2.2rem;
}

section[data-testid="stSidebar"] {
  background: #101722;
  border-right: 1px solid var(--border);
}

* {
  font-family: var(--font-main);
}

h1, h2, h3 {
  color: var(--text) !important;
  font-weight: 620 !important;
  letter-spacing: -0.01em;
}

p, li, span, div, label {
  color: var(--text);
}

small, .stCaptionContainer, [data-testid="stCaptionContainer"] {
  color: var(--muted) !important;
}

hr {
  border-color: var(--border) !important;
}

.aq-hero {
  background: linear-gradient(180deg, rgba(79, 111, 143, 0.18) 0%, rgba(79, 111, 143, 0.1) 100%);
  border: 1px solid rgba(79, 111, 143, 0.4);
  padding: 16px 18px;
  border-radius: var(--radius);
}

.aq-hero-title {
  font-size: 1.85rem;
  font-weight: 700;
  line-height: 1.15;
}

.aq-hero-sub {
  margin-top: 8px;
  color: #c7d0dc;
}

.aq-panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px 14px;
}

.aq-panel-title {
  font-weight: 640;
  color: var(--text);
  margin-bottom: 6px;
}

.aq-panel-body {
  color: #d7dde6;
  line-height: 1.45;
}

div[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px;
}

div[data-testid="stMetric"] label {
  color: var(--muted) !important;
  font-weight: 560;
}

div[data-testid="stMetricValue"] {
  color: var(--text) !important;
}

div[data-testid="stMetricDelta"] {
  color: var(--muted) !important;
}

div[data-testid="stDataFrame"] {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 8px;
}

.stDataFrame {
  font-size: 0.9rem;
}

div[data-testid="stExpander"] {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
}

div[data-testid="stExpander"] summary {
  color: var(--text);
  font-weight: 560;
}

[data-testid="stSidebar"] [role="radiogroup"] {
  gap: 6px;
}

[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] {
  display: none;
}

[data-testid="stSidebar"] div[role="radio"] {
  padding: 10px;
  border-radius: 8px;
  border: 1px solid transparent;
}

[data-testid="stSidebar"] div[role="radio"][aria-checked="true"] {
  background: var(--accent-soft);
  border: 1px solid rgba(79, 111, 143, 0.55);
}
