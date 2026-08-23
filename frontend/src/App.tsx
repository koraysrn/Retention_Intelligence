import { useEffect, useState } from "react";
import type { CSSProperties, FormEvent, ReactNode } from "react";
import { getCustomer, getInsights, getPopulation, getSummary, sendChat } from "./api";
import type {
  ChatMessage,
  CustomerProfile,
  Insight,
  Population,
  Summary,
} from "./types";

/* ============================== THEME ============================== */

const colors = {
  bg: "#f6f7f9",
  card: "#ffffff",
  text: "#111827",
  muted: "#6b7280",
  border: "#e5e7eb",
  primary: "#2563eb",
  primarySoft: "#eff6ff",
  high: "#dc2626",
  medium: "#d97706",
  low: "#16a34a",
  sidebar: "#ffffff",
};

const font =
  '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';

const baseFont: CSSProperties = {
  fontFamily: font,
  color: colors.text,
};

const cardStyle: CSSProperties = {
  background: colors.card,
  border: `1px solid ${colors.border}`,
  borderRadius: 14,
  padding: 18,
  boxShadow: "0 1px 2px rgba(17,24,39,0.04), 0 4px 16px rgba(17,24,39,0.04)",
};

const sectionTitleStyle: CSSProperties = {
  margin: 0,
  fontSize: "1.02rem",
  fontWeight: 700,
  letterSpacing: "-0.01em",
};

const sectionSubStyle: CSSProperties = {
  margin: "3px 0 0",
  color: colors.muted,
  fontSize: "0.82rem",
};

/* ============================== HELPERS ============================== */

function fmtNum(value: number | null | undefined): string {
  if (value == null) return "—";
  return Number(value).toLocaleString("en-US", { maximumFractionDigits: 2 });
}

function fmtCurrency(value: number | null | undefined): string {
  if (value == null) return "—";
  return (
    "$" +
    Number(value).toLocaleString("en-US", {
      maximumFractionDigits: value >= 1000 ? 0 : 2,
    })
  );
}

function fmtPct(value: number | null | undefined, digits = 1): string {
  if (value == null) return "—";
  return Number(value).toLocaleString("en-US", {
    maximumFractionDigits: digits,
  }) + "%";
}

function fmtDate(value: string | null | undefined): string {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function fmtBool(value: number | null | undefined): string {
  if (value == null) return "—";
  return value === 1 ? "Yes" : "No";
}

const riskTierColor = (tier: string): string =>
  tier === "high" ? colors.high : tier === "medium" ? colors.medium : colors.low;

/* ============================== UI ATOMS ============================== */

function Card({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return <div style={{ ...cardStyle, ...style }}>{children}</div>;
}

function SectionTitle({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <h2 style={sectionTitleStyle}>{title}</h2>
      {subtitle ? <p style={sectionSubStyle}>{subtitle}</p> : null}
    </div>
  );
}

function KpiCard({
  label,
  value,
  accent,
  icon,
}: {
  label: string;
  value: string;
  accent: string;
  icon: ReactNode;
}) {
  return (
    <Card
      style={{
        display: "flex",
        alignItems: "center",
        gap: 13,
        padding: 16,
      }}
    >
      <div
        style={{
          width: 46,
          height: 46,
          borderRadius: 13,
          display: "grid",
          placeItems: "center",
          color: accent,
          background: `${accent}14`,
          border: `1px solid ${accent}2e`,
          flexShrink: 0,
        }}
      >
        {icon}
      </div>
      <div>
        <div style={{ color: colors.muted, fontSize: "0.78rem" }}>{label}</div>
        <div
          style={{
            fontSize: "1.5rem",
            fontWeight: 750,
            letterSpacing: "-0.02em",
            fontVariantNumeric: "tabular-nums",
            marginTop: 2,
          }}
        >
          {value}
        </div>
      </div>
    </Card>
  );
}

function MetricBox({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div
      style={{
        background: "#f8fafc",
        border: `1px solid ${colors.border}`,
        borderRadius: 11,
        padding: "11px 13px",
      }}
    >
      <div style={{ color: colors.muted, fontSize: "0.7rem" }}>{label}</div>
      <div
        style={{
          fontSize: "1rem",
          fontWeight: 700,
          fontVariantNumeric: "tabular-nums",
          marginTop: 3,
          wordBreak: "break-word",
        }}
      >
        {value}
      </div>
      {hint ? (
        <div style={{ color: colors.muted, fontSize: "0.68rem", marginTop: 3 }}>
          {hint}
        </div>
      ) : null}
    </div>
  );
}

function CategoryCard({
  title,
  icon,
  accent,
  boxes,
}: {
  title: string;
  icon: ReactNode;
  accent: string;
  boxes: { label: string; value: string; hint?: string }[];
}) {
  return (
    <Card>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 9,
          marginBottom: 12,
        }}
      >
        <div
          style={{
            width: 30,
            height: 30,
            borderRadius: 9,
            display: "grid",
            placeItems: "center",
            color: accent,
            background: `${accent}1f`,
            border: `1px solid ${accent}2e`,
          }}
        >
          {icon}
        </div>
        <span style={{ fontWeight: 700, fontSize: "0.92rem" }}>{title}</span>
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
          gap: 10,
        }}
      >
        {boxes.map((b) => (
          <MetricBox key={b.label} label={b.label} value={b.value} hint={b.hint} />
        ))}
      </div>
    </Card>
  );
}

/* ============================== ICONS ============================== */

const iconProps = {
  width: 18,
  height: 18,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

const usersIcon = (
  <svg {...iconProps}>
    <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
    <path d="M16 3.13a4 4 0 0 1 0 7.75" />
  </svg>
);

const alertIcon = (
  <svg {...iconProps}>
    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
    <path d="M12 9v4" />
    <path d="M12 17h.01" />
  </svg>
);

const percentIcon = (
  <svg {...iconProps}>
    <line x1="19" x2="5" y1="5" y2="19" />
    <circle cx="6.5" cy="6.5" r="2.5" />
    <circle cx="17.5" cy="17.5" r="2.5" />
  </svg>
);

const trendIcon = (
  <svg {...iconProps}>
    <path d="m22 7-8.5 8.5-5-5L2 17" />
    <path d="M16 7h6v6" />
  </svg>
);

const activityIcon = (
  <svg {...iconProps}>
    <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
  </svg>
);

const clockIcon = (
  <svg {...iconProps}>
    <circle cx="12" cy="12" r="10" />
    <polyline points="12 6 12 12 16 14" />
  </svg>
);

const mouseIcon = (
  <svg {...iconProps}>
    <rect x="6" y="2.5" width="12" height="19" rx="6" />
    <path d="M12 7v4" />
  </svg>
);

const starIcon = (
  <svg {...iconProps}>
    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
  </svg>
);

const sendIcon = (
  <svg {...iconProps}>
    <path d="M22 2 11 13" />
    <path d="M22 2l-7 20-4-9-9-4 20-7z" />
  </svg>
);

const trashIcon = (
  <svg {...iconProps}>
    <path d="M3 6h18" />
    <path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2" />
    <path d="m19 6-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
  </svg>
);

/* ============================== APP ============================== */

const SUGGESTIONS = [
  "Which segments are most at risk?",
  "What are the top churn drivers?",
  "Which customers should we win back first?",
  "What offer maximizes retention ROI?",
];

export default function App() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [insights, setInsights] = useState<Insight[]>([]);
  const [population, setPopulation] = useState<Population | null>(null);
  const [customer, setCustomer] = useState<CustomerProfile | null>(null);
  const [customerInput, setCustomerInput] = useState("");
  const [customerError, setCustomerError] = useState("");
  const [loadingCustomer, setLoadingCustomer] = useState(false);

  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "bot",
      text:
        "Hello, I'm your retention analytics assistant.\n\n• Explain churn risk and its drivers\n• Recommend win-back campaigns\n• Interpret the metrics on this dashboard",
    },
  ]);
  const [chatInput, setChatInput] = useState("");
  const [typing, setTyping] = useState(false);

  useEffect(() => {
    getSummary().then(setSummary).catch(() => undefined);
    getInsights()
      .then((r) => setInsights(r.insights ?? []))
      .catch(() => undefined);
    getPopulation().then(setPopulation).catch(() => undefined);
  }, []);

  const analyzeCustomer = async (e: FormEvent) => {
    e.preventDefault();
    const id = customerInput.trim();
    if (!id) {
      setCustomerError("Please enter a customer ID.");
      setCustomer(null);
      return;
    }
    setLoadingCustomer(true);
    setCustomerError("");
    try {
      const profile = await getCustomer(id);
      setCustomer(profile);
    } catch {
      setCustomer(null);
      setCustomerError("Customer not found or an error occurred.");
    } finally {
      setLoadingCustomer(false);
    }
  };

  const submitChat = async (e: FormEvent) => {
    e.preventDefault();
    const text = chatInput.trim();
    if (!text) return;
    setMessages((prev) => [...prev, { role: "user", text }]);
    setChatInput("");
    setTyping(true);
    try {
      const res = await sendChat(text, customer ? customer.customer_id : null);
      setMessages((prev) => [...prev, { role: "bot", text: res.reply }]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: "Something went wrong, please try again." },
      ]);
    } finally {
      setTyping(false);
    }
  };

  const sendSuggestion = (text: string) => {
    setChatInput("");
    setMessages((prev) => [...prev, { role: "user", text }]);
    setTyping(true);
    sendChat(text, customer ? customer.customer_id : null)
      .then((res) =>
        setMessages((prev) => [...prev, { role: "bot", text: res.reply }])
      )
      .catch(() =>
        setMessages((prev) => [
          ...prev,
          { role: "bot", text: "Something went wrong, please try again." },
        ])
      )
      .finally(() => setTyping(false));
  };

  const clearChat = () =>
    setMessages([{ role: "bot", text: "Conversation cleared. How can I help you?" }]);

  const riskDistribution = summary?.risk_distribution ?? {};
  const riskTotal =
    Object.values(riskDistribution).reduce((a, b) => a + b, 0) || 1;

  return (
    <div style={{ ...baseFont, display: "flex", minHeight: "100vh" }}>
      {/* ==================== CHAT (20%) ==================== */}
      <aside
        style={{
          width: "20%",
          minWidth: 300,
          maxWidth: 380,
          background: colors.sidebar,
          borderRight: `1px solid ${colors.border}`,
          display: "flex",
          flexDirection: "column",
          position: "sticky",
          top: 0,
          height: "100vh",
        }}
      >
        <div
          style={{
            padding: "16px 18px 14px",
            borderBottom: `1px solid ${colors.border}`,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <h2 style={{ margin: 0, fontSize: "1rem" }}>Assistant</h2>
          <button
            onClick={clearChat}
            title="Clear conversation"
            style={{
              background: "#f1f5f9",
              border: 0,
              color: "#475569",
              width: 32,
              height: 32,
              borderRadius: 9,
              cursor: "pointer",
              display: "grid",
              placeItems: "center",
            }}
          >
            {trashIcon}
          </button>
        </div>

        {customer ? (
          <div
            style={{
              padding: "9px 16px",
              background: "#f8fafc",
              borderBottom: `1px solid ${colors.border}`,
              fontSize: "0.78rem",
              display: "flex",
              alignItems: "center",
              gap: 8,
              color: "#475569",
            }}
          >
            <span
              style={{
                background: colors.primarySoft,
                color: colors.primary,
                fontWeight: 600,
                padding: "3px 9px",
                borderRadius: 999,
                fontSize: "0.74rem",
              }}
            >
              {customer.name || customer.customer_id}
            </span>
            <span>context added</span>
            <button
              onClick={() => setCustomer(null)}
              style={{
                border: 0,
                background: "transparent",
                color: "#94a3b8",
                cursor: "pointer",
                marginLeft: "auto",
                fontSize: "0.82rem",
              }}
            >
              remove
            </button>
          </div>
        ) : null}

        <div
          style={{
            flex: 1,
            overflowY: "auto",
            padding: 16,
            display: "flex",
            flexDirection: "column",
            gap: 10,
          }}
        >
          {messages.map((m, i) => (
            <div
              key={i}
              style={{
                alignSelf: m.role === "bot" ? "flex-start" : "flex-end",
                maxWidth: "88%",
                padding: "10px 13px",
                borderRadius: 14,
                background: m.role === "bot" ? "#f1f5f9" : colors.primary,
                color: m.role === "bot" ? "#1f2937" : "#ffffff",
                borderBottomLeftRadius: m.role === "bot" ? 6 : 14,
                borderBottomRightRadius: m.role === "bot" ? 14 : 6,
                fontSize: "0.86rem",
                lineHeight: 1.5,
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
              }}
            >
              {m.text}
            </div>
          ))}
          {typing ? (
            <div
              style={{
                alignSelf: "flex-start",
                display: "inline-flex",
                gap: 4,
                padding: "13px 15px",
                background: "#f1f5f9",
                borderRadius: 14,
                borderBottomLeftRadius: 6,
              }}
            >
              <Dot delay="0s" />
              <Dot delay="0.15s" />
              <Dot delay="0.3s" />
            </div>
          ) : null}
        </div>

        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: 7,
            padding: "10px 14px 12px",
            borderTop: `1px solid ${colors.border}`,
            background: "#fbfcfe",
          }}
        >
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => sendSuggestion(s)}
              style={{
                background: "#fff",
                border: `1px solid ${colors.border}`,
                color: "#475569",
                borderRadius: 999,
                padding: "6px 11px",
                fontSize: "0.74rem",
                cursor: "pointer",
                textAlign: "left",
                fontFamily: font,
              }}
            >
              {s}
            </button>
          ))}
        </div>

        <form
          onSubmit={submitChat}
          style={{
            padding: "11px 14px 14px",
            borderTop: `1px solid ${colors.border}`,
            display: "flex",
            alignItems: "flex-end",
            gap: 8,
            background: "#fff",
          }}
        >
          <input
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            placeholder="Ask about churn, segments, or campaigns…"
            style={{
              flex: 1,
              minWidth: 0,
              padding: "10px 13px",
              border: `1px solid ${colors.border}`,
              borderRadius: 12,
              fontSize: "0.88rem",
              fontFamily: font,
              color: colors.text,
              outline: "none",
            }}
          />
          <button
            type="submit"
            title="Send"
            style={{
              background: colors.primary,
              color: "#fff",
              border: 0,
              width: 40,
              height: 40,
              borderRadius: 11,
              cursor: "pointer",
              display: "grid",
              placeItems: "center",
              flexShrink: 0,
            }}
          >
            {sendIcon}
          </button>
        </form>
      </aside>

      {/* ==================== DASHBOARD (80%) ==================== */}
      <main
        style={{
          flex: 1,
          minWidth: 0,
          background: colors.bg,
          overflowY: "auto",
        }}
      >
        <header
          style={{
            background: colors.card,
            borderBottom: `1px solid ${colors.border}`,
            padding: "16px 24px",
            position: "sticky",
            top: 0,
            zIndex: 2,
          }}
        >
          <h1
            style={{
              margin: 0,
              fontSize: "1.18rem",
              fontWeight: 700,
              letterSpacing: "-0.02em",
              maxWidth: 1280,
              marginInline: "auto",
            }}
          >
            Retention Intelligence
          </h1>
        </header>

        <div
          style={{
            maxWidth: 1280,
            margin: "0 auto",
            padding: "20px 24px 48px",
            display: "flex",
            flexDirection: "column",
            gap: 16,
          }}
        >
          {/* KPIs */}
          <section
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
              gap: 14,
            }}
          >
            <KpiCard
              label="Total Customers"
              value={summary ? fmtNum(summary.total_customers) : "—"}
              accent={colors.primary}
              icon={usersIcon}
            />
            <KpiCard
              label="High Risk"
              value={summary ? fmtNum(summary.high_risk_count) : "—"}
              accent={colors.high}
              icon={alertIcon}
            />
            <KpiCard
              label="Avg Churn Probability"
              value={
                summary ? fmtPct((summary.avg_churn_probability ?? 0) * 100) : "—"
              }
              accent={colors.medium}
              icon={percentIcon}
            />
          </section>

          {/* Customer analysis (compact) */}
          <Card style={{ padding: 14 }}>
            <SectionTitle title="Customer Analysis" />
            <form
              onSubmit={analyzeCustomer}
              style={{ display: "flex", gap: 8, marginBottom: 10 }}
            >
              <input
                value={customerInput}
                onChange={(e) => setCustomerInput(e.target.value)}
                placeholder="Enter customer ID (e.g. 1)"
                style={{
                  flex: 1,
                  minWidth: 0,
                  padding: "10px 13px",
                  border: `1px solid ${colors.border}`,
                  borderRadius: 10,
                  fontSize: "0.9rem",
                  fontFamily: font,
                  color: colors.text,
                  outline: "none",
                  background: "#f8fafc",
                }}
              />
              <button
                type="submit"
                style={{
                  background: colors.primary,
                  color: "#fff",
                  border: 0,
                  borderRadius: 10,
                  padding: "0 18px",
                  fontWeight: 600,
                  fontSize: "0.88rem",
                  cursor: "pointer",
                  fontFamily: font,
                }}
              >
                Analyze
              </button>
            </form>

            {loadingCustomer ? (
              <div style={{ color: colors.muted, fontSize: "0.86rem" }}>
                Analyzing customer…
              </div>
            ) : customerError ? (
              <div style={{ color: colors.high, fontSize: "0.86rem" }}>
                {customerError}
              </div>
            ) : customer ? (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 14,
                  flexWrap: "wrap",
                  padding: "10px 12px",
                  border: `1px dashed ${colors.border}`,
                  borderRadius: 10,
                  background: "#fbfcfe",
                }}
              >
                <span style={{ fontWeight: 700 }}>{customer.name}</span>
                <span style={{ color: colors.muted, fontSize: "0.82rem" }}>
                  #{customer.customer_id}
                </span>
                <span
                  style={{
                    fontSize: "0.74rem",
                    fontWeight: 700,
                    padding: "4px 11px",
                    borderRadius: 999,
                    textTransform: "capitalize",
                    background: `${riskTierColor(customer.risk_tier)}1f`,
                    color: riskTierColor(customer.risk_tier),
                  }}
                >
                  {customer.risk_tier} risk
                </span>
                <span
                  style={{
                    fontWeight: 750,
                    fontVariantNumeric: "tabular-nums",
                  }}
                >
                  {fmtPct(customer.churn_probability * 100, 0)}
                </span>
              </div>
            ) : (
              <div
                style={{
                  color: colors.muted,
                  fontSize: "0.84rem",
                  padding: "10px 12px",
                  border: `1px dashed ${colors.border}`,
                  borderRadius: 10,
                  background: "#fbfcfe",
                }}
              >
                Enter a customer ID above to see risk score and profile.
              </div>
            )}
          </Card>

          {/* Category metric boxes */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
              gap: 16,
            }}
          >
            <CategoryCard
              title="Sales Performance"
              accent={colors.low}
              icon={trendIcon}
              boxes={[
                {
                  label: "Total Orders",
                  value: customer
                    ? fmtNum(customer.total_orders)
                    : fmtNum(population?.total_orders_avg),
                  hint: customer ? "Orders placed so far" : "Avg orders per customer",
                },
                {
                  label: "Total Spend (USD)",
                  value: customer
                    ? fmtCurrency(customer.total_spend_usd)
                    : fmtCurrency(population?.total_spend_sum),
                  hint: customer ? "Lifetime spend" : "Total across all customers",
                },
                {
                  label: "Avg Basket Value",
                  value: customer
                    ? fmtCurrency(customer.avg_order_value)
                    : fmtCurrency(population?.avg_order_value_avg),
                  hint: "Spend per order",
                },
                {
                  label: "Avg Discount Rate",
                  value: customer
                    ? fmtPct(customer.avg_discount_pct)
                    : fmtPct(population?.avg_discount_pct_avg),
                  hint: "Discount usage habit",
                },
                {
                  label: "Predicted CLV (12m)",
                  value: customer
                    ? fmtCurrency(customer.predicted_clv)
                    : fmtCurrency(population?.predicted_clv_avg),
                  hint: "BG/NBD + Gamma-Gamma",
                },
              ]}
            />
            <CategoryCard
              title="Customer Profile"
              accent={colors.primary}
              icon={usersIcon}
              boxes={[
                {
                  label: "Demographics",
                  value: customer
                    ? [
                        customer.age ? `${customer.age} yrs` : "",
                        customer.age_group,
                        customer.country,
                      ]
                        .filter(Boolean)
                        .join(" · ")
                    : `${fmtNum(population?.age_avg)} yrs avg · ${fmtNum(
                        population?.total_customers
                      )} customers`,
                  hint: customer ? "Age, group, country" : "Population average",
                },
                {
                  label: "Segment",
                  value: customer ? customer.segment : population?.segment_top ?? "—",
                  hint: "RFM + K-Means cluster",
                },
                {
                  label: "CLV Tier",
                  value: customer
                    ? customer.predicted_clv_tier
                    : population?.clv_tier_top ?? "—",
                  hint: "Predicted lifetime value",
                },
                {
                  label: "Repeat Customer",
                  value: customer
                    ? fmtBool(customer.is_repeat_customer)
                    : fmtPct(population?.repeat_rate),
                  hint: customer ? "Purchased more than once" : "Repeat purchase rate",
                },
              ]}
            />
            <CategoryCard
              title="Digital Engagement"
              accent={colors.medium}
              icon={activityIcon}
              boxes={[
                {
                  label: "Total Sessions",
                  value: customer
                    ? fmtNum(customer.total_sessions)
                    : fmtNum(population?.total_sessions_avg),
                  hint: "Site / app opens",
                },
                {
                  label: "Preferred Device (Session)",
                  value: customer?.preferred_device_sess ?? population?.device_sess_top ?? "—",
                  hint: "Most used device",
                },
                {
                  label: "Preferred Source (Session)",
                  value: customer?.preferred_source_sess ?? population?.source_sess_top ?? "—",
                  hint: "Traffic source",
                },
                {
                  label: "Cart Abandonment",
                  value: customer
                    ? fmtBool(customer.has_abandoned_cart)
                    : fmtPct(population?.cart_abandon_rate),
                  hint: customer ? "Left a cart behind" : "Cart abandonment rate",
                },
                {
                  label: "Cart Abandon Risk",
                  value: customer
                    ? fmtPct((customer.cart_abandon_probability ?? 0) * 100, 0)
                    : fmtPct(population?.cart_abandon_prob_avg),
                  hint: "Model propensity",
                },
              ]}
            />
            <CategoryCard
              title="Experience & Preferences"
              accent="#0f766e"
              icon={starIcon}
              boxes={[
                {
                  label: "Preferred Payment",
                  value: customer?.preferred_payment ?? population?.payment_top ?? "—",
                  hint: "Card, PayPal, COD…",
                },
                {
                  label: "Preferred Device (Order)",
                  value: customer?.preferred_device_ord ?? population?.device_ord_top ?? "—",
                  hint: "Device used to order",
                },
                {
                  label: "Preferred Source (Order)",
                  value: customer?.preferred_source ?? population?.source_ord_top ?? "—",
                  hint: "Order channel",
                },
                {
                  label: "Top Category",
                  value: customer?.top_category_bought ?? population?.top_category ?? "—",
                  hint: "Most bought category",
                },
                {
                  label: "Avg Rating",
                  value:
                    customer?.avg_rating_given != null
                      ? String(customer.avg_rating_given)
                      : population?.avg_rating_avg != null
                        ? String(population.avg_rating_avg)
                        : "—",
                  hint: "Average review score",
                },
                {
                  label: "Discount Sensitivity",
                  value: customer
                    ? fmtPct(customer.discount_sensitivity, 0)
                    : fmtPct(population?.discount_sensitivity_avg),
                  hint: "Incentive propensity",
                },
              ]}
            />
            <CategoryCard
              title="Behavior Boxes"
              accent="#db2777"
              icon={mouseIcon}
              boxes={[
                {
                  label: "Last Session Date",
                  value: customer ? fmtDate(customer.last_session_date) : "—",
                  hint: customer
                    ? "Most recent visit"
                    : "Select a customer to view",
                },
                {
                  label: "Preferred Source",
                  value: customer ? customer.preferred_source ?? "—" : "—",
                  hint: customer
                    ? "Most frequent order channel"
                    : "Select a customer to view",
                },
                {
                  label: "Last Order Date",
                  value: customer ? fmtDate(customer.last_order_date) : "—",
                  hint: customer
                    ? "Most recent purchase"
                    : "Select a customer to view",
                },
              ]}
            />
            <CategoryCard
              title="Time Boxes"
              accent="#7c3aed"
              icon={clockIcon}
              boxes={[
                {
                  label: "Signup Date",
                  value: customer ? fmtDate(customer.signup_date) : "—",
                  hint: customer
                    ? "First registered date"
                    : "Select a customer to view",
                },
                {
                  label: "First Session Date",
                  value: customer ? fmtDate(customer.first_session_date) : "—",
                  hint: customer
                    ? "First site / app visit"
                    : "Select a customer to view",
                },
              ]}
            />
          </div>

          {/* Risk distribution + insights */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
              gap: 16,
            }}
          >
            <Card>
              <SectionTitle title="Risk Distribution" />
              <div style={{ display: "flex", flexDirection: "column", gap: 13 }}>
                {Object.entries(riskDistribution).map(([tier, count]) => {
                  const pct = (count / riskTotal) * 100;
                  return (
                    <div
                      key={tier}
                      style={{
                        display: "grid",
                        gridTemplateColumns: "78px 1fr 56px",
                        alignItems: "center",
                        gap: 12,
                      }}
                    >
                      <span
                        style={{
                          fontSize: "0.82rem",
                          color: colors.muted,
                          textTransform: "capitalize",
                        }}
                      >
                        {tier}
                      </span>
                      <div
                        style={{
                          background: "#eef1f5",
                          borderRadius: 999,
                          height: 10,
                          overflow: "hidden",
                        }}
                      >
                        <div
                          style={{
                            width: `${pct.toFixed(1)}%`,
                            height: "100%",
                            borderRadius: 999,
                            background: riskTierColor(tier),
                          }}
                        />
                      </div>
                      <span
                        style={{
                          textAlign: "right",
                          fontSize: "0.82rem",
                          fontWeight: 600,
                          fontVariantNumeric: "tabular-nums",
                        }}
                      >
                        {fmtNum(count)}
                      </span>
                    </div>
                  );
                })}
                {Object.keys(riskDistribution).length === 0 ? (
                  <div style={{ color: colors.muted, fontSize: "0.86rem" }}>
                    No risk data available.
                  </div>
                ) : null}
              </div>
            </Card>

            <Card>
              <SectionTitle
                title="Business Insights"
                subtitle="Five data-backed takeaways."
              />
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {insights.length === 0 ? (
                  <div style={{ color: colors.muted, fontSize: "0.86rem" }}>
                    Loading insights…
                  </div>
                ) : (
                  insights.map((ins, i) => (
                    <div
                      key={i}
                      style={{
                        display: "flex",
                        gap: 10,
                        alignItems: "flex-start",
                      }}
                    >
                      <span
                        style={{
                          width: 22,
                          height: 22,
                          borderRadius: 999,
                          background: colors.primarySoft,
                          color: colors.primary,
                          display: "grid",
                          placeItems: "center",
                          fontSize: "0.72rem",
                          fontWeight: 700,
                          flexShrink: 0,
                        }}
                      >
                        {i + 1}
                      </span>
                      <div>
                        <div style={{ fontWeight: 700, fontSize: "0.86rem" }}>
                          {ins.title}
                        </div>
                        <div
                          style={{ color: colors.muted, fontSize: "0.82rem" }}
                        >
                          {ins.text}
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}

function Dot({ delay }: { delay: string }) {
  return (
    <span
      style={{
        width: 6,
        height: 6,
        borderRadius: "50%",
        background: "#94a3b8",
        animation: "bounce 1.2s infinite ease-in-out",
        animationDelay: delay,
      }}
    />
  );
}
