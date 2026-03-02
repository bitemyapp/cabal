use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use cinch_rs::agent::config::{
    HarnessCacheConfig, HarnessEvictionConfig, HarnessPlanExecuteConfig, HarnessSessionConfig,
    HarnessSummarizerConfig,
};
use cinch_rs::prelude::*;
use clap::Parser;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

// ── Config file ──────────────────────────────────────────────────────

/// Loaded from ~/.config/cabal.toml
#[derive(Deserialize, Default, Debug)]
#[serde(default)]
struct ConfigFile {
    boss_model: Option<String>,
    models: Vec<String>,
    max_rounds: Option<u32>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    agent_max_rounds: Option<u32>,
    max_repeated_failures: Option<u32>,
}

fn config_path() -> PathBuf {
    dirs::config_dir()
        .map(|d| d.join("cabal.toml"))
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string()))
                .join(".config/cabal.toml")
        })
}

fn load_config_file() -> ConfigFile {
    let path = config_path();

    match std::fs::read_to_string(&path) {
        Ok(contents) => match toml::from_str::<ConfigFile>(&contents) {
            Ok(cfg) => {
                debug!(path = %path.display(), "Loaded config file");
                cfg
            }
            Err(e) => {
                warn!(path = %path.display(), error = %e, "Failed to parse config file, using defaults");
                ConfigFile::default()
            }
        },
        Err(_) => ConfigFile::default(),
    }
}

// ── CLI ──────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "cabal",
    about = "Multi-agent deliberation harness",
    after_help = "\
EXAMPLE:
  cabal -q \"What is the bug in src/parser.rs?\" \\
    -m z-ai/glm-5 \\
    -m openai/gpt-5.3-codex \\
    -m anthropic/claude-opus-4.6 \\
    -m google/gemini-3.1-pro-preview \\
    -m deepseek/deepseek-v3.2 \\
    -m x-ai/grok-4.1-fast \\
    --boss-model openai/gpt-5.3-codex \\
    --workdir /path/to/project \\
    --max-rounds 3

  Each -m flag adds one analyst agent using that model.
  Repeat -m to add more agents to the deliberation.

  Defaults can be set in ~/.config/cabal.toml:
    boss_model = \"openai/gpt-5.3-codex\"
    models = [\"anthropic/claude-opus-4.6\", \"openai/gpt-5.3-codex\", \"google/gemini-3.1-pro-preview\"]
    max_rounds = 3"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// The question or task to deliberate on (mutually exclusive with -f).
    #[arg(short, long, conflicts_with = "question_file")]
    question: Option<String>,

    /// Path to a file containing the question (mutually exclusive with -q).
    #[arg(short = 'f', long, conflicts_with = "question")]
    question_file: Option<PathBuf>,

    /// Models to use for analyst agents (one per agent). Overrides config file.
    #[arg(short, long = "model")]
    models: Vec<String>,

    /// Model for the boss agent. Overrides config file.
    #[arg(long)]
    boss_model: Option<String>,

    /// Working directory for source code tools.
    #[arg(short, long)]
    workdir: Option<String>,

    /// Maximum deliberation rounds. Overrides config file.
    #[arg(long)]
    max_rounds: Option<u32>,

    /// Max tokens per LLM response. Overrides config file.
    #[arg(long)]
    max_tokens: Option<u32>,

    /// Temperature for analyst agents. Overrides config file.
    #[arg(long)]
    temperature: Option<f32>,

    /// Max tool-use rounds per analyst harness invocation. Overrides config file.
    #[arg(long)]
    agent_max_rounds: Option<u32>,

    /// Abort an agent after this many consecutive identical failing tool calls.
    #[arg(long)]
    max_repeated_failures: Option<u32>,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Generate a default ~/.config/cabal.toml configuration file.
    GenerateConfig,
}

/// Resolved configuration with all defaults applied.
struct ResolvedConfig {
    question: String,
    models: Vec<String>,
    boss_model: String,
    workdir: String,
    max_rounds: u32,
    max_tokens: u32,
    temperature: f32,
    agent_max_rounds: u32,
    max_repeated_failures: u32,
}

const DEFAULT_CONFIG: &str = r#"boss_model = "openai/gpt-5.3-codex"
models = [
    "z-ai/glm-5",
    "openai/gpt-5.3-codex",
    "anthropic/claude-opus-4.6",
    "google/gemini-3.1-pro-preview",
    "deepseek/deepseek-v3.2",
    "x-ai/grok-4.1-fast",
]
max_rounds = 3
# max_tokens = 4096
# temperature = 0.3
# agent_max_rounds = 50
# max_repeated_failures = 3
"#;

fn generate_config() -> Result<(), Box<dyn std::error::Error>> {
    let path = config_path();

    if path.exists() {
        eprintln!("Config file already exists: {}", path.display());
        eprintln!("Remove it first if you want to regenerate.");
        std::process::exit(1);
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, DEFAULT_CONFIG)?;
    println!("Generated config: {}", path.display());
    Ok(())
}

fn resolve_config(cli: Cli, cfg: ConfigFile) -> Result<ResolvedConfig, String> {
    // Question: CLI only (no config file equivalent).
    let question = match (&cli.question, &cli.question_file) {
        (Some(q), _) => q.clone(),
        (_, Some(path)) => std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read question file {}: {e}", path.display()))?,
        _ => return Err("Provide either -q <question> or -f <file>".to_string()),
    };

    // Models: CLI overrides config if any -m flags were passed.
    let models = if cli.models.is_empty() {
        cfg.models
    } else {
        cli.models
    };
    if models.is_empty() {
        return Err(
            "No models specified. Use -m or set models in ~/.config/cabal.toml".to_string(),
        );
    }

    Ok(ResolvedConfig {
        question,
        models,
        boss_model: cli
            .boss_model
            .or(cfg.boss_model)
            .unwrap_or_else(|| "anthropic/claude-sonnet-4".to_string()),
        workdir: cli.workdir.unwrap_or_else(|| ".".to_string()),
        max_rounds: cli.max_rounds.or(cfg.max_rounds).unwrap_or(5),
        max_tokens: cli.max_tokens.or(cfg.max_tokens).unwrap_or(4096),
        temperature: cli.temperature.or(cfg.temperature).unwrap_or(0.3),
        agent_max_rounds: cli.agent_max_rounds.or(cfg.agent_max_rounds).unwrap_or(50),
        max_repeated_failures: cli
            .max_repeated_failures
            .or(cfg.max_repeated_failures)
            .unwrap_or(3),
    })
}

// ── Tool argument types ──────────────────────────────────────────────

#[derive(Deserialize, JsonSchema)]
struct SubmitAnalysisArgs {
    /// Your complete analysis answering the question. Be thorough.
    analysis: String,
    /// Distinct numbered points of your analysis (one sentence each).
    key_points: Vec<String>,
    /// Your confidence level from 0.0 to 1.0.
    confidence: f64,
}

#[derive(Deserialize, JsonSchema)]
struct SubmitReviewArgs {
    /// Whether you agree with the other agent's analysis.
    agrees: bool,
    /// Your detailed response to the other agent's analysis.
    response: String,
    /// Points from the other analysis you agree with (by number, 1-indexed).
    agreed_points: Vec<usize>,
    /// Points you disagree with, each with your counter-argument.
    disagreements: Vec<Disagreement>,
    /// Any new points you want to add after considering their analysis.
    new_points: Vec<String>,
}

#[derive(Deserialize, Serialize, JsonSchema, Clone, Debug)]
struct Disagreement {
    /// The point number (1-indexed) you disagree with.
    point_number: usize,
    /// Your counter-argument.
    counter_argument: String,
}

// ── Data structures for deliberation state ───────────────────────────

#[derive(Clone, Debug)]
struct AgentAnalysis {
    agent_id: usize,
    model: String,
    analysis: String,
    key_points: Vec<String>,
    confidence: f64,
}

#[derive(Clone, Debug)]
struct AgentReview {
    reviewer_id: usize,
    reviewed_id: usize,
    agrees: bool,
    response: String,
    agreed_points: Vec<usize>,
    disagreements: Vec<Disagreement>,
    new_points: Vec<String>,
}

// ── Per-model metrics tracking ────────────────────────────────────────

#[derive(Default, Debug)]
struct ModelMetrics {
    prompt_tokens: u64,
    completion_tokens: u64,
    kv_cache_reused_tokens: u64,
    api_calls: u64,
}

#[derive(Default)]
struct SessionMetrics {
    by_model: HashMap<String, ModelMetrics>,
}

impl SessionMetrics {
    fn record(
        &mut self,
        model: &str,
        prompt_tokens: u64,
        completion_tokens: u64,
        kv_cache_reused_tokens: u64,
    ) {
        let entry = self.by_model.entry(model.to_string()).or_default();
        entry.prompt_tokens += prompt_tokens;
        entry.completion_tokens += completion_tokens;
        entry.kv_cache_reused_tokens += kv_cache_reused_tokens;
        entry.api_calls += 1;
    }

    fn emit_logs(&self) {
        debug!("── Session token metrics ──────────────────────────────");
        let mut total_prompt = 0u64;
        let mut total_completion = 0u64;
        let mut total_cached = 0u64;

        for (model, m) in &self.by_model {
            debug!(
                model = %model,
                api_calls = m.api_calls,
                prompt_tokens = m.prompt_tokens,
                completion_tokens = m.completion_tokens,
                total_tokens = m.prompt_tokens + m.completion_tokens,
                kv_cache_reused_tokens = m.kv_cache_reused_tokens,
                "Model metrics"
            );
            total_prompt += m.prompt_tokens;
            total_completion += m.completion_tokens;
            total_cached += m.kv_cache_reused_tokens;
        }

        let total = total_prompt + total_completion;
        let naive_total = total_prompt + total_completion + total_cached;
        debug!(
            models = self.by_model.len(),
            total_prompt_tokens = total_prompt,
            total_completion_tokens = total_completion,
            total_tokens = total,
            total_kv_cache_reused_tokens = total_cached,
            naive_total_without_cache = naive_total,
            cache_savings_pct = if naive_total > 0 {
                format!("{:.1}%", total_cached as f64 / naive_total as f64 * 100.0)
            } else {
                "0.0%".to_string()
            },
            "Session totals"
        );
        debug!("───────────────────────────────────────────────────────");
    }
}

// ── Stuck-loop detection ─────────────────────────────────────────────

/// Detects when a model repeatedly produces the same failing tool call.
/// Shared between the event handler (which records calls) and the stop
/// signal (which checks the abort flag).
struct StuckLoopDetector {
    /// How many consecutive identical failures before aborting.
    threshold: u32,
    /// Signature of the last failing tool call: (name, arguments).
    last_failure: Option<(String, String)>,
    /// How many times the current failure has repeated.
    consecutive_failures: u32,
    /// Set to true when the threshold is hit.
    aborted: bool,
    /// Label for log messages.
    label: String,
    /// The last tool call we saw (before we know the result).
    pending_call: Option<(String, String)>,
}

impl StuckLoopDetector {
    fn new(label: String, threshold: u32) -> Self {
        Self {
            threshold,
            last_failure: None,
            consecutive_failures: 0,
            aborted: false,
            label,
            pending_call: None,
        }
    }

    fn on_tool_executing(&mut self, name: &str, arguments: &str) {
        self.pending_call = Some((name.to_string(), arguments.to_string()));
    }

    fn on_tool_result(&mut self, _name: &str, result: &str) {
        let Some(call) = self.pending_call.take() else {
            return;
        };
        let is_error = result.starts_with("Error");
        if is_error {
            if self.last_failure.as_ref() == Some(&call) {
                self.consecutive_failures += 1;
            } else {
                self.last_failure = Some(call);
                self.consecutive_failures = 1;
            }
            if self.consecutive_failures >= self.threshold {
                warn!(
                    "[{}] Stuck loop detected: same failing tool call repeated {} times, aborting",
                    self.label, self.consecutive_failures
                );
                self.aborted = true;
            }
        } else {
            // Successful call resets the counter.
            self.last_failure = None;
            self.consecutive_failures = 0;
        }
    }

    fn should_stop(&self) -> bool {
        self.aborted
    }
}

// ── Event handler ────────────────────────────────────────────────────

fn agent_event_handler<'a>(
    label: &'a str,
    detector: &'a Mutex<StuckLoopDetector>,
) -> impl EventHandler + 'a {
    EventObserver::new(move |event: &HarnessEvent<'_>| match event {
        HarnessEvent::RoundStart {
            round, max_rounds, ..
        } => {
            info!("[{label}] Round {round}/{max_rounds}");
        }
        HarnessEvent::Text(text) => {
            info!("[{label}] Response: {text}");
        }
        HarnessEvent::Reasoning(text) => {
            info!("[{label}] Reasoning: {text}");
        }
        HarnessEvent::ToolCallsReceived { count, round } => {
            info!("[{label}] {count} tool call(s) in round {round}");
        }
        HarnessEvent::ToolExecuting { name, arguments } => {
            let args_preview: String = arguments.chars().take(200).collect();
            info!(
                "[{label}] Tool call: {name}({args_preview}{})",
                if arguments.len() > 200 { "..." } else { "" }
            );
            detector.lock().unwrap().on_tool_executing(name, arguments);
        }
        HarnessEvent::ToolResult { name, result, .. } => {
            let preview: String = result.chars().take(300).collect();
            info!(
                "[{label}] Tool result {name}: {preview}{}",
                if result.len() > 300 { "..." } else { "" }
            );
            detector.lock().unwrap().on_tool_result(name, result);
        }
        HarnessEvent::Finished => {
            info!("[{label}] Finished (no more tool calls)");
        }
        HarnessEvent::RoundLimitReached { max_rounds } => {
            info!("[{label}] Hit round limit ({max_rounds})");
        }
        HarnessEvent::EmptyResponse {
            attempt,
            max_retries,
            ..
        } => {
            warn!("[{label}] Empty response, retrying ({attempt}/{max_retries})");
        }
        _ => {}
    })
}

// ── Build analyst tools ──────────────────────────────────────────────

fn build_analyst_tools(workdir: &str, result: Arc<Mutex<Option<SubmitAnalysisArgs>>>) -> ToolSet {
    let submit = FnTool::new(
        ToolDef::new(
            "submit_analysis",
            "Submit your final analysis to the boss agent. Call this ONCE when you have \
             thoroughly investigated the question and are ready to give your answer. Include \
             all key points as distinct items.",
            json_schema_for::<SubmitAnalysisArgs>(),
        ),
        move |args: SubmitAnalysisArgs| {
            let result = Arc::clone(&result);
            async move {
                *result.lock().unwrap() = Some(args);
                "Analysis submitted successfully. You are done.".to_string()
            }
        },
    );

    ToolSet::new()
        .with_common_tools(workdir)
        .with(submit)
        .with_max_result_bytes(50_000)
}

fn build_review_tools(result: Arc<Mutex<Option<SubmitReviewArgs>>>) -> ToolSet {
    let submit = FnTool::new(
        ToolDef::new(
            "submit_review",
            "Submit your review of another agent's analysis. Indicate agreement/disagreement \
             on each point and provide counter-arguments for disagreements.",
            json_schema_for::<SubmitReviewArgs>(),
        ),
        move |args: SubmitReviewArgs| {
            let result = Arc::clone(&result);
            async move {
                *result.lock().unwrap() = Some(args);
                "Review submitted successfully. You are done.".to_string()
            }
        },
    );

    // Review phase: no code tools needed, just the submit tool + think tool.
    ToolSet::new().with(submit)
}

// ── Analyst agent: initial analysis ──────────────────────────────────

struct AnalystParams<'a> {
    client: &'a OpenRouterClient,
    agent_id: usize,
    model: &'a str,
    question: &'a str,
    workdir: &'a str,
    max_tokens: u32,
    temperature: f32,
    agent_max_rounds: u32,
    max_repeated_failures: u32,
    metrics: &'a Mutex<SessionMetrics>,
}

async fn run_analyst(params: AnalystParams<'_>) -> Result<AgentAnalysis, String> {
    let AnalystParams {
        client,
        agent_id,
        model,
        question,
        workdir,
        max_tokens,
        temperature,
        agent_max_rounds,
        max_repeated_failures,
        metrics,
    } = params;
    info!(agent_id, model, "Starting analyst initial analysis");

    let result_slot: Arc<Mutex<Option<SubmitAnalysisArgs>>> = Arc::new(Mutex::new(None));
    let tools = build_analyst_tools(workdir, Arc::clone(&result_slot));

    let system_prompt = format!(
        "You are Analyst Agent #{agent_id}. You are part of a deliberation council.\n\n\
         Your task is to thoroughly analyze the following question by examining the source code \
         in the working directory. Use the available tools (read_file, grep, list_dir, find_files) \
         to investigate the codebase.\n\n\
         When you have completed your analysis, call the `submit_analysis` tool with:\n\
         - Your complete analysis\n\
         - A list of distinct, numbered key points (one clear claim per point)\n\
         - Your confidence level (0.0-1.0)\n\n\
         Be thorough but precise. Each key point should be a falsifiable claim that other \
         agents can agree or disagree with.\n\n\
         IMPORTANT: You MUST call submit_analysis to deliver your answer. Do not just \
         output text -- use the tool."
    );

    let config = HarnessConfig {
        plan_execute: HarnessPlanExecuteConfig::disabled(),
        session: HarnessSessionConfig::disabled(),
        cache: HarnessCacheConfig::disabled(),
        eviction: HarnessEvictionConfig::disabled(),
        summarizer: HarnessSummarizerConfig::disabled(),
        memory_prompt: None,
        ..HarnessConfig::new(model, &system_prompt)
    }
    .with_max_rounds(agent_max_rounds)
    .with_max_tokens(max_tokens)
    .with_temperature(temperature);

    let messages = vec![
        Message::system(&system_prompt),
        Message::user(format!("Question: {question}")),
    ];

    let label = format!("Agent#{agent_id} {model}");
    let detector = Mutex::new(StuckLoopDetector::new(label.clone(), max_repeated_failures));
    let handler = agent_event_handler(&label, &detector);
    let harness_result = Harness::new(client, &tools, config)
        .with_event_handler(&handler)
        .with_stop_signal(|| detector.lock().unwrap().should_stop())
        .run(messages)
        .await?;

    metrics.lock().unwrap().record(
        model,
        harness_result.total_prompt_tokens,
        harness_result.total_completion_tokens,
        0, // Initial analysis: no prior context, no KV cache reuse.
    );

    debug!(
        agent_id,
        model,
        rounds = harness_result.rounds_used,
        tokens = harness_result.total_tokens(),
        cost = harness_result.estimated_cost_usd,
        "Analyst finished initial analysis"
    );

    let aborted = detector.lock().unwrap().aborted;
    if aborted {
        return Err(format!(
            "Agent #{agent_id} ({model}) aborted: stuck in a loop"
        ));
    }

    let submitted = result_slot
        .lock()
        .unwrap()
        .take()
        .ok_or_else(|| format!("Agent #{agent_id} ({model}) did not call submit_analysis"))?;

    Ok(AgentAnalysis {
        agent_id,
        model: model.to_string(),
        analysis: submitted.analysis,
        key_points: submitted.key_points,
        confidence: submitted.confidence,
    })
}

// ── Analyst agent: review another agent's analysis ───────────────────

struct ReviewParams<'a> {
    client: &'a OpenRouterClient,
    reviewer_id: usize,
    reviewer_model: &'a str,
    reviewed: &'a AgentAnalysis,
    prior_context: &'a [Message],
    max_tokens: u32,
    temperature: f32,
    max_repeated_failures: u32,
    metrics: &'a Mutex<SessionMetrics>,
}

async fn run_review(params: ReviewParams<'_>) -> Result<(AgentReview, Vec<Message>), String> {
    let ReviewParams {
        client,
        reviewer_id,
        reviewer_model,
        reviewed,
        prior_context,
        max_tokens,
        temperature,
        max_repeated_failures,
        metrics,
    } = params;
    info!(
        reviewer_id,
        reviewed_id = reviewed.agent_id,
        "Starting review"
    );

    let result_slot: Arc<Mutex<Option<SubmitReviewArgs>>> = Arc::new(Mutex::new(None));
    let tools = build_review_tools(Arc::clone(&result_slot));

    let system_prompt = format!(
        "You are Analyst Agent #{reviewer_id}. You previously analyzed a question and \
         submitted your analysis. Now you are reviewing another agent's analysis.\n\n\
         Review the analysis carefully. For each key point, decide if you agree or disagree.\n\
         If you disagree, provide a specific counter-argument.\n\
         If you have new insights prompted by their analysis, include them.\n\n\
         Call `submit_review` with your assessment. You MUST use the tool to submit."
    );

    // Build messages: start with the prior context (preserves KV cache),
    // then append the new review request.
    let mut messages = prior_context.to_vec();

    // Ensure system message is present at the start.
    if messages.is_empty() || messages[0].role != cinch_rs::MessageRole::System {
        messages.insert(0, Message::system(&system_prompt));
    }

    let points_formatted = reviewed
        .key_points
        .iter()
        .enumerate()
        .map(|(i, p)| format!("  {}. {p}", i + 1))
        .collect::<Vec<_>>()
        .join("\n");

    let review_prompt = format!(
        "Please review the following analysis from Agent #{agent_id} ({model}):\n\n\
         Analysis:\n{analysis}\n\n\
         Key Points:\n{points}\n\n\
         Confidence: {confidence:.0}%\n\n\
         Call submit_review with your assessment of each point.",
        agent_id = reviewed.agent_id,
        model = reviewed.model,
        analysis = reviewed.analysis,
        points = points_formatted,
        confidence = reviewed.confidence * 100.0,
    );
    messages.push(Message::user(&review_prompt));

    let config = HarnessConfig {
        plan_execute: HarnessPlanExecuteConfig::disabled(),
        session: HarnessSessionConfig::disabled(),
        cache: HarnessCacheConfig::disabled(),
        eviction: HarnessEvictionConfig::disabled(),
        summarizer: HarnessSummarizerConfig::disabled(),
        memory_prompt: None,
        ..HarnessConfig::new(reviewer_model, &system_prompt)
    }
    .with_max_rounds(3)
    .with_max_tokens(max_tokens)
    .with_temperature(temperature);

    let label = format!(
        "Agent#{reviewer_id} {reviewer_model}->#{}",
        reviewed.agent_id
    );
    let detector = Mutex::new(StuckLoopDetector::new(label.clone(), max_repeated_failures));
    let handler = agent_event_handler(&label, &detector);
    let harness_result = Harness::new(client, &tools, config)
        .with_event_handler(&handler)
        .with_stop_signal(|| detector.lock().unwrap().should_stop())
        .run(messages.clone())
        .await?;

    // Estimate KV cache reuse: prior_context chars / 4 approximates tokens
    // that form a cacheable prefix the provider doesn't need to recompute.
    let prior_context_chars: usize = prior_context
        .iter()
        .map(|m| m.content.as_ref().map_or(0, |c| c.len()))
        .sum();
    let kv_cache_reused = (prior_context_chars / 4) as u64;

    metrics.lock().unwrap().record(
        reviewer_model,
        harness_result.total_prompt_tokens,
        harness_result.total_completion_tokens,
        kv_cache_reused,
    );

    debug!(
        reviewer_id,
        reviewed_id = reviewed.agent_id,
        rounds = harness_result.rounds_used,
        tokens = harness_result.total_tokens(),
        "Review complete"
    );

    if detector.lock().unwrap().aborted {
        return Err(format!(
            "Agent #{reviewer_id} ({reviewer_model}) aborted reviewing #{}: stuck in a loop",
            reviewed.agent_id
        ));
    }

    let submitted = result_slot.lock().unwrap().take().ok_or_else(|| {
        format!("Agent #{reviewer_id} ({reviewer_model}) did not call submit_review")
    })?;

    let review = AgentReview {
        reviewer_id,
        reviewed_id: reviewed.agent_id,
        agrees: submitted.agrees,
        response: submitted.response,
        agreed_points: submitted.agreed_points,
        disagreements: submitted.disagreements,
        new_points: submitted.new_points,
    };

    // Return updated message history for KV cache continuity.
    let updated_context = harness_result.messages;
    // Keep the full history so subsequent reviews build on it.
    Ok((review, updated_context))
}

// ── Consensus detection ──────────────────────────────────────────────

#[derive(Debug)]
struct ConsensusReport {
    unanimous: Vec<String>,
    plurality: Vec<(String, usize, usize)>, // (point, agree_count, total)
    minority: Vec<(String, usize, usize)>,
    all_agree: bool,
}

fn check_consensus(
    analyses: &[AgentAnalysis],
    reviews: &[AgentReview],
    agent_count: usize,
) -> ConsensusReport {
    // Collect all unique points across all agents.
    let mut all_points: Vec<(usize, usize, String)> = Vec::new(); // (agent_id, point_idx, text)
    for a in analyses {
        for (i, p) in a.key_points.iter().enumerate() {
            all_points.push((a.agent_id, i + 1, p.clone()));
        }
    }

    // For each point from each agent, count how many other agents agreed.
    let mut unanimous = Vec::new();
    let mut plurality = Vec::new();
    let mut minority = Vec::new();

    for (agent_id, point_idx, point_text) in &all_points {
        let relevant_reviews: Vec<&AgentReview> = reviews
            .iter()
            .filter(|r| r.reviewed_id == *agent_id)
            .collect();

        let agree_count = relevant_reviews
            .iter()
            .filter(|r| r.agreed_points.contains(point_idx))
            .count();

        // +1 for the original author
        let total_supporters = agree_count + 1;

        if total_supporters == agent_count {
            unanimous.push(point_text.clone());
        } else if total_supporters > agent_count / 2 {
            plurality.push((point_text.clone(), total_supporters, agent_count));
        } else {
            minority.push((point_text.clone(), total_supporters, agent_count));
        }
    }

    let all_agree = plurality.is_empty() && minority.is_empty();

    ConsensusReport {
        unanimous,
        plurality,
        minority,
        all_agree,
    }
}

// ── Boss agent: compile final dossier ────────────────────────────────

struct DossierParams<'a> {
    client: &'a OpenRouterClient,
    boss_model: &'a str,
    question: &'a str,
    analyses: &'a [AgentAnalysis],
    all_reviews: &'a [Vec<AgentReview>],
    consensus: &'a ConsensusReport,
    max_tokens: u32,
    max_repeated_failures: u32,
    metrics: &'a Mutex<SessionMetrics>,
}

async fn compile_dossier(params: DossierParams<'_>) -> Result<String, String> {
    let DossierParams {
        client,
        boss_model,
        question,
        analyses,
        all_reviews,
        consensus,
        max_tokens,
        max_repeated_failures,
        metrics,
    } = params;
    info!("Boss compiling final dossier");

    let system_prompt = "You are the Boss Agent compiling a final dossier from a multi-agent \
         deliberation. Produce a comprehensive, well-structured report that synthesizes \
         all findings. Use markdown formatting. Be thorough and precise.";

    let mut context = String::new();
    context.push_str(&format!("# Question\n\n{question}\n\n"));

    context.push_str("# Agent Analyses\n\n");
    for a in analyses {
        context.push_str(&format!(
            "## Agent #{} ({}), confidence: {:.0}%\n\n{}\n\nKey Points:\n",
            a.agent_id,
            a.model,
            a.confidence * 100.0,
            a.analysis,
        ));
        for (i, p) in a.key_points.iter().enumerate() {
            context.push_str(&format!("{}. {p}\n", i + 1));
        }
        context.push('\n');
    }

    context.push_str("# Review Rounds\n\n");
    for (round_idx, round_reviews) in all_reviews.iter().enumerate() {
        context.push_str(&format!("## Round {}\n\n", round_idx + 1));
        for r in round_reviews {
            context.push_str(&format!(
                "Agent #{} reviewing Agent #{}:\n- Agrees overall: {}\n- Agreed points: {:?}\n",
                r.reviewer_id, r.reviewed_id, r.agrees, r.agreed_points,
            ));
            if !r.disagreements.is_empty() {
                context.push_str("- Disagreements:\n");
                for d in &r.disagreements {
                    context.push_str(&format!(
                        "  - Point {}: {}\n",
                        d.point_number, d.counter_argument
                    ));
                }
            }
            if !r.new_points.is_empty() {
                context.push_str("- New points raised:\n");
                for p in &r.new_points {
                    context.push_str(&format!("  - {p}\n"));
                }
            }
            context.push_str(&format!("- Response: {}\n\n", r.response));
        }
    }

    context.push_str("# Consensus Status\n\n");
    if !consensus.unanimous.is_empty() {
        context.push_str("## Unanimous Points\n");
        for p in &consensus.unanimous {
            context.push_str(&format!("- {p}\n"));
        }
    }
    if !consensus.plurality.is_empty() {
        context.push_str("\n## Plurality Points\n");
        for (p, supporters, total) in &consensus.plurality {
            context.push_str(&format!("- {p} ({supporters}/{total} agents)\n"));
        }
    }
    if !consensus.minority.is_empty() {
        context.push_str("\n## Minority Points\n");
        for (p, supporters, total) in &consensus.minority {
            context.push_str(&format!("- {p} ({supporters}/{total} agents)\n"));
        }
    }

    let user_prompt = format!(
        "{context}\n\n\
         ---\n\n\
         Based on the above deliberation, compile a final dossier that:\n\
         1. Starts with a clear, direct answer to the question\n\
         2. Lists all points that achieved UNANIMOUS agreement\n\
         3. Lists points with PLURALITY support, noting the level of agreement\n\
         4. Lists MINORITY positions with the reasoning behind them\n\
         5. Provides a synthesis that weighs the evidence\n\n\
         Format as a well-structured markdown document."
    );

    let config = HarnessConfig {
        plan_execute: HarnessPlanExecuteConfig::disabled(),
        session: HarnessSessionConfig::disabled(),
        cache: HarnessCacheConfig::disabled(),
        eviction: HarnessEvictionConfig::disabled(),
        summarizer: HarnessSummarizerConfig::disabled(),
        memory_prompt: None,
        ..HarnessConfig::new(boss_model, system_prompt)
    }
    .with_max_rounds(1)
    .with_max_tokens(max_tokens * 2);

    let messages = vec![Message::system(system_prompt), Message::user(&user_prompt)];

    let boss_label = format!("Boss {boss_model}");
    let detector = Mutex::new(StuckLoopDetector::new(
        boss_label.clone(),
        max_repeated_failures,
    ));
    let handler = agent_event_handler(&boss_label, &detector);
    let tools = ToolSet::new();
    let result = Harness::new(client, &tools, config)
        .with_event_handler(&handler)
        .with_stop_signal(|| detector.lock().unwrap().should_stop())
        .run(messages)
        .await?;

    metrics.lock().unwrap().record(
        boss_model,
        result.total_prompt_tokens,
        result.total_completion_tokens,
        0, // Boss: single call, no prior context reuse.
    );

    debug!(
        tokens = result.total_tokens(),
        cost = result.estimated_cost_usd,
        "Boss dossier compiled"
    );

    Ok(result.text())
}

// ── Main ─────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing (controlled via RUST_LOG env var).
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("cabal=info")),
        )
        .with_target(true)
        .init();

    let cli = Cli::parse();

    if let Some(Command::GenerateConfig) = cli.command {
        return generate_config();
    }

    let cfg = load_config_file();
    let rc = resolve_config(cli, cfg).map_err(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    })?;
    let agent_count = rc.models.len();

    info!(
        agent_count,
        max_rounds = rc.max_rounds,
        boss_model = %rc.boss_model,
        "Starting cabal deliberation"
    );
    for (i, m) in rc.models.iter().enumerate() {
        info!(agent_id = i, model = %m, "Analyst configured");
    }

    let api_key = std::env::var("OPENROUTER_API_KEY")
        .map_err(|_| "OPENROUTER_API_KEY environment variable not set")?;
    let client = OpenRouterClient::new(&api_key)?;
    let metrics = Mutex::new(SessionMetrics::default());

    // ── Phase 1: Initial parallel analysis ───────────────────────────
    info!("Phase 1: Initial analysis");

    let mut analysis_handles = Vec::new();
    for (i, model) in rc.models.iter().enumerate() {
        let client_ref = &client;
        let question = &rc.question;
        let workdir = &rc.workdir;
        let metrics_ref = &metrics;
        let max_tokens = rc.max_tokens;
        let temperature = rc.temperature;
        let agent_max_rounds = rc.agent_max_rounds;
        let max_repeated_failures = rc.max_repeated_failures;

        analysis_handles.push(async move {
            run_analyst(AnalystParams {
                client: client_ref,
                agent_id: i,
                model,
                question,
                workdir,
                max_tokens,
                temperature,
                agent_max_rounds,
                max_repeated_failures,
                metrics: metrics_ref,
            })
            .await
        });
    }

    let analysis_results = futures::future::join_all(analysis_handles).await;
    let mut analyses: Vec<AgentAnalysis> = Vec::new();
    for result in analysis_results {
        match result {
            Ok(a) => {
                info!(
                    agent_id = a.agent_id,
                    points = a.key_points.len(),
                    confidence = a.confidence,
                    "Agent submitted analysis"
                );
                analyses.push(a);
            }
            Err(e) => {
                warn!("Agent analysis failed: {e}");
            }
        }
    }

    if analyses.len() < 2 {
        eprintln!(
            "Error: Need at least 2 successful analyses to deliberate, got {}",
            analyses.len()
        );
        std::process::exit(1);
    }

    // ── Phase 2: Deliberation rounds ─────────────────────────────────
    // Each agent reviews every other agent's analysis, one at a time.
    // Maintain per-agent message history for KV cache continuity.

    let mut agent_contexts: Vec<Vec<Message>> = (0..agent_count).map(|_| Vec::new()).collect();
    let mut all_round_reviews: Vec<Vec<AgentReview>> = Vec::new();

    for round in 0..rc.max_rounds {
        info!(round = round + 1, "Deliberation round");

        // Each reviewer processes other agents' analyses sequentially
        // (to preserve KV cache continuity per reviewer), but different
        // reviewers run in parallel.
        let mut reviewer_handles = Vec::new();
        for reviewer_idx in 0..analyses.len() {
            let reviewer = &analyses[reviewer_idx];
            let client_ref = &client;
            let analyses_ref = &analyses;
            let prior_context = &agent_contexts[reviewer_idx];
            let metrics_ref = &metrics;
            let max_tokens = rc.max_tokens;
            let temperature = rc.temperature;
            let max_repeated_failures = rc.max_repeated_failures;

            reviewer_handles.push(async move {
                let mut reviews = Vec::new();
                let mut ctx = prior_context.to_vec();

                for (reviewed_idx, reviewed) in analyses_ref.iter().enumerate() {
                    if reviewer_idx == reviewed_idx {
                        continue;
                    }
                    match run_review(ReviewParams {
                        client: client_ref,
                        reviewer_id: reviewer.agent_id,
                        reviewer_model: &reviewer.model,
                        reviewed,
                        prior_context: &ctx,
                        max_tokens,
                        temperature,
                        max_repeated_failures,
                        metrics: metrics_ref,
                    })
                    .await
                    {
                        Ok((review, updated_ctx)) => {
                            debug!(
                                reviewer = review.reviewer_id,
                                reviewed = review.reviewed_id,
                                agrees = review.agrees,
                                "Review submitted"
                            );
                            ctx = updated_ctx;
                            reviews.push(review);
                        }
                        Err(e) => {
                            warn!(
                                reviewer = reviewer_idx,
                                reviewed = reviewed_idx,
                                "Review failed: {e}"
                            );
                        }
                    }
                }
                (reviewer_idx, reviews, ctx)
            });
        }

        let reviewer_results = futures::future::join_all(reviewer_handles).await;
        let mut round_reviews: Vec<AgentReview> = Vec::new();
        for (reviewer_idx, reviews, updated_ctx) in reviewer_results {
            agent_contexts[reviewer_idx] = updated_ctx;
            round_reviews.extend(reviews);
        }

        all_round_reviews.push(round_reviews.clone());

        // Check consensus after this round.
        let all_reviews_flat: Vec<AgentReview> =
            all_round_reviews.iter().flat_map(|r| r.clone()).collect();
        let consensus = check_consensus(&analyses, &all_reviews_flat, analyses.len());

        info!(
            round = round + 1,
            unanimous = consensus.unanimous.len(),
            plurality = consensus.plurality.len(),
            minority = consensus.minority.len(),
            all_agree = consensus.all_agree,
            "Consensus check"
        );

        if consensus.all_agree {
            info!("All agents reached consensus!");
            break;
        }

        // If not last round, update analyses with new points from reviews
        // so the next round incorporates them.
        for review in &round_reviews {
            if !review.new_points.is_empty()
                && let Some(a) = analyses
                    .iter_mut()
                    .find(|a| a.agent_id == review.reviewer_id)
            {
                for p in &review.new_points {
                    if !a.key_points.contains(p) {
                        a.key_points.push(p.clone());
                    }
                }
            }
        }
    }

    // ── Phase 3: Boss compiles dossier ───────────────────────────────
    info!("Phase 3: Compiling final dossier");

    let all_reviews_flat: Vec<AgentReview> =
        all_round_reviews.iter().flat_map(|r| r.clone()).collect();
    let consensus = check_consensus(&analyses, &all_reviews_flat, analyses.len());

    let dossier = compile_dossier(DossierParams {
        client: &client,
        boss_model: &rc.boss_model,
        question: &rc.question,
        analyses: &analyses,
        all_reviews: &all_round_reviews,
        consensus: &consensus,
        max_tokens: rc.max_tokens,
        max_repeated_failures: rc.max_repeated_failures,
        metrics: &metrics,
    })
    .await?;

    // ── Output ───────────────────────────────────────────────────────
    println!("\n{}", "=".repeat(80));
    println!("CABAL DELIBERATION DOSSIER");
    println!("{}\n", "=".repeat(80));
    println!("{dossier}");

    // ── Emit metrics (always recorded, logged at debug level) ────────
    metrics.lock().unwrap().emit_logs();

    Ok(())
}
