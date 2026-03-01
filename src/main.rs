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
  Repeat -m to add more agents to the deliberation."
)]
struct Cli {
    /// The question or task to deliberate on.
    #[arg(short, long)]
    question: String,

    /// Models to use for analyst agents (one per agent).
    /// Example: -m z-ai/glm-5 -m openai/gpt-5.3-codex -m anthropic/claude-opus-4.6
    #[arg(short, long = "model", required = true)]
    models: Vec<String>,

    /// Model for the boss agent that compiles the final dossier.
    #[arg(long, default_value = "anthropic/claude-sonnet-4")]
    boss_model: String,

    /// Working directory for source code tools.
    #[arg(short, long, default_value = ".")]
    workdir: String,

    /// Maximum deliberation rounds (initial analysis + review rounds).
    #[arg(long, default_value_t = 5)]
    max_rounds: u32,

    /// Max tokens per LLM response.
    #[arg(long, default_value_t = 4096)]
    max_tokens: u32,

    /// Temperature for analyst agents.
    #[arg(long, default_value_t = 0.3)]
    temperature: f32,

    /// Max tool-use rounds per analyst harness invocation.
    #[arg(long, default_value_t = 15)]
    agent_max_rounds: u32,
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

    let handler = LoggingHandler;
    let harness_result = Harness::new(client, &tools, config)
        .with_event_handler(&handler)
        .run(messages)
        .await?;

    debug!(
        agent_id,
        rounds = harness_result.rounds_used,
        tokens = harness_result.total_tokens(),
        cost = harness_result.estimated_cost_usd,
        "Analyst finished initial analysis"
    );

    let submitted = result_slot
        .lock()
        .unwrap()
        .take()
        .ok_or_else(|| format!("Agent #{agent_id} did not call submit_analysis"))?;

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

    let handler = LoggingHandler;
    let harness_result = Harness::new(client, &tools, config)
        .with_event_handler(&handler)
        .run(messages.clone())
        .await?;

    debug!(
        reviewer_id,
        reviewed_id = reviewed.agent_id,
        rounds = harness_result.rounds_used,
        tokens = harness_result.total_tokens(),
        "Review complete"
    );

    let submitted = result_slot
        .lock()
        .unwrap()
        .take()
        .ok_or_else(|| format!("Agent #{reviewer_id} did not call submit_review"))?;

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

async fn compile_dossier(
    client: &OpenRouterClient,
    boss_model: &str,
    question: &str,
    analyses: &[AgentAnalysis],
    all_reviews: &[Vec<AgentReview>],
    consensus: &ConsensusReport,
    max_tokens: u32,
) -> Result<String, String> {
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

    let handler = LoggingHandler;
    let tools = ToolSet::new();
    let result = Harness::new(client, &tools, config)
        .with_event_handler(&handler)
        .run(messages)
        .await?;

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
    let agent_count = cli.models.len();

    info!(
        agent_count,
        max_rounds = cli.max_rounds,
        boss_model = %cli.boss_model,
        "Starting cabal deliberation"
    );
    for (i, m) in cli.models.iter().enumerate() {
        info!(agent_id = i, model = %m, "Analyst configured");
    }

    let api_key = std::env::var("OPENROUTER_KEY")
        .map_err(|_| "OPENROUTER_KEY environment variable not set")?;
    let client = OpenRouterClient::new(&api_key)?;

    // ── Phase 1: Initial parallel analysis ───────────────────────────
    info!("Phase 1: Initial analysis");

    let mut analysis_handles = Vec::new();
    for (i, model) in cli.models.iter().enumerate() {
        let client_ref = &client;
        let question = &cli.question;
        let workdir = &cli.workdir;
        let max_tokens = cli.max_tokens;
        let temperature = cli.temperature;
        let agent_max_rounds = cli.agent_max_rounds;

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

    for round in 0..cli.max_rounds {
        info!(round = round + 1, "Deliberation round");

        let mut round_reviews: Vec<AgentReview> = Vec::new();

        // Each agent reviews every other agent's analysis sequentially
        // (one at a time as specified), but different reviewers run in parallel.
        for reviewer_idx in 0..analyses.len() {
            let reviewer = &analyses[reviewer_idx];

            for (reviewed_idx, reviewed) in analyses.iter().enumerate() {
                if reviewer_idx == reviewed_idx {
                    continue;
                }
                let prior_context = &agent_contexts[reviewer_idx];

                match run_review(ReviewParams {
                    client: &client,
                    reviewer_id: reviewer.agent_id,
                    reviewer_model: &reviewer.model,
                    reviewed,
                    prior_context,
                    max_tokens: cli.max_tokens,
                    temperature: cli.temperature,
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
                        agent_contexts[reviewer_idx] = updated_ctx;
                        round_reviews.push(review);
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

    let dossier = compile_dossier(
        &client,
        &cli.boss_model,
        &cli.question,
        &analyses,
        &all_round_reviews,
        &consensus,
        cli.max_tokens,
    )
    .await?;

    // ── Output ───────────────────────────────────────────────────────
    println!("\n{}", "=".repeat(80));
    println!("CABAL DELIBERATION DOSSIER");
    println!("{}\n", "=".repeat(80));
    println!("{dossier}");

    Ok(())
}
