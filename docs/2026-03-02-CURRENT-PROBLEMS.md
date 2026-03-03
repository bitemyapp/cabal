It's roughly modeled on stuff like https://github.com/karpathy/llm-council and the many many many forks of it that exist, but I'm trying to accomplish a few things with my version:

- I want to tune up the prompts to match how I've been doing the same thing manually for awhile now. It's annoying and slow copying and pasting agent responses back and forth. I think I have some insights into how to make it more effective and intelligent based on what I've been doing manually.

- The OpenRouter API is kinda unreliable and it makes it annoying to get the minimum 2 agents required to drive toward consensus (there are round limits for the whole debate and for the individual inference cycles in the deliberation rounds). It doesn't require absolute consensus, if unanimity isn't achieved before the round max is hit, the boss agent just summarizes which points had unanimity, which had majority consent, and which were minority assertions.

- The OpenRouter API doesn't seem (I could be wrong) to expose the same controls and APIs that the Anthropic API exposes for controlling stuff like KV cache. I'm not actually sure their API even supports intelligent routing for inference caching. I'd hope that it did but _I don't know_. Not having inference caching makes these debates extremely expensive because you're fanning out multiple agents (3 or more, generally. I do 2 or 3 manually) and the context window keeps growing as they go through rounds of analysis and debate.

- A number of models that are purportedly pretty good for tool-calling and coding like `moonshotai/kimi-k2-thinking` and deepseek 3.2 seem to not be able to generate valid tool calls at all. I'm sure it's our fault, but I don't know why some models can handle it and some cannot.

- Some models that can generate valid tool calls initially sometimes forget to do so even when the instructions are in the context window.

- Probably because of cache busting or cache not being used at all (unclear to me at time of writing), the inference for this is very expensive. One particular test where `anthropic/claude-opus-4.6` was chugging along by itself before I had proper cohort aborts ended up costing me like $25 or more.

- Related to the inference being expensive/caching not working properly, I'd like to make this work for multiple inference APIs. Anthropic's API is extremely expensive but the caching APIs are strong. Being able to do testing with local inference strongly appeals to me for local development. I have an RTX 6000 Pro Max-Q and an RTX 5090 so it seems to me like I should be able to run a few MoE/quantized coding optimized agents concurrently on those. It might make it easier to debug and validate the KV cache functionality with a local inference API as well.

- Related to that thought, it'd be nice to support APIs that allow using subscription plan token budgets as well. I think OpenAI is friendly about this but I don't think you're supposed to use your Claude Code subscription tokens outside of Claude Code. It's might be possible to re-architect this as something more deeply integrated into Claude Code as a means of obeying the EULA and still getting to use your tokens, maybe an agent skill or something. I'd still want non-anthropic models part of the debate swarm. Model diversity is part of the point. I think one of the llm-council forks has a Claude skill integration, I think [`sherifkozman/the-llm-council`](https://github.com/sherifkozman/the-llm-council) on GitHub is the best reference for this, but again model diversity is part of the point so it isn't sufficient.

- A friend very helpfully suggested rig: [https://docs.rs/rig-core/latest/rig/#model-providers](https://docs.rs/rig-core/latest/rig/#model-providers), that might be be worth looking at and comparing to cinch-rs.
