# Awesome LLM Quantitative Trading Papers

<div align="center">

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![GitHub stars](https://img.shields.io/github/stars/Tom-roujiang/Awesome-LLM-Quantitative-Trading-Papers)](https://github.com/Tom-roujiang/Awesome-LLM-Quantitative-Trading-Papers)

</div>

A curated list of research papers, benchmarks, tools, and resources about large language models for quantitative trading and investment research.

## 📋 Contents

- [🤖 Trading Agents](#trading-agents)
- [📊 Financial Benchmarks](#financial-benchmarks)
- [📈 Arenas](#arenas)
- [🔥 LLM Post-Training](#llm-post-training)
- [💲 Stock Prediction](#stock-prediction)
- [📄 Factor Mining](#factor-mining)
- [☀️ Forecasting](#forecasting)
- [📚 Surveys](#surveys)
- [🎯 Curation Policy](#curation-policy)
- [🤝 Contributing](#contributing)
- [📝 License](#license)

## 🤖 Trading Agents

- CryptoTrade: A Reflective LLM-based Agent to Guide Zero-shot Cryptocurrency Trading (NUS, EMNLP 2024). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.09546)
- ContestTrade: A Multi-Agent Trading System Based on Internal Contest Mechanism (Finstep, 2025-08). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2508.00554) [![Code](https://img.shields.io/github/stars/FinStep-AI/ContestTrade.svg?style=social&label=Star)](https://github.com/FinStep-AI/ContestTrade)
- TradingAgents: Multi-Agents LLM Financial Trading Framework (UCLA, MIT, Tauric Research, 2025-06). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2412.20138)
- AlphaAgents: Large Language Model based Multi-Agents for Equity Portfolio Constructions (2025-08). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2508.11152)
- QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading (SBU, 2025-09). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2509.09995) [![Code](https://img.shields.io/github/stars/Y-Research-SBU/QuantAgent.svg?style=social&label=Star)](https://github.com/Y-Research-SBU/QuantAgent)
- Trade in Minutes! Rationality-Driven Agentic System for Quantitative Financial Trading (TJU, MSRA, ICLR 2026). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.04787)
- TradeTrap: Are LLM-based Trading Agents Truly Reliable and Faithful? (Shanghai AI Lab, 2025-12). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.02261) [![Code](https://img.shields.io/github/stars/Yanlewen/TradeTrap.svg?style=social&label=Star)](https://github.com/Yanlewen/TradeTrap)
- AlphaCrafter: A Full-Stack Multi-Agent Framework for Cross-Sectional Quantitative Trading (NJU, 2026-05). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.05580)

## 📊 Financial Benchmarks

- Can LLM-based Financial Investing Strategies Outperform the Market in Long Run? (University of Edinburgh, KDD 2026). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.07078)
- FCMR: Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning (Hanyang University, ACL 2025). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2412.12567)
- FINMME: Benchmark Dataset for Financial Multi-Modal Reasoning Evaluation (PKU, 2025-05). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.24714)
- FinMMR: Make Financial Numerical Reasoning More Multimodal, Comprehensive, and Challenging (2025-08). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.04625v1)
- FinRAGBench-V: A Benchmark for Multimodal RAG with Visual Citation in the Financial Domain (2025-05). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.17471)
- FinSearchComp: Towards a Realistic, Expert-Level Evaluation of Financial Search and Reasoning (ByteDance Seed, 2025-09). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2509.13160) [![Project Page](https://img.shields.io/badge/Project_Page-00CED1)](https://randomtutu.github.io/FinSearchComp/) [![Dataset](https://img.shields.io/badge/Hugging_Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/ByteSeedXpert/FinSearchComp)
- FinDeepResearch: Evaluating Deep Research Agents in Rigorous Financial Analysis (NUS, 2025-10). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.13936)
- FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol (Qwen Dianjin team, 2026-03). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2603.24943) [![Code](https://img.shields.io/github/stars/aliyun/qwen-dianjin.svg?style=social&label=Star)](https://github.com/aliyun/qwen-dianjin)
- PHANTOM: A Benchmark for Hallucination Detection in Financial Long-Context QA (Goldman Sachs, NeurIPS 2025). [![Paper](https://img.shields.io/badge/OpenReview-8B1A1A)](https://openreview.net/pdf?id=5YQAo0S3Hm)
- AlphaForgeBench: Benchmarking End-to-End Trading Strategy Design with Large Language Models (NTU, HKUST, 2026-02). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.18481)
- QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies (Lime, 2026-04). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2604.15151) [![Code](https://img.shields.io/github/stars/LimexAILab/QuantCode-Bench.svg?style=social&label=Star)](https://github.com/LimexAILab/QuantCode-Bench)

## 📈 Arenas

- DeepFund: Will LLM be Professional at Fund Investment? A Live Arena Perspective (HKUST, NeurIPS 2025). [![Code](https://img.shields.io/github/stars/HKUSTDial/DeepFund.svg?style=social&label=Star)](https://github.com/HKUSTDial/DeepFund)
- AI-Trader: Can AI Beat the Market? (HKU, 2025-12). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.10971) [![Code](https://img.shields.io/github/stars/HKUDS/AI-Trader.svg?style=social&label=Star)](https://github.com/HKUDS/AI-Trader) 

## 🔥 LLM Post-Training

- Xitadel-QuantBench: A Trading Benchmark Scored Against the Best Human Competition Strategy on the Same Unseen Day (SimReal, 2026-09). [![Code](https://img.shields.io/github/stars/Simreal-AI/Xitadel-QuantBench.svg?style=social&label=Star)](https://github.com/Simreal-AI/Xitadel-QuantBench) 
- MM-DREX: Multimodal-Driven Dynamic Routing of LLM Experts for Financial Trading (ZJU, CityU, 2025-09). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.05080)
- Trading-R1: Financial Trading with LLM Reasoning via Reinforcement Learning (UCLA, UW, Stanford, Tauric Research, 2025-09). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2509.11420)
- RETuning: Upgrading Inference-Time Scaling for Stock Movement Prediction with Large Language Models (HKUST, Hithink Research, IDEA, 2025-10). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2510.21604) [![Code](https://img.shields.io/github/stars/LinXueyuanStdio/RETuning.svg?style=social&label=Star)](https://github.com/LinXueyuanStdio/RETuning)
- AlphaQuanter: An End-to-End Tool-Orchestrated Agentic Reinforcement Learning Framework for Stock Trading (HKUST, 2025-10). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.14264) [![Code](https://img.shields.io/github/stars/horizon-llm/AlphaQuanter.svg?style=social&label=Star)](https://github.com/horizon-llm/AlphaQuanter)
- Alpha-R1: Alpha Screening with LLM Reasoning via Reinforcement Learning (Finstep, SJTU, 2025-12). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2512.23515) [![Code](https://img.shields.io/github/stars/FinStep-AI/Alpha-R1.svg?style=social&label=Star)](https://github.com/FinStep-AI/Alpha-R1)
- Janus-Q: End-to-End Event-Driven Trading via Hierarchical-Gated Reward Modeling (HKUST, 2026-02). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/html/2602.19919v1)

## 💲 Stock Prediction

- Exploring the Synergy of Quantitative Factors and Newsflow Representations from Large Language Models for Stock Return Prediction (RAM, 2025-11). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.15691)
- StockMem: An Event-Reflection Memory Framework for Stock Forecasting (SUFE, 2025-12). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2512.02720)

## 📄 Factor Mining

- LLMFactor: Extracting Profitable Factors through Prompts for Explainable Stock Movement Prediction (The University of Tokyo, 2024-06). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2406.10811)
- R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization (CMU, MSRA, NeurIPS 2025). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.15155) [![Code](https://img.shields.io/github/stars/microsoft/RD-Agent.svg?style=social&label=Star)](https://github.com/microsoft/RD-Agent)
- Alpha-GPT: Human-AI Interactive Alpha Mining for Quantitative Investment (HKUST, 2025-09). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2308.00016)
- FactorMAD: A Multi-Agent Debate Framework Based on Large Language Models for Interpretable Stock Alpha Factor Mining (THU, ICAIF 2025). [![Paper](https://img.shields.io/badge/Paper-6A5ACD)](https://dl.acm.org/doi/10.1145/3768292.3770377)
- QuantaAlpha: LLM-Driven Self-Evolving Framework for Factor Mining (SUFE, 2026-02). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2602.07085) [![Code](https://img.shields.io/github/stars/QuantaAlpha/QuantaAlpha.svg?style=social&label=Star)](https://github.com/QuantaAlpha/QuantaAlpha)
- FactorMiner: A Self-Evolving Agent with Skills and Experience Memory for Financial Alpha Discovery (THU, 2026-02). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2602.14670)
- Cognitive Alpha Mining via LLM-Driven Code-Based Evolution (HKU, GIM, ACL 2026). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.18850)
- AlphaAgentEvo: Evolution-Oriented Alpha Mining via Self-Evolving Agentic Reinforcement Learning (SYSU, NTU, ICLR 2026). [![Paper](https://img.shields.io/badge/OpenReview-8B1A1A)](https://openreview.net/pdf?id=lNmZrawUMu)

## ☀️ Forecasting

- FutureX: An Advanced Live Benchmark for LLM Agents in Future Prediction (Bytedance Seed, 2025-09). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2508.11987) [![Project Page](https://img.shields.io/badge/Project_Page-00CED1)](https://futurex-ai.github.io)
- AIA Forecaster: Technical Report (Bridgewater AIA Research, 2025-11). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.07678)
- FinDeepForecast: A Live Multi-Agent System for Benchmarking Deep Research Agents in Financial Forecasting (NUS, 2026-01). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2601.05039) [![Project Page](https://img.shields.io/badge/Project_Page-00CED1)](https://openfinarena.com)

## 📚 Surveys

- From Deep Learning to LLMs: A Survey of AI in Quantitative Investment (HKUST, 2025-03). [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2503.21422v1)
- LLMs for Quantitative Investment Research: A Practitioner's Guide (UCL, DWS, 2025-12). [![Paper](https://img.shields.io/badge/Paper-6A5ACD)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5934015)

## 🎯 Curation Policy

This list focuses on resources where large language models, multimodal language models, or LLM-based agents are used for quantitative trading, investment research, portfolio construction, forecasting, alpha discovery, financial benchmarks, or evaluation infrastructure. Included resources should be papers, preprints, technical reports, benchmarks, datasets, frameworks, arenas, open-source tools, surveys, or practitioner guides with a clear connection to this scope.

General finance resources without a meaningful LLM component, generic LLM resources without a quantitative finance angle, marketing pages, paid products, affiliate links, unverified claims, and short news posts are out of scope.

## 🤝 Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening an issue or pull request. New entries should fit the curation policy, use accurate metadata, and keep Markdown links in the same style as the existing list, such as `[![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](...)` or `[![Code](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)](...)`.

## 📝 License

This project is licensed under the [Creative Commons Attribution 4.0 International](LICENSE) license.
